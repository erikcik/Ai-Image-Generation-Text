import os
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from data_utils import list_images, create_annotations
from safetensors.torch import save_file
from PIL import Image
import torchvision.transforms as transforms
import time
import datetime

class LaceDataset(Dataset):
    def __init__(self, image_dir, instance_prompt, tokenizer, size=512):
        """Initialize dataset with direct image loading.
        
        Args:
            image_dir: Directory containing images
            instance_prompt: Prompt to use for all images
            tokenizer: Tokenizer for text processing
            size: Image size for training
        """
        self.image_paths = list_images(image_dir)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.transform(image)
        
        # Tokenize prompt
        tokenized = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids.squeeze(0)
        }

def run(config):
    """Run LoRA training with the specified configuration."""
    print("Initializing LoRA training...")
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="wandb" if config.get("use_wandb") else None
    )
    
    # Load model components
    print(f"Loading base model: {config['pretrained_model']}")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config["pretrained_model"],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Move all pipeline components to float16 and GPU
    pipeline.to(accelerator.device, torch.float16)
    
    # Make sure both text encoders are available
    if not hasattr(pipeline, 'text_encoder_2'):
        raise ValueError("SDXL pipeline missing text_encoder_2. Make sure you're using the correct model.")
    
    # Move text encoders to device
    pipeline.text_encoder = pipeline.text_encoder.to(accelerator.device, dtype=torch.float16)
    pipeline.text_encoder_2 = pipeline.text_encoder_2.to(accelerator.device, dtype=torch.float16)
    
    # Setup LoRA configuration
    lora_config = {
        "r": int(config.get("lora_rank", 4)),
        "alpha": float(config.get("lora_alpha", 32)),
        "target_modules": [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
    }
    
    print(f"Using LoRA config: {lora_config}")
    
    # Try to enable memory efficient attention
    try:
        pipeline.unet.enable_xformers_memory_efficient_attention()
        print("Successfully enabled xformers memory efficient attention")
    except Exception as e:
        print("Warning: xformers not available. Using default attention.")
    
    # Enable gradient checkpointing
    pipeline.unet.enable_gradient_checkpointing()
    
    # Initialize LoRA weights
    print("Initializing LoRA weights...")
    lora_state_dict = initialize_lora(pipeline, config)
    
    # Configure training parameters
    trainable_params = []
    for name, param in lora_state_dict.items():
        if isinstance(param, torch.Tensor) and param.requires_grad:
            trainable_params.append(param)
    
    if not trainable_params:
        raise ValueError(f"No trainable parameters found. Modules found: {len(lora_state_dict)}")
    
    print(f"Number of trainable parameters: {len(trainable_params)}")
    
    # Initialize dataset
    dataset = LaceDataset(
        image_dir=config["images_dir"],
        instance_prompt=config["instance_prompt"],
        tokenizer=pipeline.tokenizer,
        size=int(config["resolution"])
    )
    
    print(f"Dataset initialized with {len(dataset)} images")
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True
    )
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Create optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.999),
        eps=float(config["optimizer_eps"]),
        weight_decay=1e-2
    )
    
    # Create gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Create static projection matrices once
    if config.get("text_encoder_projection", True):
        print("Initializing text projection matrices...")
        with torch.no_grad():
            main_proj = torch.nn.Linear(768, 2048, bias=False).to(
                device=accelerator.device, 
                dtype=torch.float16
            )
            # Initialize with small values
            torch.nn.init.normal_(main_proj.weight, mean=0.0, std=0.02)
    
    # Prepare for training
    pipeline.unet, optimizer, train_dataloader = accelerator.prepare(
        pipeline.unet, optimizer, train_dataloader
    )
    
    # Move VAE and text encoder to GPU and float16
    pipeline.vae = pipeline.vae.to(accelerator.device, dtype=torch.float16)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Update learning rate
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=config["max_train_steps"],
    )
    
    # Training loop
    progress_bar = tqdm(
        total=int(config["max_train_steps"]),
        desc="Training",
        disable=False,
        ncols=100  # Fixed width for cleaner display
    )
    
    global_step = 0
    total_loss = 0
    log_interval = 10  # Update stats every 10 steps
    
    pipeline.unet.train()
    start_time = time.time()
    
    try:
        for epoch in range(int(config.get("num_epochs", 1))):
            if global_step >= int(config["max_train_steps"]):
                break
                
            for batch in train_dataloader:
                if global_step >= int(config["max_train_steps"]):
                    break
                
                # Training step with trainable_params
                current_loss = train_step(
                    batch, 
                    pipeline, 
                    noise_scheduler, 
                    optimizer, 
                    main_proj, 
                    accelerator, 
                    config,
                    trainable_params,
                    scaler
                )
                
                # Update statistics
                total_loss += current_loss
                
                # Update progress bar every step
                elapsed = time.time() - start_time
                steps_per_sec = (global_step + 1) / elapsed
                remaining_steps = config["max_train_steps"] - global_step - 1
                eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'step/s': f'{steps_per_sec:.2f}',
                    'eta': f'{datetime.timedelta(seconds=int(eta))}',
                })
                progress_bar.update(1)
                
                # Log average loss periodically
                if (global_step + 1) % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    print(f"\nStep {global_step+1}/{config['max_train_steps']}, "
                          f"Average Loss: {avg_loss:.4f}, "
                          f"Speed: {steps_per_sec:.2f} steps/s")
                    total_loss = 0
                
                # Save checkpoint
                if global_step > 0 and global_step % config["save_steps"] == 0:
                    try:
                        save_path = os.path.join(config["lora_output_dir"], f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        save_lora_weights(pipeline.unet, save_path)
                        print(f"Saved valid checkpoint at step {global_step}")
                    except Exception as e:
                        print(f"Failed to save checkpoint: {str(e)}")
                
                global_step += 1
                if global_step >= config["max_train_steps"]:
                    break

    except Exception as e:
        print(f"\nTraining interrupted: {str(e)}")
        # Save checkpoint on error
        save_lora_weights(pipeline.unet, config["lora_output_dir"], None)
        raise e
    
    print("\nTraining completed successfully!")

    # And for final save:
    try:
        save_lora_weights(pipeline.unet, config["lora_output_dir"])
        print("Saved final LoRA weights")
    except Exception as e:
        print(f"Failed to save final weights: {str(e)}")
        raise

def initialize_lora(pipeline, config):
    """Initialize LoRA weights properly with Xavier initialization"""
    lora_state_dict = {}
    found_modules = 0
    
    # Keep track of processed modules to avoid recursion
    processed_modules = set()
    
    for name, module in pipeline.unet.named_modules():
        # Skip if already processed
        if name in processed_modules:
            continue
            
        # Only process direct attention modules
        if any(target in name.split(".")[-1] for target in ["to_q", "to_k", "to_v", "to_out.0"]):
            if hasattr(module, "weight"):
                try:
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                    
                    # Initialize LoRA weights with proper initialization
                    lora_down = torch.nn.Linear(in_features, config["lora_rank"], bias=False)
                    lora_up = torch.nn.Linear(config["lora_rank"], out_features, bias=False)
                    
                    # Xavier initialization
                    torch.nn.init.xavier_uniform_(lora_down.weight)
                    torch.nn.init.zeros_(lora_up.weight)
                    
                    # Move to device and set dtype
                    lora_down = lora_down.to(device=module.weight.device, dtype=torch.float16)
                    lora_up = lora_up.to(device=module.weight.device, dtype=torch.float16)
                    
                    # Store weights in state dict
                    lora_state_dict[f"{name}.lora_down.weight"] = lora_down.weight
                    lora_state_dict[f"{name}.lora_up.weight"] = lora_up.weight
                    
                    # Mark as processed
                    processed_modules.add(name)
                    found_modules += 1
                    
                    if found_modules % 10 == 0:  # Print progress less frequently
                        print(f"Initialized {found_modules} LoRA modules...")
                        
                except Exception as e:
                    print(f"Skipping module {name}: {str(e)}")
                    continue
    
    print(f"\nCompleted LoRA initialization with {found_modules} modules")
    return lora_state_dict

def save_lora_weights(pipeline, output_dir, global_step=None):
    """Extract and save LoRA weights from the UNet"""
    lora_state_dict = {}
    
    for name, module in pipeline.unet.named_modules():
        if any(target in name for target in ["to_q", "to_k", "to_v", "to_out.0"]):
            if hasattr(module, "lora_down"):
                lora_state_dict[f"{name}.lora_down.weight"] = module.lora_down.weight.detach().cpu()
                lora_state_dict[f"{name}.lora_up.weight"] = module.lora_up.weight.detach().cpu()
    
    if not lora_state_dict:
        raise ValueError("No LoRA weights found in UNet")
    
    save_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    save_file(lora_state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}")

def train_step(batch, pipeline, noise_scheduler, optimizer, main_proj, accelerator, config, trainable_params, scaler):
    """Single training step with improved stability"""
    try:
        # Move batch to GPU and convert to float16
        batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
        batch["input_ids"] = batch["input_ids"].to(accelerator.device)
        
        # Forward pass with gradient scaling
        with torch.amp.autocast('cuda'):
            # Get latents
            latents = pipeline.vae.encode(batch["pixel_values"]).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            prompt_embeds = pipeline.text_encoder(batch["input_ids"], output_hidden_states=True).hidden_states[-2]
            if prompt_embeds.shape[-1] != 2048:
                prompt_embeds = main_proj(prompt_embeds)
            
            pooled_prompt_embeds = pipeline.text_encoder_2(batch["input_ids"], output_hidden_states=True).last_hidden_state
            if pooled_prompt_embeds.ndim == 3:
                pooled_prompt_embeds = pooled_prompt_embeds.mean(dim=1)
            
            # UNet forward pass
            noise_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds.to(dtype=torch.float16),
                    "time_ids": torch.cat([
                        torch.tensor((config["resolution"], config["resolution"]), device=latents.device, dtype=torch.long),
                        torch.tensor((0, 0), device=latents.device, dtype=torch.long),
                        torch.tensor((config["resolution"], config["resolution"]), device=latents.device, dtype=torch.long),
                    ]).unsqueeze(0).repeat(latents.shape[0], 1).to(dtype=torch.float16)
                }
            ).sample
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        # Skip bad losses
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            optimizer.zero_grad()
            return float('inf')
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])
        
        # Skip step if gradients are invalid
        if grad_norm.isnan() or grad_norm.isinf():
            optimizer.zero_grad()
            return float('inf')
        
        # Update weights with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        return loss.item()
            
    except Exception as e:
        print(f"Error in training step: {str(e)}")
        optimizer.zero_grad()
        return float('inf')

def validate_checkpoint(lora_state_dict):
    """Validate LoRA weights before saving"""
    try:
        # Check if weights exist
        if not lora_state_dict:
            print("Empty LoRA state dict")
            return False
            
        # Validate each tensor
        for name, tensor in lora_state_dict.items():
            # Check for NaN or Inf
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Invalid values in {name}")
                return False
            
            # Check for zero tensors
            if torch.all(tensor == 0):
                print(f"All zeros in {name}")
                return False
                
            # Check for reasonable magnitudes
            if torch.abs(tensor).max() > 100:
                print(f"Large values in {name}")
                return False
                
        return True
    except Exception as e:
        print(f"Error validating checkpoint: {str(e)}")
        return False 