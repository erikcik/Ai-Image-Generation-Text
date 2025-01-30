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
    lora_state_dict = {}
    found_modules = 0
    
    for name, module in pipeline.unet.named_modules():
        if any(target in name for target in lora_config["target_modules"]):
            try:
                if hasattr(module, 'in_features'):
                    in_features = module.in_features
                    out_features = module.out_features
                elif hasattr(module, 'weight'):
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                else:
                    continue
                
                print(f"Adding LoRA to layer: {name} ({in_features} -> {out_features})")
                
                # Initialize LoRA weights in float16
                lora_down = torch.zeros((lora_config["r"], in_features), 
                                     dtype=torch.float16, 
                                     requires_grad=True)
                lora_up = torch.zeros((out_features, lora_config["r"]), 
                                    dtype=torch.float16,
                                    requires_grad=True)
                
                # Initialize with small random values
                torch.nn.init.kaiming_uniform_(lora_down)
                torch.nn.init.zeros_(lora_up)
                
                lora_state_dict[f"{name}.lora_down.weight"] = lora_down
                lora_state_dict[f"{name}.lora_up.weight"] = lora_up
                lora_state_dict[f"{name}.alpha"] = torch.tensor(lora_config["alpha"], 
                                                              dtype=torch.float16)
                
                found_modules += 1
            except Exception as e:
                print(f"Warning: Error initializing LoRA for {name}: {str(e)}")
                continue
    
    print(f"Found and initialized {found_modules} modules for LoRA training")
    
    if found_modules == 0:
        raise ValueError("No suitable attention modules found for LoRA training")
    
    # Configure training parameters
    trainable_params = []
    for name, param in lora_state_dict.items():
        if isinstance(param, torch.Tensor) and param.requires_grad:
            trainable_params.append(param)
    
    if not trainable_params:
        raise ValueError(f"No trainable parameters found. Modules found: {found_modules}")
    
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
    
    # Create optimizer with correct dtype and gradient clipping
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["learning_rate"]),
        weight_decay=1e-4,
        eps=1e-8,  # Add epsilon for numerical stability
    )
    
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
    
    # Training loop
    progress_bar = tqdm(range(int(config["max_train_steps"])))
    global_step = 0
    
    pipeline.unet.train()
    for epoch in range(int(config.get("num_epochs", 1))):
        for batch in train_dataloader:
            if global_step >= int(config["max_train_steps"]):
                break
                
            # Move batch to GPU and convert to float16
            batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
            batch["input_ids"] = batch["input_ids"].to(accelerator.device)
            
            # Forward pass
            try:
                # Debug shapes at each step
                print("\nDebug Shapes:")
                
                # 1. Encode images
                with torch.no_grad():
                    latents = pipeline.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215
                
                # 2. Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                    device=latents.device
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 3. Get text embeddings
                with torch.no_grad():
                    # Text encoder 1 (main)
                    prompt_embeds = pipeline.text_encoder(
                        batch["input_ids"],
                        output_hidden_states=True,
                        return_dict=True
                    ).hidden_states[-2]
                    
                    # Project main embeddings using the static projection
                    if prompt_embeds.shape[-1] != 2048:
                        prompt_embeds = main_proj(prompt_embeds)
                    
                    # Text encoder 2 (pooled)
                    pooled_prompt_embeds = pipeline.text_encoder_2(
                        batch["input_ids"],
                        output_hidden_states=True,
                        return_dict=True
                    ).last_hidden_state
                    
                    # Mean pool if needed
                    if pooled_prompt_embeds.ndim == 3:
                        pooled_prompt_embeds = pooled_prompt_embeds.mean(dim=1)
                
                # 4. Create time embeddings
                orig_size = (int(config["resolution"]), int(config["resolution"]))
                target_size = (int(config["resolution"]), int(config["resolution"]))
                crops_coords_top_left = (0, 0)
                
                add_time_ids = torch.cat([
                    torch.tensor(orig_size, device=latents.device, dtype=torch.long),
                    torch.tensor(crops_coords_top_left, device=latents.device, dtype=torch.long),
                    torch.tensor(target_size, device=latents.device, dtype=torch.long),
                ]).unsqueeze(0).repeat(latents.shape[0], 1)
                
                # 5. Prepare final inputs for UNet
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds.to(dtype=torch.float16),
                    "time_ids": add_time_ids.to(dtype=torch.float16)
                }
                
                # Print shapes before UNet
                print("\nFinal shapes before UNet:")
                print(f"Noisy latents: {noisy_latents.shape}")
                print(f"Timesteps: {timesteps.shape}")
                print(f"Prompt embeds: {prompt_embeds.shape}")
                print(f"Text embeds: {added_cond_kwargs['text_embeds'].shape}")
                print(f"Time ids: {added_cond_kwargs['time_ids'].shape}")
                
                # 6. UNet forward pass
                noise_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                print(f"\nPredicted noise shape: {noise_pred.shape}")
                
                # 7. Compute loss with scaling
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Check for NaN loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("\n=== Warning: NaN or Inf loss detected! ===")
                    print(f"Noise pred stats: min={noise_pred.min()}, max={noise_pred.max()}, mean={noise_pred.mean()}")
                    print(f"Target noise stats: min={noise.min()}, max={noise.max()}, mean={noise.mean()}")
                    # Skip this batch
                    continue
                
                # 8. Backward pass with gradient clipping
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                
                if global_step % config.get("print_frequency", 10) == 0:
                    print(f"\nStep {global_step}: Loss = {loss.item():.4f}")
                
            except RuntimeError as e:
                print("\n=== Error Details ===")
                print(f"Error: {str(e)}")
                print("\nDebug Information:")
                print(f"Current loss: {loss.item() if 'loss' in locals() else 'Not computed'}")
                print(f"Gradient norm: {torch.nn.utils.clip_grad_norm_(trainable_params, float('inf'))}")
                raise e
            
            # Save checkpoint
            if global_step > 0 and global_step % int(config.get("save_steps", 500)) == 0:
                print(f"\nSaving checkpoint at step {global_step}")
                save_path = os.path.join(config["lora_output_dir"], f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                save_file(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.safetensors"))
            
            progress_bar.update(1)
            global_step += 1
    
    # Save final weights
    print(f"Saving final LoRA weights to {config['lora_output_dir']}")
    save_file(lora_state_dict, os.path.join(config["lora_output_dir"], "pytorch_lora_weights.safetensors"))
    print("Training completed successfully!") 