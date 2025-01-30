import os
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from data_utils import list_images
from safetensors.torch import save_file

class LaceDataset(Dataset):
    def __init__(self, annotations_file, tokenizer, size=512):
        self.image_paths = []
        self.prompts = []
        
        with open(annotations_file, 'r') as f:
            for line in f:
                path, prompt = line.strip().split('\t')
                self.image_paths.append(path)
                self.prompts.append(prompt)
        
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # TODO: Implement image loading and preprocessing
        # For now return dummy data
        return {
            "pixel_values": torch.randn(3, self.size, self.size),
            "input_ids": self.tokenizer(
                self.prompts[idx],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)
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
        torch_dtype=torch.float16
    )
    
    # Setup LoRA configuration
    lora_config = {
        "r": int(config.get("lora_rank", 4)),
        "alpha": float(config.get("lora_alpha", 32)),
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
    }
    
    # Try to enable memory efficient attention
    try:
        pipeline.unet.enable_xformers_memory_efficient_attention()
        print("Successfully enabled xformers memory efficient attention")
    except Exception as e:
        print("Warning: xformers not available. Using default attention. This may increase memory usage.")
        print("To install xformers, run: pip install xformers")
    
    # Enable gradient checkpointing for memory efficiency
    pipeline.unet.enable_gradient_checkpointing()
    
    # Initialize LoRA weights
    print("Initializing LoRA weights...")
    lora_state_dict = {}
    for name, module in pipeline.unet.named_modules():
        if any(target in name for target in lora_config["target_modules"]):
            try:
                # Get input and output features
                if hasattr(module, 'in_features'):
                    in_features = module.in_features
                    out_features = module.out_features
                else:
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                
                # Initialize LoRA weights
                lora_down = torch.zeros((lora_config["r"], in_features), requires_grad=True)
                lora_up = torch.zeros((out_features, lora_config["r"]), requires_grad=True)
                
                # Initialize with small random values
                torch.nn.init.kaiming_uniform_(lora_down)
                torch.nn.init.zeros_(lora_up)
                
                lora_state_dict[f"{name}.lora_down.weight"] = lora_down
                lora_state_dict[f"{name}.lora_up.weight"] = lora_up
                lora_state_dict[f"{name}.alpha"] = torch.tensor(lora_config["alpha"])
            except Exception as e:
                print(f"Warning: Skipping layer {name} due to error: {str(e)}")
                continue
    
    # Configure training parameters
    trainable_params = []
    for name, param in lora_state_dict.items():
        if isinstance(param, torch.Tensor) and param.requires_grad:
            trainable_params.append(param)
    
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check LoRA configuration.")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["learning_rate"]),
        weight_decay=1e-4
    )
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Prepare dataset and dataloader
    dataset = LaceDataset(
        os.path.join(config["images_dir"], "annotations.txt"),
        tokenizer=pipeline.tokenizer,
        size=int(config["resolution"])
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True
    )
    
    # Prepare for training
    pipeline.unet, optimizer, train_dataloader = accelerator.prepare(
        pipeline.unet, optimizer, train_dataloader
    )
    
    # Training loop
    progress_bar = tqdm(range(int(config["max_train_steps"])))
    global_step = 0
    
    pipeline.unet.train()
    for epoch in range(int(config.get("num_epochs", 1))):
        for batch in train_dataloader:
            if global_step >= int(config["max_train_steps"]):
                break
                
            # Forward pass
            latents = pipeline.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * 0.18215
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (latents.shape[0],),
                device=latents.device
            )
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            encoder_hidden_states = pipeline.text_encoder(batch["input_ids"])[0]
            
            noise_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states
            ).sample
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Save checkpoint periodically
            if global_step > 0 and global_step % int(config.get("save_steps", 500)) == 0:
                print(f"\nSaving checkpoint at step {global_step}")
                save_path = os.path.join(config["lora_output_dir"], f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                save_file(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.safetensors"))
            
            progress_bar.update(1)
            global_step += 1
    
    # Save final LoRA weights
    print(f"Saving final LoRA weights to {config['lora_output_dir']}")
    save_file(lora_state_dict, os.path.join(config["lora_output_dir"], "pytorch_lora_weights.safetensors"))
    print("Training completed successfully!") 