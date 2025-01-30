import os
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from data_utils import list_images
from PIL import Image
import torchvision.transforms as transforms

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
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        
        # Add SDXL-specific time IDs
        original_size = (1024, 1024)
        target_size = (self.size, self.size)
        crop_coords = (0, 0, self.size, self.size)
        
        add_time_ids = torch.tensor([
            original_size[0],  # Original image width
            original_size[1],  # Original image height
            target_size[0],    # Target image width
            target_size[1],    # Target image height
            crop_coords[0],    # Crop top
            crop_coords[1],    # Crop left
            crop_coords[2],    # Crop bottom
            crop_coords[3],    # Crop right
        ])
        
        return {
            "pixel_values": image,
            "input_ids": self.tokenizer(
                self.prompts[idx],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0),
            "time_ids": add_time_ids
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
    
    # Freeze base model and add LoRA
    pipeline.unet.add_lora_weights(
        lora_rank=config.get("lora_rank", 4),
        lora_alpha=config.get("lora_alpha", 32)
    )
    pipeline.unet.enable_lora()
    
    # Configure training parameters
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4
    )
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Prepare dataset
    dataset = LaceDataset(
        os.path.join(config["images_dir"], "annotations.txt"),
        tokenizer=pipeline.tokenizer,
        size=config["resolution"]
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        shuffle=True
    )
    
    # Prepare components with accelerator
    pipeline.unet, optimizer, train_dataloader = accelerator.prepare(
        pipeline.unet, optimizer, train_dataloader
    )
    
    # Training loop
    progress_bar = tqdm(range(config["max_train_steps"]))
    global_step = 0
    
    pipeline.unet.train()
    for epoch in range(config.get("num_epochs", 1)):
        for batch in train_dataloader:
            if global_step >= config["max_train_steps"]:
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
            
            # Add time embeddings required by SDXL
            added_cond_kwargs = {"time_ids": batch["time_ids"].to(latents.device)}
            
            noise_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress
            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            global_step += 1
            
            # Save checkpoint periodically
            if global_step % config.get("save_steps", 500) == 0:
                checkpoint_dir = os.path.join(config["lora_output_dir"], f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                pipeline.unet.save_lora_weights(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
            
    # Save final LoRA weights
    print(f"Saving LoRA weights to {config['lora_output_dir']}")
    pipeline.unet.save_lora_weights(config["lora_output_dir"])
    print("Training completed successfully!") 