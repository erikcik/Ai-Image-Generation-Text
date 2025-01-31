import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import AutoPipelineForText2Image, DDPMScheduler
from diffusers.optimization import LoRAAttnProcs
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging
from data_utils import list_images

class LaceDataset(Dataset):
    def __init__(self, image_dir, instance_prompt, tokenizer, size=512):
        self.image_paths = list_images(image_dir)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "image": image,
            "text": self.instance_prompt.replace("<myspecialstyle>", "lace fabric")
        }

def run(config):
    """Minimal LoRA training implementation based on project overview example"""
    try:
        # Initialize accelerator
        accelerator = Accelerator(mixed_precision="fp16")
        
        # Load components
        pipeline = AutoPipelineForText2Image.from_pretrained(
            config["pretrained_model"],
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"], subfolder="tokenizer")
        
        # Setup dataset
        dataset = LaceDataset(
            config["images_dir"],
            config["instance_prompt"],
            tokenizer,
            config["resolution"]
        )
        train_dataloader = DataLoader(dataset, batch_size=config["train_batch_size"], shuffle=True)
        
        # Configure LoRA
        lora_attn_procs = LoRAAttnProcs(pipeline.unet, r=4, dropout=0.0)
        pipeline.unet.set_attn_processor(lora_attn_procs)
        
        # Training setup
        optimizer = torch.optim.AdamW(lora_attn_procs.parameters(), lr=config["learning_rate"])
        noise_scheduler = DDPMScheduler.from_pretrained(config["pretrained_model"], subfolder="scheduler")
        
        # Prepare components
        pipeline, optimizer, train_dataloader = accelerator.prepare(
            pipeline, optimizer, train_dataloader
        )

        # Training loop
        progress_bar = tqdm(range(config["max_train_steps"]))
        global_step = 0
        
        while global_step < config["max_train_steps"]:
            for batch in train_dataloader:
                if global_step >= config["max_train_steps"]:
                    break
                
                # Convert images to latents
                images = [img.resize((512,512)) for img in batch["image"]]
                pixel_values = torch.stack([
                    pipeline.feature_extractor(img, return_tensors="pt").pixel_values[0]
                    for img in images
                ]).to(accelerator.device)
                
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample() * 0.18215
                
                # Noise addition
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Text encoding
                text_inputs = tokenizer(
                    batch["text"],
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                # Forward pass
                noise_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=pipeline.text_encoder(text_inputs)[0]
                ).sample
                
                # Loss calculation
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # Update progress
                progress_bar.update(1)
                global_step += 1
                
                # Simple logging
                if global_step % 100 == 0:
                    logging.info(f"Step {global_step}, Loss: {loss.item():.4f}")
                    
                # Basic checkpoint saving
                if global_step % 500 == 0:
                    checkpoint_path = os.path.join(config["lora_output_dir"], f"checkpoint-{global_step}")
                    lora_attn_procs.save_pretrained(checkpoint_path)
                    logging.info(f"Saved checkpoint at step {global_step}")

        # Final save
        lora_attn_procs.save_pretrained(config["lora_output_dir"])
        logging.info("Training completed successfully")
        return True

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise 