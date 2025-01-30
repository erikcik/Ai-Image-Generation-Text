import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import logging

def run(config):
    """Run inference and refinement with trained LoRA weights."""
    try:
        # Initialize base pipeline with LoRA
        print("Loading base model with LoRA weights...")
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            config["pretrained_model"],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # Load LoRA weights
        base_pipe.load_lora_weights(config["lora_output_dir"])
        
        # Generate initial image
        print(f"Generating image with prompt: {config['instance_prompt']}")
        initial_image = base_pipe(
            prompt=config["instance_prompt"],
            num_inference_steps=config.get("num_inference_steps", 30),
            guidance_scale=config.get("guidance_scale", 7.5),
            output_type="latent"
        ).images[0]
        
        # Initialize refiner pipeline
        print("Loading refiner model...")
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            config["refiner_model"],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # Refine the image
        print("Refining generated image...")
        refined_image = refiner_pipe(
            prompt=config["instance_prompt"],
            image=initial_image,
            num_inference_steps=config.get("refiner_steps", 20),
            strength=config.get("refiner_strength", 0.3),
            guidance_scale=config.get("refiner_guidance", 7.5)
        ).images[0]
        
        # Save output
        output_path = f"{config['lora_output_dir']}/refined_output.png"
        refined_image.save(output_path)
        print(f"Saved refined image to {output_path}")
        
        # Cleanup
        del base_pipe
        del refiner_pipe
        torch.cuda.empty_cache()
        
        return output_path
        
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")
        raise 