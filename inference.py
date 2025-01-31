import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import logging

def run(config):
    """Minimal inference implementation based on project overview example"""
    try:
        # Load base model with LoRA
        print("Loading base model with LoRA...")
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            config["pretrained_model"],
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        base_pipe.load_lora_weights(config["lora_output_dir"])

        # Generate initial image
        print(f"Generating image with prompt: {config['instance_prompt']}")
        result = base_pipe(
            prompt=config["instance_prompt"],
            num_inference_steps=config.get("num_inference_steps", 30),
            guidance_scale=config.get("guidance_scale", 7.5),
            output_type="latent"
        )
        latents = result.images[0]

        # Load and apply refiner
        print("Applying refiner...")
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            config["refiner_model"],
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

        refined_image = refiner_pipe(
            prompt=config["instance_prompt"],
            image=latents,
            num_inference_steps=config.get("refiner_steps", 20),
            strength=config.get("refiner_strength", 0.3)
        ).images[0]

        # Save result
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