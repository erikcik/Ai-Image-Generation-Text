# AI Image Generation Model Documentation

## Overview
This project implements a Stable Diffusion XL-based image generation system with LoRA fine-tuning, specifically designed for lace embroidered fabric designs.

## Setup Instructions

### 1. Google Colab Environment Setup
1. Open Google Colab
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install required dependencies:
   ```bash
   pip install torch diffusers transformers accelerate wandb
   ```

### 2. Project Structure
Place your images in Google Drive (e.g., `/content/drive/MyDrive/babanne-images/`). The project consists of:
- `config.yaml`: Configuration parameters
- `main.py`: CLI entry point
- `train.py`: LoRA training implementation
- `inference.py`: Image generation and refinement
- `data_utils.py`: Image processing utilities

### 3. Configuration
Edit `config.yaml` to set your parameters:
```yaml
drive_mount_path: "/content/drive"
images_dir: "/content/drive/MyDrive/babanne-images"
lora_output_dir: "/content/drive/MyDrive/lora_output"
instance_prompt: "a photo of <myspecialstyle>"
resolution: 512
train_batch_size: 2
learning_rate: 1e-4
max_train_steps: 1000
```

## Usage Guide

### 1. Training Process
1. Prepare your training images (200-300 .jpg files) in the specified Google Drive folder
2. Run training:
   ```bash
   python main.py --mode train
   ```
3. Monitor the training progress in the output log
4. LoRA weights will be saved to `lora_output_dir`

### 2. Generating Images
1. Run inference:
   ```bash
   python main.py --mode inference
   ```
2. Generated images will be:
   - Created using the base SDXL model with LoRA weights
   - Refined using the SDXL refiner
   - Saved to the specified output directory

### 3. Advanced Configuration

#### Training Parameters
- `train_batch_size`: Adjust based on available GPU memory
- `learning_rate`: Default 1e-4, adjust if training is unstable
- `max_train_steps`: Increase for better results, decrease for faster training

#### Inference Parameters
- `num_inference_steps`: More steps = better quality but slower
- `guidance_scale`: Higher values = stronger adherence to prompt
- `refiner_strength`: Controls refinement intensity (0.0-1.0)

## Troubleshooting

### Common Issues

1. Out of Memory (OOM) Errors
   - Reduce `train_batch_size`
   - Enable gradient checkpointing
   - Use mixed precision training

2. Image Quality Issues
   - Increase `num_inference_steps`
   - Adjust `guidance_scale`
   - Try different `refiner_strength` values

3. Training Problems
   - Check image format compatibility
   - Verify prompt formatting
   - Ensure proper Google Drive mounting

### Error Messages

1. "CUDA out of memory"
   - Solution: Reduce batch size or image resolution

2. "No such file or directory"
   - Check Google Drive paths
   - Verify file permissions

3. "Failed to load LoRA weights"
   - Ensure training completed successfully
   - Check path to LoRA weights

## Best Practices

1. Image Preparation
   - Use high-quality source images
   - Maintain consistent image sizes
   - Ensure proper file formats (.jpg preferred)

2. Training
   - Start with default parameters
   - Monitor training loss
   - Save checkpoints regularly

3. Inference
   - Test different prompt variations
   - Experiment with guidance scales
   - Use batch processing for multiple images

## Performance Optimization

1. Training Speed
   - Use mixed precision training
   - Enable gradient checkpointing
   - Optimize batch size

2. Memory Usage
   - Monitor GPU memory
   - Clean cache between generations
   - Use appropriate precision settings

## Support and Resources

- Diffusers Documentation: [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- SDXL Documentation: [Stability AI](https://stability.ai/stable-diffusion)
- LoRA Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 