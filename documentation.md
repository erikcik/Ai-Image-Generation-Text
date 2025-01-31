# AI Image Generation System Documentation

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Configuration Guide](#configuration-guide)
3. [Training Process](#training-process)
4. [Image Generation](#image-generation)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

## Environment Setup <a name="environment-setup"></a>

### Requirements
- Google Colab Pro+ (Recommended)
- NVIDIA GPU (T4 or better)
- Python 3.9+
- 15GB+ Disk Space

### Installation Steps
```bash
# Install core dependencies
!pip install torch==2.0.1+cu118
!pip install diffusers==0.19.3 transformers==4.31.0 accelerate==0.21.0

# Install additional utilities
!pip install xformers wandb safetensors
```

### Google Drive Mounting
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Configuration Guide <a name="configuration-guide"></a>

### config.yaml Structure
```yaml
drive_mount_path: "/content/drive"
images_dir: "/content/drive/MyDrive/babanne-images"
lora_output_dir: "/content/drive/MyDrive/lora_output"
instance_prompt: "a photo of <myspecialstyle> lace fabric"
# ... other parameters
```

### Required Modifications
1. Set `images_dir` to your Google Drive folder containing lace images
2. Customize `instance_prompt` with your unique token
3. Adjust training parameters based on GPU capacity:
   ```yaml
   train_batch_size: 1  # Reduce if OOM errors occur
   resolution: 512      # 768 for higher quality (requires more VRAM)
   ```

## Training Process <a name="training-process"></a>

### Starting Training
```bash
python main.py --mode train
```

### Expected Output
```
Mounting Google Drive...
Starting LoRA training...
Loading base model: stabilityai/stable-diffusion-xl-base-1.0
Creating annotations for 250 images...
Step 100, Loss: 0.1245
Saved checkpoint at step 500
Training completed successfully!
```

### Monitoring Training
1. Check loss values decreasing over time
2. Verify checkpoint saving
3. Monitor GPU memory usage (nvidia-smi)

## Image Generation <a name="image-generation"></a>

### Generating New Designs
```bash
python main.py --mode inference
```

### Output Files
- `refined_output.png` in your lora_output_dir
- Multiple versions with timestamps if run repeatedly

### Custom Prompts
Modify `instance_prompt` in config.yaml:
```yaml
instance_prompt: "close-up of <myspecialstyle> lace pattern with gold threads"
```

## Troubleshooting <a name="troubleshooting"></a>

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solutions:
- Reduce batch_size in config.yaml
- Lower resolution to 512
- Enable memory optimizations:
  ```python
  pipe.enable_xformers_memory_efficient_attention()
  pipe.enable_model_cpu_offload()
  ```

**2. Missing Dependencies**
```bash
# Fix missing packages
!pip install [missing-package]
```

**3. Poor Generation Quality**
- Increase training steps (2000-5000)
- Use higher quality source images
- Experiment with different learning rates (1e-5 to 1e-4)

## Best Practices <a name="best-practices"></a>

### Training Tips
- Use 200-300 high-quality JPEG images
- Maintain consistent image dimensions
- Use descriptive prompts with unique token
- Start with 1000 training steps, increase gradually

### Generation Tips
- Try different refiner strengths (0.2-0.5)
- Experiment with guidance scales (5-15)
- Combine with negative prompts:
  ```yaml
  negative_prompt: "blurry, low quality, duplicate"
  ```

### Performance Optimization
```yaml
# config.yaml optimizations
mixed_precision: "fp16"  # For modern GPUs
gradient_checkpointing: true
use_xformers: true
```

## Support <a name="support"></a>
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LoRA Training Guide](https://huggingface.co/docs/diffusers/training/lora)
- [SDXL Technical Report](https://arxiv.org/abs/2307.01952) 