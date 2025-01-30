# AI Image Generation Model Documentation

## Table of Contents
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Environment Setup <a name="environment-setup"></a>

### Google Colab Requirements
- GPU: A100 or V100 recommended
- Runtime: Python 3.10+
- Storage: Minimum 25GB free space in Google Drive

```bash
# Install required packages
!pip install -U torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
!pip install -U diffusers transformers accelerate peft safetensors wandb
```

### Google Drive Mounting
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Configuration <a name="configuration"></a>

Edit `config.yaml` with these key parameters:

```yaml
# Required Parameters
images_dir: "/content/drive/MyDrive/babanne-images"  # Path to training images
instance_prompt: "a photo of <myspecialstyle>"        # Your custom concept token

# Training Optimization
train_batch_size: 2              # Reduce if OOM errors occur
learning_rate: 1e-4              # Start with 1e-5 for small datasets
resolution: 512                   # Match your image dimensions

# Inference Parameters
num_inference_steps: 30          # 20-50 steps recommended
refiner_strength: 0.3            # 0.2-0.4 for subtle refinements
```

## Training <a name="training"></a>

### Preparing Your Dataset
- Place 200-300 JPG images in your Google Drive folder
- Images should be square aspect ratio (512x512 recommended)
- Name images consistently (e.g., `design_001.jpg`, `design_002.jpg`)

### Starting Training
```bash
python main.py --mode train
```

#### Expected Output
```
Mounting Google Drive...
Starting LoRA training...
Created annotations file at /content/drive/.../annotations.txt
Training step 100/1000 | Loss: 0.123
Saving checkpoint to /content/drive/.../lora_output/checkpoint-100
```

## Inference <a name="inference"></a>

### Generating New Designs
```bash
python main.py --mode inference
```

#### Example Output
```
Loading base model with LoRA weights...
Generating image with prompt: a photo of <myspecialstyle>
Refining generated image...
Saved refined image to /content/drive/.../lora_output/refined_output.png
```

### Custom Prompts (Advanced)
```python
# In inference.py modify:
prompt = "a close-up photo of <myspecialstyle> fabric with floral patterns"
```

## Troubleshooting <a name="troubleshooting"></a>

### Common Issues

**CUDA Out of Memory**
- Reduce `train_batch_size` in config.yaml
- Add `--mixed_precision fp16` to training command

**Missing Dependencies**
```bash
# Update packages
!pip install --upgrade diffusers transformers accelerate
```

**Model Not Loading**
- Verify Google Drive path in config.yaml
- Check available storage space
- Ensure LoRA weights exist in output directory

## Advanced Usage <a name="advanced-usage"></a>

### Multi-Concept Training
```yaml
# config.yaml
instance_prompt: "a photo of <myspecialstyle> in <anotherstyle>"
```

### Quality Refinement
```yaml
# config.yaml
refiner_steps: 40
refiner_guidance: 8.5
```

### Progress Monitoring
```bash
# Add to training command
--log_with wandb
```

> **Tip:** Use 50-100 inference steps and 0.25-0.35 refiner strength for high-quality outputs 