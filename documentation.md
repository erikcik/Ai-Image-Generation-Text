# AI Image Generation Model Documentation

## Table of Contents
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Environment Setup <a name="environment-setup"></a>

### Google Colab Setup (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Run these setup commands:
```python
# Install required packages
!pip install -q diffusers transformers accelerate safetensors
!sudo apt -qq install git-lfs

# Clone your project (if using git) or upload files manually
!git clone <your-repository-url>

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### Local Setup (Alternative)
If running locally:
1. Install dependencies:
```bash
pip install diffusers transformers accelerate safetensors
```
2. Ensure you have sufficient GPU memory (16GB+ recommended)
3. Update `config.yaml` paths for local directories

## Configuration <a name="configuration"></a>

Edit `config.yaml` with these key parameters:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `images_dir` | Path to training images | `/content/drive/MyDrive/babanne-images` |
| `instance_prompt` | Unique concept token prompt | `"a photo of <myspecialstyle>"` |
| `pretrained_model` | Base SDXL model | `"stabilityai/stable-diffusion-xl-base-1.0"` |
| `train_batch_size` | Training batch size | `2` |
| `max_train_steps` | Training iterations | `1000` |

**Important:** Keep the `<myspecialstyle>` token in prompts for proper LoRA training.

## Training <a name="training"></a>

### Starting Training
```bash
python main.py --mode train
```

**Process Flow:**
1. Mounts Google Drive
2. Processes images from `images_dir`
3. Creates annotation files
4. Trains LoRA adapters
5. Saves weights to `lora_output_dir`

## Inference <a name="inference"></a>

### Generating Images
```bash
python main.py --mode inference
```

**Generation Pipeline:**
1. Loads base SDXL model with LoRA weights
2. Generates initial image
3. Refines output with SDXL Refiner
4. Saves final image to `lora_output_dir/refined_output.png`

## Troubleshooting <a name="troubleshooting"></a>

### Common Issues

**CUDA Out of Memory**
- Reduce `train_batch_size` in config
- Use `resolution: 512` instead of higher values

**Missing Dependencies**
```bash
# Update packages
!pip install --upgrade diffusers transformers accelerate
```

**Model Loading Errors**
- Verify model paths in config
- Check available disk space on Google Drive

## Advanced Usage <a name="advanced-usage"></a>

### Custom Prompts
```yaml
# In config.yaml
instance_prompt: "a detailed embroidery pattern of <myspecialstyle> on fabric"
```

### LoRA Parameters
```yaml
# Add to config.yaml for advanced control
lora_rank: 8       # Higher values increase model flexibility
lora_alpha: 64     # Scales LoRA contribution
```

### Multi-GPU Training
```python
# Launch with Accelerate
!accelerate config  # Set up distributed training
!accelerate launch train.py
```

### Output Samples
![Sample Output](https://via.placeholder.com/512x512.png?text=Embroidery+Sample) 