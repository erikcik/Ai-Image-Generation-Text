# AI Image Generation Model Documentation

## Table of Contents
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Environment Setup <a name="environment-setup"></a>

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Upload your project files to Google Drive
4. In a Colab cell, run:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q diffusers transformers accelerate safetensors
!pip install -q xformers
!sudo apt -qq install git-lfs

# Change to your project directory
%cd /content/drive/MyDrive/your-project-directory

# Run the script
!python main.py --mode train  # or inference
```

### Option 2: Local Setup
1. Install dependencies:
```bash
# Basic dependencies
pip install diffusers transformers accelerate safetensors

# Optional but recommended for memory efficient training
pip install xformers

# If xformers installation fails, try:
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
# OR
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

2. Create directories:
```bash
mkdir -p babanne-images lora_output
```

3. Update config.yaml:
```yaml
drive_mount_path: "./drive"  # This will be ignored in local setup
images_dir: "./babanne-images"
lora_output_dir: "./lora_output"
# ... rest of the config ...
```

4. Place your training images in the `babanne-images` directory

5. Run the script:
```bash
python main.py --mode train  # or inference
```

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

**'NoneType' object has no attribute 'kernel' Error**
- This error occurs when running the script directly instead of in Colab
- Solution: Either:
  1. Run in Google Colab (recommended)
  2. Update config.yaml to use local paths
  
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

**Xformers Installation Error**
- Error message: "Unable to register cuFFT factory" or xformers related errors
- Solutions:
  1. Install xformers manually: `pip install xformers`
  2. If that fails, try installing with CUDA version specific index:
     ```bash
     pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
     ```
  3. If xformers cannot be installed, the model will still work but may use more memory

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