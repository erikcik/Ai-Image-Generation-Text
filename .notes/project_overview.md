```markdown
# project_overview.md

Below is the **Product Requirements Document (PRD)** for implementing the AI image generation model. This PRD will reside in `project_overview.md` within the `.notes` folder.  
A **separate file**, `documentation.md`, will be created to store all user-focused and developer-oriented documentation and tutorials (instead of placing them here).

---

## AI Image Generation Model PRD

### 1. Overview

The goal is to create an AI model that:

1. Accepts a large number of images (200–300) of lace embroidered fabric designs in **`.jpg`** format (primarily).
2. Trains using LoRA (Low-Rank Adaptation) on the base model `stabilityai/stable-diffusion-xl-base-1.0`.
3. Generates 20–30 new images based on input queries.
4. Refines the output images using `stabilityai/stable-diffusion-xl-refiner-1.0`.
5. Runs in a Google Colab environment with Google Drive integration.
6. Provides a **separate** file `documentation.md` for all detailed instructions (both for new and experienced developers).

### 2. Project Objectives

1. **Data Processing**  
   - All raw images (primarily `.jpg`) will be uploaded and processed from a Google Drive folder (e.g., `/babanne-images`).

2. **Training**  
   - Fine-tune the base model using LoRA adaptation, referencing a unique concept token like `<myspecialstyle>`.
   - Train on 200–300 `.jpg` images.

3. **Inference & Sampling**  
   - Use the trained LoRA weights with the base model to generate new designs.
   - Refine each output using the `stabilityai/stable-diffusion-xl-refiner-1.0` model.

4. **Modular Design**  
   - Keep code in dedicated files (`main.py`, `data_utils.py`, `train.py`, `inference.py`).
   - Store configuration in `config.yaml` for easy parameter tuning.

5. **Documentation**  
   - Maintain a separate file, `documentation.md`, with step-by-step instructions on usage (from uploading images to training and inference).

---

## Implementation Roadmap

This roadmap outlines each implementation step. Developer prompts at the end reference `@project_overview.md` to guide incremental development. All **changes** made during implementation must be reflected in the directory structure, **and** a corresponding entry (or update) must be added to the [Task List](task_list.md) using the format shown below.

> **Example `task_list.md` format:**
> ```markdown
> # Task List
> 
> ## High Priority
> - [ ] Initial project setup and configuration (**Status:** To Do, **Notes:** Set up basic project structure and dependencies)
> - [ ] [Task 2] (**Status:** To Do, **Notes:** [Notes])
> 
> ## Medium Priority
> - [ ] [Task 3] (**Status:** To Do, **Notes:** [Notes])
> - [ ] [Task 4] (**Status:** To Do, **Notes:** [Notes])
> 
> ## Low Priority
> - [ ] [Task 5] (**Status:** To Do, **Notes:** [Notes])
> - [ ] [Task 6] (**Status:** To Do, **Notes:** [Notes])
> 
> ## Completed
> - [x] Create project structure and documentation
> - [x] Set up .cursorrules and .cursorignore
> ```

When each step is completed, update the relevant task to reflect its **Status** (e.g., “In Progress” or “Completed”) and optionally include **Notes** (e.g., new issues or details discovered).

### 1. **Setup & File Organization**

**Objective:** Place all project files in the root directory without additional subfolders.

1. **Required Files in Root**  
   - `config.yaml`
   - `main.py`
   - `data_utils.py`
   - `train.py`
   - `inference.py`
   - `documentation.md` *(created for detailed instructions)*
   - `project_overview.md` *(this file, containing the PRD, located in `.notes`)*

2. **Note on Directories**  
   - No additional subdirectories are necessary.  
   - Images (e.g., `.jpg` files) will live in Google Drive (e.g., `/content/drive/MyDrive/babanne-images`).

---

### 2. **Configuration File**

**Objective:** Store project parameters and hyperparameters in a single YAML file (`config.yaml`).

1. **Sample `config.yaml` Contents:**

   ```yaml
   drive_mount_path: "/content/drive"
   images_dir: "/content/drive/MyDrive/babanne-images"
   lora_output_dir: "/content/drive/MyDrive/lora_output"
   instance_prompt: "a photo of <myspecialstyle>"
   class_prompt: "a photo of a painting"  # optional if prior preservation is needed
   pretrained_model: "stabilityai/stable-diffusion-xl-base-1.0"
   refiner_model: "stabilityai/stable-diffusion-xl-refiner-1.0"
   resolution: 512
   train_batch_size: 2
   learning_rate: 1e-4
   max_train_steps: 1000
   ```

2. **Notes:**  
   - Each field is customizable.  
   - Add or adjust parameters as needed (e.g., training steps, batch size).
   - After creation, add a corresponding entry in your `task_list.md` to reflect that the config file is completed or updated.

---

### 3. **Main Script**

**Objective:** Provide a single entry point (`main.py`) to run training or inference.

1. **Responsibilities of `main.py`:**  
   - Parse command-line arguments (e.g., `--mode train` or `--mode inference`).  
   - Load `config.yaml`.  
   - Optionally mount Google Drive (if used in a Colab environment).  
   - Call `train.run(config)` or `inference.run(config)` based on the selected mode.

2. **Example Flow:**  
   - `python main.py --mode train` → Runs training  
   - `python main.py --mode inference` → Runs inference

3. **Task List Updates:**  
   - Mark tasks (e.g., “initial project setup” or “add main script”) in `task_list.md` and update their status once completed.

---

### 4. **Data Utilities**

**File:** `data_utils.py`

**Objective:**  
Handle image processing tasks like listing and annotating images.

1. **Functions:**
   - `list_images(image_dir, extensions={".jpg"})`:  
     Returns a list of file paths in `image_dir` matching the `.jpg` extension.
   - `create_annotations(image_dir, prompt)`:  
     Creates an `annotations.txt` file where each line is `<image_path>\t<prompt>`.

2. **Usage:**  
   - Called by the training script to generate annotation files before training.
   - Document progress in `task_list.md` after implementation or any subsequent updates.

---

### 5. **Training**

**File:** `train.py`

**Objective:**  
Run LoRA adaptation using your annotated dataset.

1. **Process:**
   1. Load config from `config.yaml`.
   2. Generate `annotations.txt` using `create_annotations(...)`.
   3. Construct the training command (e.g., using `accelerate` to launch `train_lora.py` or a custom script).
   4. Execute the command in a subprocess or function call.

2. **Output:**  
   - Stores LoRA weights (and potentially logs or partial checkpoints) in `lora_output_dir`.

3. **Task List Note:**  
   - When training setup is done or changed, update the corresponding item in `task_list.md` (e.g., “LoRA training setup”).

---

### 6. **Inference**

**File:** `inference.py`

**Objective:**  
Generate new images using the trained LoRA weights, then refine them.

1. **Process:**
   - Load base pipeline with LoRA weights applied (if code merges them).
   - Generate initial images from the instance prompt or a custom prompt.
   - Use the refiner pipeline to enhance the generated images.
   - Save final images locally (e.g., `refined_output.png`).

2. **Output:**  
   - `.png` or `.jpg` files with refined results.

3. **Task List Update:**  
   - Mark inference tasks complete or in progress after implementing logic in `inference.py`.

---

### 7. **Documentation**

**File:** `documentation.md`

**Objective:**  
Provide a **separate** file that thoroughly explains how to:
- Set up the environment (e.g., install libraries, mount Google Drive).
- Modify `config.yaml` to point to your `.jpg` images.
- Run training via `main.py --mode train`.
- Run inference via `main.py --mode inference`.
- Troubleshoot common issues (e.g., out-of-memory, missing dependencies).

> **Important:** The entire end-user guide for both novices and experienced developers will live in `documentation.md`. This includes step-by-step instructions, sample commands, and tips for debugging.

---

## Final File Structure

```
.
├── config.yaml         # Stores parameters (paths, hyperparams, etc.)
├── main.py             # CLI entry point for train or inference mode
├── data_utils.py       # Image processing and annotation creation
├── train.py            # Runs LoRA training
├── inference.py        # Performs inference and refinement
├── documentation.md    # Detailed usage and developer documentation
└── .notes
    ├── directory_structure.md
    ├── meeting_notes.md
    ├── project_overview.md  # (This file) The PRD for the overall project
    └── task_list.md
```

*(The rest of your project files like `bruh.txt`, `.cursorignore`, `.cursorrules`, etc. remain unchanged.)*

---

## Example Developer Prompts

Below are sample prompts that developers might use to implement each stage according to **@project_overview.md**. Each prompt also reminds you to **update `task_list.md`** to reflect progress and any directory changes.

1. **Initial Setup Prompt:**
   ```
   ok lets start implementing ai image generation model according to @project_overview.md
   first lets setup 1. Setup & File Organization
   make sure we update task_list.md with all changes
   ```
2. **Data Preparation Prompt:**
   ```
   ok lets start implementing ai image generation model according to @project_overview.md
   next lets setup 2. Data Utilities
   remember to document updates in task_list.md and directory structure
   ```
3. **Training Prompt:**
   ```
   ok lets start implementing ai image generation model according to @project_overview.md
   now lets setup 3. Training with LoRA adaptation
   after each step, we'll mark tasks in task_list.md
   ```
4. **Inference Prompt:**
   ```
   ok lets start implementing ai image generation model according to @project_overview.md
   finally lets setup 4. Inference & Refinement
   don't forget to reflect these changes in task_list.md
   ```
5. **Documentation Prompt:**
   ```
   ok let's finalize the project with the instructions from @project_overview.md
   create the documentation.md file as described in the prd
   update task_list.md to show documentation is completed
   ```

Documentation:
Below is a minimal (but end-to-end) illustration of how you might:

    Gather a dataset (200–300 .jpg lace embroidered fabric images).
    Train a LoRA (Low-Rank Adaptation) on the Stable Diffusion XL Base model (stabilityai/stable-diffusion-xl-base-1.0).
    Generate new images using your fine-tuned LoRA weights.
    Refine the generated images using the Stable Diffusion XL Refiner (stabilityai/stable-diffusion-xl-refiner-1.0).

    Important

        This code is purely demonstrative/minimal. You’ll almost certainly need to adjust hyperparameters (learning rate, batch size, number of steps, etc.) for quality results.
        The Hugging Face diffusers repository contains official, more complete training scripts (like train_text_to_image_lora.py), which you can customize.
        Make sure to install packages: pip install diffusers transformers accelerate datasets safetensors xformers (plus xformers if you want memory-efficient attention).
        You must have a GPU (e.g., via a cloud service or a local machine with CUDA) for training and inference in a reasonable time.

1. Directory Structure

Assume you have a directory with your training images (200–300 .jpg files) like:

my_lace_dataset/
    image_0001.jpg
    image_0002.jpg
    ...
    image_0300.jpg

2. Minimal Training Code (LoRA on SDXL Base)

Below is a single-file example (call it train_lora_sdxl.py) that:

    Loads images from my_lace_dataset/.
    Uses a basic Hugging Face Dataset wrapper.
    Trains LoRA on the SDXL base model.

#!/usr/bin/env python
# train_lora_sdxl.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from diffusers import AutoPipelineForText2Image, DDPMScheduler
from diffusers.optimization import LoRAConfig, LoRAAttnProcs
from transformers import AutoTokenizer
from datasets import load_dataset

from accelerate import Accelerator
from tqdm.auto import tqdm

# -----------------
# 1. Hyperparameters
# -----------------
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = "./sdxl-lora-lace"
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_TRAIN_STEPS = 1000  # Adjust as needed
LORA_RANK = 4           # LoRA rank
TRAIN_CAPTION = "lace embroidered fabric"  # Simple universal caption for all images (example)

# -----------------
# 2. Prepare Dataset
# -----------------
class LaceImagesDataset(Dataset):
    def __init__(self, image_folder, caption):
        self.paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]
        self.caption = caption

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return {
            "pixel_values": image,
            "text": self.caption
        }

def collate_fn(examples):
    # images and captions in batch
    images = [ex["pixel_values"] for ex in examples]
    texts = [ex["text"] for ex in examples]
    return {
        "pixel_values": images,
        "text": texts
    }

# Instantiate dataset & data loader
dataset = LaceImagesDataset("my_lace_dataset", TRAIN_CAPTION)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# -----------------
# 3. Initialize Accelerator
# -----------------
accelerator = Accelerator(
    mixed_precision="fp16"  # or "bf16", depending on your hardware
)

# -----------------
# 4. Load Base Pipeline
# -----------------
print("Loading base SDXL pipeline...")
pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to(accelerator.device)

# We will need the text encoder tokenizer for text processing
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

# Switch to a simpler scheduler for training (e.g., DDPMScheduler or EulerAncestralDiscreteScheduler, etc.)
pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

# -----------------
# 5. Set Up LoRA
# -----------------
# Create LoRA config
lora_config = LoRAConfig(
    r=LORA_RANK,
    alpha=1.0,
    dropout=0.0,
    bias="none",
    # We only want to adapt cross-attention layers
    # so we set `target_modules` to ["CrossAttention"]
    # If you want to adapt unet + text encoder, you can specify that too.
)

# Create LoRA-attention processors
lora_attn_procs = LoRAAttnProcs(pipeline.unet, lora_config)
pipeline.unet.set_attn_processor(lora_attn_procs)

# Optionally, also adapt the text encoder’s attention (commented out for minimal example)
# text_encoder_attn_procs = LoRAAttnProcs(pipeline.text_encoder, lora_config)
# pipeline.text_encoder.set_attn_processor(text_encoder_attn_procs)

# -----------------
# 6. Optimizer
# -----------------
# Collect trainable parameters from LoRA
params_to_optimize = (
    lora_attn_procs.parameters()
    # + text_encoder_attn_procs.parameters() # if you also adapt the text encoder
)
optimizer = torch.optim.AdamW(params_to_optimize, lr=LEARNING_RATE)

# Prepare everything with Accelerator
pipeline.unet, optimizer, train_dataloader = accelerator.prepare(
    pipeline.unet, optimizer, train_dataloader
)

# -----------------
# 7. Training Loop
# -----------------
global_step = 0
pipeline.unet.train()

for epoch in range(1000):  # a simple loop, you might want to rely on global steps
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
        # Convert images to tensor
        images = [img.resize((512,512)) for img in batch["pixel_values"]]
        images = torch.stack([torch.tensor(pipeline.feature_extractor(img, return_tensors="pt")["pixel_values"][0]) for img in images])
        images = images.to(accelerator.device, dtype=torch.float16)

        # Convert text to input_ids
        texts = batch["text"]
        inputs = tokenizer(
            texts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(accelerator.device)

        # 1. Get model prediction for the noise
        noise_pred = pipeline.unet(
            images,
            pipeline.scheduler.timesteps[-1], # just an example of single-step
            encoder_hidden_states=pipeline.text_encoder(inputs.input_ids)[0]
        ).sample

        # 2. Compute simple loss (L2 between predicted noise and actual noise)
        # This is a simplified training approach for demonstration
        with torch.no_grad():
            noise = torch.randn_like(noise_pred)

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # 3. Backprop
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        if global_step >= NUM_TRAIN_STEPS:
            break
    if global_step >= NUM_TRAIN_STEPS:
        print("Training complete!")
        break

# -----------------
# 8. Save LoRA weights
# -----------------
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    print(f"Saving LoRA weights to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lora_attn_procs.save_pretrained(OUTPUT_DIR)  # just save unet attn procs
    # If you adapted the text encoder, save that too:
    # text_encoder_attn_procs.save_pretrained(os.path.join(OUTPUT_DIR, "text_encoder"))

Key points:

    LoRAAttnProcs (from diffusers.optimization) creates LoRA layers for the UNet’s cross-attention.
    We do a simple training loop with a naive noise prediction objective. (Real training uses a more robust approach, but this is enough for demonstration.)
    We periodically compare the predicted noise to random noise, backprop through the LoRA parameters, and save the LoRA weights at the end.

3. Minimal Inference/Generation Code

Now, we want to:

    Load the base SDXL model.
    Load the LoRA weights we saved.
    Generate images from custom prompts.
    Refine those images with the XL Refiner.

Create a script, for example, inference_sdxl_lora.py:

#!/usr/bin/env python
# inference_sdxl_lora.py

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.optimization import LoRAAttnProcs

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
LORA_WEIGHTS_DIR = "./sdxl-lora-lace"

if __name__ == "__main__":
    # -----------------
    # 1. Load Base Pipeline
    # -----------------
    base_pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16
    )
    base_pipe.to("cuda")

    # You can switch to a more advanced scheduler for inference
    base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config)

    # -----------------
    # 2. Load LoRA Weights
    # -----------------
    lora_attn_procs = LoRAAttnProcs.load_pretrained(LORA_WEIGHTS_DIR)
    base_pipe.unet.set_attn_processor(lora_attn_procs)
    # If you also adapted a text encoder, load that too

    # -----------------
    # 3. Generate Images w/ LoRA
    # -----------------
    prompt = "A close-up of elegant white lace embroidered fabric, intricate pattern, 8k resolution"
    negative_prompt = "grainy, low quality"

    # We output latents so we can pass them to the refiner
    base_latents = base_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        output_type="latent"  # get latent representations, not images
    ).images

    # If you want direct images from base (without refinement), you can do:
    # base_images = base_pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     num_inference_steps=30
    # ).images
    # base_images[0].save("base_result.png")

    # -----------------
    # 4. Refine Using SDXL Refiner
    # -----------------
    refiner_pipe = DiffusionPipeline.from_pretrained(
        REFINER_MODEL,
        torch_dtype=torch.float16
    )
    refiner_pipe.to("cuda")
    refiner_pipe.scheduler = DPMSolverMultistepScheduler.from_config(refiner_pipe.scheduler.config)

    refined_images = refiner_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_latents,
        num_inference_steps=30
    ).images

    # Save final results
    for i, img in enumerate(refined_images):
        img.save(f"refined_output_{i}.png")
    print("Refinement done. Check saved images.")

How It Works (Step by Step)

    Load Base Model
    We load stabilityai/stable-diffusion-xl-base-1.0 with half-precision (torch.float16), then move it to the GPU.

    Load LoRA Weights
    We fetch the saved attention processors (LoRAAttnProcs) from the directory where we stored them and inject them into the base UNet. This effectively merges the LoRA layers.

    Generate Latents
    Instead of returning final PIL images, we request output_type="latent" from the base model. This gives us a latent representation of the images.

    Refine with stabilityai/stable-diffusion-xl-refiner-1.0
    We load a second DiffusionPipeline for the refiner model, pass in the same prompt and the latents from the base step, and let it produce higher-quality final images.

    Save
    We simply .save(...) the final images as .png.

4. Final Notes

    Hyperparameter Tuning:
        BATCH_SIZE, LEARNING_RATE, NUM_TRAIN_STEPS, and LORA_RANK are just starting points. Real results might require longer training, a better LR schedule, or different optimizer settings.
    Text Encoder LoRA:
        In many advanced LoRA training strategies, you also adapt the text encoder to better align the new concept. Above, we demonstrated only UNet LoRA for simplicity.
    Refiner Usage:
        The refiner expects latents (not standard images). By running base_pipe(..., output_type="latent"), you can chain the latents directly into the refiner pipeline.
    Memory:
        SDXL is large. If you run out of memory, consider gradient accumulation, smaller batch sizes, or turning on xFormers memory-efficient attention.

This setup should give you the minimal steps for:

    Collecting images
    Training LoRA on SDXL Base
    Generating new images with queries
    Refining them using SDXL Refiner