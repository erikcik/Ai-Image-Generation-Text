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

