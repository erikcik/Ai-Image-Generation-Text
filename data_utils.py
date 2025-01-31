import os
import logging
from pathlib import Path
from typing import Set, List, Optional
from PIL import Image

def list_images(image_dir: str, extensions: Set[str] = None) -> List[str]:
    """List all image files with supported extensions in directory."""
    extensions = extensions or {".jpg", ".jpeg", ".png"}
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")
    
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    
    if not images:
        raise ValueError(f"No images found in {image_dir} with extensions: {extensions}")
    
    return sorted([str(img) for img in images])

def create_annotations(image_dir: str, prompt: str) -> str:
    """Create simple annotations file with fixed prompt for all images."""
    images = list_images(image_dir)
    annotations_path = Path(image_dir) / "annotations.txt"
    
    logging.info(f"Creating annotations for {len(images)} images")
    with open(annotations_path, "w") as f:
        for img_path in images:
            f.write(f"{img_path}\t{prompt}\n")
    
    return str(annotations_path)

def create_annotations(image_dir: str, instance_prompt: str, class_prompt: Optional[str] = None) -> str:
    """Create annotations file mapping images to prompts.
    
    Args:
        image_dir: Directory containing training images
        instance_prompt: Prompt template for instance images (e.g., "a photo of <myspecialstyle>")
        class_prompt: Optional prompt template for class images
    
    Returns:
        Path to created annotations file
    """
    images = list_images(image_dir)
    if not images:
        raise ValueError(f"No images found in {image_dir} with supported extensions")
    
    # Create instance annotations
    annotations_path = Path(image_dir) / "annotations.txt"
    with open(annotations_path, 'w') as f:
        for img_path in images:
            f.write(f"{img_path}\t{instance_prompt}\n")
    
    # Create class annotations if provided
    if class_prompt:
        class_annotations_path = Path(image_dir) / "class_annotations.txt"
        with open(class_annotations_path, 'w') as f:
            for img_path in images:
                f.write(f"{img_path}\t{class_prompt}\n")
    
    return str(annotations_path) 