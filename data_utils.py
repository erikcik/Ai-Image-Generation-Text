import os
from pathlib import Path
from typing import Set, List, Optional

def list_images(image_dir: str, extensions: Optional[Set[str]] = None) -> List[str]:
    """List all image files in the specified directory with given extensions.
    
    Args:
        image_dir: Path to directory containing images
        extensions: Set of file extensions to include (default: {".jpg", ".jpeg", ".png"})
    
    Returns:
        Sorted list of image paths
    """
    extensions = extensions or {".jpg", ".jpeg", ".png"}
    
    # Convert to Path object
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    
    # Check if directory is empty
    if not any(image_dir.iterdir()):
        raise ValueError(f"Image directory is empty: {image_dir}")
    
    image_files = []
    for ext in extensions:
        found = list(image_dir.glob(f"*{ext.lower()}")) + list(image_dir.glob(f"*{ext.upper()}"))
        image_files.extend(found)
        print(f"Found {len(found)} files with extension {ext}")
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir} with extensions: {extensions}")
    
    return [str(f) for f in sorted(set(image_files))]

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