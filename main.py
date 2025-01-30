import argparse
import yaml
import os
from pathlib import Path

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def mount_drive(config):
    """Mount Google Drive if running in Colab environment."""
    try:
        # Check if running in IPython/Colab environment
        get_ipython()  # This will raise NameError if not in IPython
        
        try:
            from google.colab import drive
            print("Running in Colab. Mounting Google Drive...")
            drive.mount(config['drive_mount_path'])
            print(f"Looking for images in: {config['images_dir']}")
            # Don't modify paths when in Colab
            return config
        except ImportError:
            print("Google Colab import failed. Running in different IPython environment.")
            return _setup_local_paths(config)
    except NameError:
        print("Running in standard Python environment.")
        return _setup_local_paths(config)

def _setup_local_paths(config):
    """Setup local directories when not using Google Drive."""
    # Check if the Google Drive path exists first
    if os.path.exists(config['images_dir']):
        print(f"Found Google Drive path: {config['images_dir']}")
        return config
        
    # If Google Drive path doesn't exist, try local paths
    output_dir = Path(config['lora_output_dir'].replace('/content/drive/MyDrive/', './'))
    images_dir = Path(config['images_dir'].replace('/content/drive/MyDrive/', './'))
    
    # Create directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config paths
    config['lora_output_dir'] = str(output_dir)
    config['images_dir'] = str(images_dir)
    print(f"Using local directories:")
    print(f"- Images: {config['images_dir']}")
    print(f"- Output: {config['lora_output_dir']}")
    
    return config

def main():
    parser = argparse.ArgumentParser(description='AI Image Generation Model')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                       help='Mode to run: train or inference')
    args = parser.parse_args()
    
    try:
        config = load_config()
        config = mount_drive(config)  # Update config with correct paths
        
        # Verify image directory exists
        if not os.path.exists(config['images_dir']):
            raise FileNotFoundError(
                f"Image directory not found: {config['images_dir']}\n"
                f"Please make sure your images are in the correct location."
            )
        
        # List contents of image directory
        print("\nChecking image directory contents:")
        try:
            files = os.listdir(config['images_dir'])
            print(f"Found {len(files)} files in {config['images_dir']}")
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} image files")
        except Exception as e:
            print(f"Error listing directory: {str(e)}")
        
        if args.mode == 'train':
            import train
            train.run(config)
        else:
            import inference
            inference.run(config)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you're running in Google Colab")
        print("2. Check that Google Drive is mounted:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        print(f"3. Verify your images are in: {config['images_dir']}")
        print("4. Check documentation.md for setup instructions")

if __name__ == '__main__':
    main() 