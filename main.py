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
        except ImportError:
            print("Google Colab import failed. Running in different IPython environment.")
            _setup_local_paths(config)
    except NameError:
        print("Running in standard Python environment. Using local paths.")
        _setup_local_paths(config)

def _setup_local_paths(config):
    """Setup local directories when not using Google Drive."""
    # Create output directory if it doesn't exist
    output_dir = Path(config['lora_output_dir'].replace('/content/drive/MyDrive/', './'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config paths to local
    config['lora_output_dir'] = str(output_dir)
    config['images_dir'] = config['images_dir'].replace('/content/drive/MyDrive/', './')
    print(f"Using local directory: {config['lora_output_dir']}")

def main():
    parser = argparse.ArgumentParser(description='AI Image Generation Model')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                       help='Mode to run: train or inference')
    args = parser.parse_args()
    
    try:
        config = load_config()
        mount_drive(config)
        
        if args.mode == 'train':
            import train
            train.run(config)
        else:
            import inference
            inference.run(config)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease make sure you're either:")
        print("1. Running this in Google Colab (recommended)")
        print("2. Have set up the correct local directories")
        print("\nSee documentation.md for setup instructions.")

if __name__ == '__main__':
    main() 