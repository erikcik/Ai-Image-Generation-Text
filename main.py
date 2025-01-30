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
        # Check if we're in Colab
        import google.colab
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount(config['drive_mount_path'])
    except ImportError:
        print("Not running in Google Colab. Skipping drive mount.")
        # Ensure the output directory exists locally
        Path(config['lora_output_dir']).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='AI Image Generation Model')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                       help='Mode to run: train or inference')
    args = parser.parse_args()
    
    config = load_config()
    mount_drive(config)
    
    if args.mode == 'train':
        import train
        train.run(config)
    else:
        import inference
        inference.run(config)

if __name__ == '__main__':
    main() 