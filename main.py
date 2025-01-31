import argparse
import yaml
from google.colab import drive
import train
import inference

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def mount_drive(config):
    print("Mounting Google Drive...")
    drive.mount(config['drive_mount_path'])

def main():
    parser = argparse.ArgumentParser(description='AI Image Generation Model')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                       help='Mode to run: train or inference')
    args = parser.parse_args()
    
    config = load_config()
    mount_drive(config)
    
    if args.mode == 'train':
        train.run(config)
    else:
        inference.run(config)

if __name__ == '__main__':
    main() 