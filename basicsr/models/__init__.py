import torch
import cv2
import os
import argparse
import yaml
from basicsr.models import process_image
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to the model configuration file (YML)')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights (.pth)')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder with images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    with open(args.opt, mode='r') as f:
        opt = yaml.safe_load(f)

    # Set the model to validation/test mode
    opt['is_train'] = False
    opt['val']['suffix'] = '' # Avoids creating a subdirectory for results

    model = load_model(opt, args.weights)

    for filename in sorted(os.listdir(args.input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            print(f"Processing {filename}...")
            input_path = os.path.join(args.input_folder, filename)
            output_image = process_image(model, input_path)
            output_path = os.path.join(args.output_folder, filename)
            cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            print(f"Saved processed image to {output_path}")

if __name__ == "__main__":
    main()