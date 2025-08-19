import torch
import cv2
import os
import argparse
import yaml
from basicsr.models import create_model
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import numpy as np

def load_model(opt, weights_path):
    """Loads the model using the provided options and weights."""
    # Set device and initialize the model
    opt['dist'] = False
    opt['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(opt)

    # Load the network weights
    load_net = torch.load(weights_path)
    load_net = load_net['params']
    model.net_g.load_state_dict(load_net, strict=True)

    return model

def process_image(model, img_path):
    """Processes a single image."""
    # Read and prepare the image
    img_bytes = open(img_path, 'rb').read()
    img = imfrombytes(img_bytes, float32=True)
    img = img2tensor(img, bgr2rgb=True, float32=True)
    img = img.unsqueeze(0).to('cuda')

    # Inference
    with torch.no_grad():
        model.feed_data(data={'lq': img})
        model.test()
        visuals = model.get_current_visuals()
        restored = visuals['rlt']

    # Convert to numpy and save
    restored = torch.clamp(restored, 0, 1)
    restored = restored.squeeze(0).cpu().permute(1, 2, 0).numpy()
    return (restored * 255.0).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to the model configuration file (YML)')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights (.pth)')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder with images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Parse options file
    with open(args.opt, mode='r') as f:
        opt = yaml.safe_load(f)

    # Load the model
    model = load_model(opt, args.weights)

    # Process each image in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            print(f"Processing {filename}...")
            input_path = os.path.join(args.input_folder, filename)
            output_image = process_image(model, input_path)
            output_path = os.path.join(args.output_folder, filename)
            cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            print(f"Saved processed image to {output_path}")

if __name__ == "__main__":
    main()