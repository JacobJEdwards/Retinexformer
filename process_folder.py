import torch
import cv2
import os
import argparse
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
import numpy as np

def load_model(weights_path):
    """Loads the Retinexformer model and weights."""
    model = RetinexFormer()
    load_net = torch.load(weights_path)
    model.load_state_dict(load_net['params'])
    model.eval()

    return model.cuda()

def process_image(model, img_path):
    """Processes a single image."""
    img = cv2.imread(img_path)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        restored, _, _ = model(img)
    restored = torch.clamp(restored, 0, 1)
    restored = restored.cpu().squeeze().permute(1, 2, 0).numpy()
    return (restored * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder with images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder to save processed images')
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Load the model
    model = load_model(args.weights)

    # Process each image in the input folder
    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")
            input_path = os.path.join(args.input_folder, filename)
            output_image = process_image(model, input_path)
            output_path = os.path.join(args.output_folder, filename)
            cv2.imwrite(output_path, output_image)
            print(f"Saved processed image to {output_path}")

if __name__ == "__main__":
    main()