import torch
import cv2
import os
import argparse
import yaml
from basicsr.models import create_model
from basicsr.utils import imfrombytes, img2tensor
import numpy as np
import torch.nn.functional as F

def load_model(opt, weights_path):
    """Loads the model using the provided options and weights."""
    opt['dist'] = False
    opt['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(opt)

    load_net = torch.load(weights_path)

    if 'params_ema' in load_net:
        keyname = 'params_ema'
    elif 'params' in load_net:
        keyname = 'params'
    else:
        keyname = None

    if keyname:
        model.net_g.load_state_dict(load_net[keyname], strict=True)
    else:
        model.net_g.load_state_dict(load_net, strict=True)

    return model

def process_image(model, img_path):
    """Processes a single image by directly using the model's network."""
    # Read and prepare the image
    img_bytes = open(img_path, 'rb').read()
    img = imfrombytes(img_bytes, float32=True)
    img = img2tensor(img, bgr2rgb=True, float32=True)
    img = img.unsqueeze(0).to('cuda')

    # --- THIS IS THE FIX ---
    # Pad the image to be divisible by a factor (e.g., 16 or 32)
    mod_pad_h, mod_pad_w = 0, 0
    h, w = img.size()[2], img.size()[3]
    factor = 16 # A safe factor for many networks
    if h % factor != 0:
        mod_pad_h = factor - h % factor
    if w % factor != 0:
        mod_pad_w = factor - w % factor
    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # -----------------------

    # Directly run inference with the network (model.net_g)
    with torch.no_grad():
        model.net_g.eval() # Set to evaluation mode
        output = model.net_g(img)

    # --- CROP THE PADDING ---
    # Remove the padding to restore original size
    _, _, h, w = output.size()
    output = output[:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]
    # ------------------------

    # Convert tensor to savable image format
    restored = torch.clamp(output, 0, 1)
    restored = restored.squeeze(0).cpu().permute(1, 2, 0).numpy()
    return (restored * 255.0).round().astype(np.uint8)

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

    opt['is_train'] = False
    if 'val' in opt.keys():
        opt['val']['suffix'] = ''

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