import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import itertools
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from model import UNetGenerator
from utils import encode_mask, decode_segmap


class InferenceArgs: #Class to simulate the arguments passed by argparse in main.py
    def __init__(self, feat, norm_type, n_layer_g, dropout, dropout_mode):
        self.feat = feat
        self.norm_type = norm_type
        self.n_layer_g = n_layer_g
        self.dropout = dropout
        self.dropout_mode = dropout_mode


def check_if_discriminator(state_dict): # It checks if the loaded model is a discriminator by looking at the keys of the state dict
    keys = list(state_dict.keys())
    if any('model.' in k for k in keys) and not any('downs' in k for k in keys):
        return True
    return False


def attempt_load_model(model_path, device, config): #Attempts to load the model handling size errors
    try:
        netG = UNetGenerator(config).to(device) # Initialize the generator
        checkpoint = torch.load(model_path, map_location=device)
        # Extract the state_dict with the weights
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        if check_if_discriminator(state_dict):
            return None, False, "IS_DISCRIMINATOR"

        netG.load_state_dict(state_dict) # Load the weights in the model
        netG.eval() # Set the net in evaluation mode

        # If we are in 'paper' mode, re-enable dropout
        if config.dropout_mode == 'paper' and config.dropout > 0:

            def enable_dropout(m): # Check if m is a dropout layer and call m.train on it
                if isinstance(m, torch.nn.Dropout):
                    m.train() # It enables the dropout

            netG.apply(enable_dropout)

        return netG, True, "OK"

    except RuntimeError as e:
        if "size mismatch" in str(e) or "Missing key" in str(e):
            return None, False, "SHAPE_MISMATCH"
        return None, False, "ERROR"


def find_correct_parameters(model_path, device, user_dropout, user_mode): #Automatically searches for the model configuration
    print(f"AUTO-DETECT] Analyzing model: {os.path.basename(model_path)}")

    # Order of probability for parameters
    possible_n_layers = [4, 5, 3, 6, 8]        # Layer of the generator
    possible_feats = [32, 64, 128]          # Number of features
    possible_norms = ['instance', 'batch']  # Normalization types

    for n_layer, feat, norm in itertools.product(possible_n_layers, possible_feats, possible_norms):

        config = InferenceArgs(feat, norm, n_layer, user_dropout, user_mode)
        model, success, status = attempt_load_model(model_path, device, config)

        if success:
            print(f"Parameters Found: Layers={n_layer}, Feat={feat}, Norm={norm}")
            return model, config
        elif status == "IS_DISCRIMINATOR":
            raise ValueError("ERROR: You are using a DISCRIMINATOR file (_D_). Use the GENERATOR file (_G_).")

    raise ValueError("IMPOSSIBLE TO LOAD: No combination of parameters worked.")


def preprocess_exactly_like_dataset(mask_path, target_size=(256, 512)):
    
    # It replicates exactly the dataset.py logic with added debug for raw values before encoding.
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"File non trovato: {mask_path}")

    mask = Image.open(mask_path)

    mask = encode_mask(mask) # 34 classes to 19

    # Resize to 256x512 using the nearest interpolation
    mask = TF.resize(mask, [target_size[0], target_size[1]], interpolation=transforms.InterpolationMode.NEAREST)

    # Convert to numpy array
    mask_np = np.array(mask)
    unique_processed = np.unique(mask_np)

    if len(unique_processed) == 1 and unique_processed[0] == 255: # Error if the mask is of the wrong type
        print("CRITICAL ERROR: The mask has become ALL 255 (Void).")

    return torch.from_numpy(mask_np).long() # Convert in to torch tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the generator .pth file")
    parser.add_argument('--mask_path', type=str, required=True, help="Path to the mask image (labelIds)")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout (e.g., 0.5)")
    parser.add_argument('--dropout_mode', type=str, default='paper', choices=['paper', 'full'])

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    netG, best_config = find_correct_parameters(args.model_path, device, args.dropout, args.dropout_mode)

    # Load and Process Mask
    mask_tensor = preprocess_exactly_like_dataset(args.mask_path)
    mask_input = mask_tensor.unsqueeze(0).to(device)

    # Generation
    print("\nGenerating image...")
    with torch.no_grad():
        fake_img = netG(mask_input)

    # decode_segmap returns a tensor [1, 3, H, W] normalized [-1, 1]
    mask_decoded = decode_segmap(mask_input, device)

    # Conversion for Matplotlib [0, 1]
    mask_vis = (mask_decoded.squeeze(0).cpu() + 1) / 2.0
    mask_vis = mask_vis.permute(1, 2, 0).numpy()

    fake_vis = (fake_img.squeeze(0).cpu() + 1) / 2.0
    fake_vis = fake_vis.permute(1, 2, 0).numpy()
    fake_vis = np.clip(fake_vis, 0, 1)

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"Input Mask (Decoded by utils.py)")
    plt.imshow(mask_vis)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Generated Image")
    plt.imshow(fake_vis)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

#python generate_image.py --model_path ~/../../Volumes/Avvo\ SSD/Università/DeapLearning/models/run_final_cGAN_lambda100_200e_G_best_fid.pth --mask_path ./masks/frankfurt_000000_002963_gtFine_labelIds.png --dropout 0.5 --dropout_mode paper
