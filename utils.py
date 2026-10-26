from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import numpy as np
import torch

# Mapping from 34 to 19 classes for Cityscapes
CITYSCAPES_MAPPING = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
}

def encode_mask(mask): # Convert to 34 classes in to 19
    mask_np = np.array(mask) # Convert PIL image to NumPy array for processing
    mask_mapped = np.zeros_like(mask_np) # Create an array of zeros with the same shape as the input

    for k, v in CITYSCAPES_MAPPING.items():
        mask_mapped[mask_np == k] = v

    return Image.fromarray(mask_mapped) # Convert NumPy array back to PIL image


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images normalized to [-1, 1]"""
    # Convert from [-1, 1] to [0, 1]
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    psnr_vals = []
    for i in range(img1_np.shape[0]):
        psnr = peak_signal_noise_ratio(img2_np[i], img1_np[i], data_range=1.0)
        psnr_vals.append(psnr)

    return sum(psnr_vals) / len(psnr_vals)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images normalized to [-1, 1]"""
    # Convert from [-1, 1] to [0, 1
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    ssim_vals = []
    for i in range(img1_np.shape[0]):
        # channel_axis=0 because the format is [C, H, W]
        ssim = structural_similarity(img2_np[i], img1_np[i], data_range=1.0, channel_axis=0)
        ssim_vals.append(ssim)

    return sum(ssim_vals) / len(ssim_vals)

def decode_segmap(mask, device):

    # Color palette for the mask
    palette = torch.tensor([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
        [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
    ], dtype=torch.uint8).to(device)

    mask_temp = mask.clone()
    mask_temp[mask == 255] = 19 # Avoids out-of-bounds errors when accessing the palette

    rgb = palette[mask_temp]

    rgb = rgb.permute(0, 3, 1, 2).float() # Reorder channels: [batch, H, W, 3] → [batch, 3, H, W]
    return (rgb / 127.5) - 1.0