import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

def save_checkpoint(state, filename: Path):
    """Saves checkpoint to a file."""
    print("=> Saving checkpoint")
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(checkpoint_path: Path, model, optimizer, scheduler):
    """Loads model, optimizer, and scheduler states from a checkpoint."""
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch'], checkpoint['best_psnr'], checkpoint['best_ssim'], checkpoint['best_combined_score']

def calculate_metrics(clear_img, dehazed_img):
    """Calculates PSNR and SSIM between two images."""
    clear_np = clear_img.squeeze().cpu().numpy().transpose(1, 2, 0)
    dehazed_np = dehazed_img.squeeze().cpu().numpy().transpose(1, 2, 0)
    dehazed_np = np.clip(dehazed_np, 0., 1.)
    psnr_value = psnr(clear_np, dehazed_np, data_range=1.0)
    ssim_value = ssim(clear_np, dehazed_np, multichannel=True, data_range=1.0, channel_axis=2)
    return psnr_value, ssim_value

def show_images(hazy, clear, dehazed, num_images=3):
    """Displays a side-by-side comparison of hazy, clear, and dehazed images."""
    hazy = hazy.cpu().permute(0, 2, 3, 1).numpy()
    clear = clear.cpu().permute(0, 2, 3, 1).numpy()
    dehazed = dehazed.cpu().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    fig.suptitle("Visual Comparison", fontsize=16)
    for i in range(num_images):
        axes[i, 0].imshow(hazy[i])
        axes[i, 0].set_title("Hazy Input")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(clear[i])
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        axes[i, 2].imshow(np.clip(dehazed[i], 0, 1))
        axes[i, 2].set_title("Model Output")
        axes[i, 2].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()