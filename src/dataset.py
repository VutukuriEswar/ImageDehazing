import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np

class DehazeDataset(Dataset):
    """Custom Dataset for loading paired hazy and clear images."""
    def __init__(self, hazy_dir: Path, clear_dir: Path, transform=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.hazy_images = sorted([f for f in hazy_dir.iterdir() if f.suffix.lower() in image_extensions])
        self.clear_images = sorted([f for f in clear_dir.iterdir() if f.suffix.lower() in image_extensions])

        assert len(self.hazy_images) == len(self.clear_images), \
            f"Mismatch in number of images: {len(self.hazy_images)} hazy vs {len(self.clear_images)} clear"

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = self.hazy_images[idx]
        clear_path = self.clear_images[idx]

        hazy_image = Image.open(hazy_path).convert("RGB")
        clear_image = Image.open(clear_path).convert("RGB")

        if self.transform:
            # Apply the same random transforms to both images
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            hazy_image = self.transform(hazy_image)
            torch.manual_seed(seed)
            clear_image = self.transform(clear_image)

        return {'hazy': hazy_image, 'clear': clear_image}