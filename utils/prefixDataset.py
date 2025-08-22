import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PrefixStitchDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform

        # ✅ Collect only "c_" image files
        self.image_files = [
            f for f in os.listdir(image_dir) if f.startswith("c_")
        ]

        # Define transforms
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image filename (e.g. c_00164.png)
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Derive mask filename (replace 'c_' with 'e_')
        mask_name = img_name.replace("c_", "e_", 1)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load mask (if exists)
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("RGB")  # Convert to RGB to match image dimensions
        else:
            # Create a black mask if file doesn't exist
            mask_img = Image.new("RGB", image.size, (0, 0, 0))

        # Apply transforms
        image = self.transform(image)
        mask = self.transform(mask_img)

        return image, mask
