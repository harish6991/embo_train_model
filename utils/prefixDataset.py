import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PrefixStitchDataset(Dataset):
    def __init__(self, image_dir, tgt_dir, mask_dir, image_size=256, transform=None):
        self.image_dir = image_dir
        self.tgt_dir = tgt_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform

        # ✅ Collect only "c_" image files
        self.image_files = [
            f for f in os.listdir(image_dir) if f.startswith("c_")
        ]

        # Define transforms for images and targets (GRAYSCALE)
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization for grayscale
        ])
        
        # Define transform for masks (no normalization)
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # Keep masks in [0, 1] range
        ])

        print(f"📁 PrefixStitchDataset initialized (GRAYSCALE MODE):")
        print(f"   📂 Images: {image_dir} ({len(self.image_files)} c_ files)")
        print(f"   📂 Targets: {tgt_dir}")
        print(f"   📂 Masks: {mask_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image filename (e.g. c_00164.png)
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Derive target filename (replace 'c_' with 'e_')
        tgt_name = img_name.replace("c_", "e_", 1)
        tgt_path = os.path.join(self.tgt_dir, tgt_name)
        
        # Derive mask filename (keep same name as image or use e_ prefix)
        mask_name = img_name  # Use same filename as image
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Alternative: if masks use 'e_' prefix like targets
        if not os.path.exists(mask_path):
            mask_name = tgt_name  # Try with e_ prefix
            mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image as GRAYSCALE
        image = Image.open(img_path).convert("L")

        # Load target as GRAYSCALE (if exists)
        if os.path.exists(tgt_path):
            tgt_img = Image.open(tgt_path).convert("L")
        else:
            # Create a black target if file doesn't exist
            tgt_img = Image.new("L", image.size, 0)
            
        # Load mask (if exists)
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")  # Load as grayscale
        else:
            # Create a white mask if file doesn't exist (assuming full embroidery)
            mask_img = Image.new("L", image.size, 255)
            print(f"⚠️ Mask not found for {img_name}, using white mask")

        # Apply transforms
        image = self.image_transform(image)
        target = self.image_transform(tgt_img)
        mask = self.mask_transform(mask_img)

        return image, target, mask
