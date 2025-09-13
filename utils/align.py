import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch


class AlignedEmbroideryDataset(Dataset):
    def __init__(self, root_dir, mask_dir="masks", image_size=256):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # Get all image files
        self.image_files = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.image_files.sort()

        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # converts to [0,1]
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and process the main image
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        mid = w // 2

        target_img = img.crop((0, 0, mid, h))
        input_img = img.crop((mid, 0, w, h))

        target_tensor = self.transform(target_img)
        input_tensor = self.transform(input_img)

        return {'A': input_tensor,'B': target_tensor,'A_paths': img_path,'B_paths': img_path}
