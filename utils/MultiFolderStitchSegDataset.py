import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MultiFolderStitchSegDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.class_ids = list(image_dirs.keys())
        self.transform = transform

        # ✅ Handle naming convention: c_XXXXX.png (images) vs e_XXXXX.png (masks)
        all_files = set()
        for class_id in self.class_ids:
            img_dir = image_dirs[class_id]
            mask_dir = mask_dirs[class_id]

            # Get files that exist in both directories
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                img_files = set(os.listdir(img_dir))
                mask_files = set(os.listdir(mask_dir))

                # Convert mask filenames to image filenames (e_XXXXX.png -> c_XXXXX.png)
                mask_to_img_files = set()
                for mask_file in mask_files:
                    if mask_file.startswith('e_'):
                        img_file = 'c_' + mask_file[2:]  # Replace 'e_' with 'c_'
                        mask_to_img_files.add(img_file)

                # Find intersection between actual image files and converted mask filenames
                common_files = img_files.intersection(mask_to_img_files)
                all_files.update(common_files)

        self.image_files = sorted(list(all_files))
        print(f"Dataset initialized with {len(self.image_files)} valid files")
        
        # Print class mapping for debugging
        print("Class mapping:")
        for class_id in self.class_ids:
            print(f"  Folder {class_id} -> Class {class_id + 1}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]  # This is c_XXXXX.png
        final_mask = None
        final_image = None

        # Find which class this image belongs to
        image_class = None
        for class_id in self.class_ids:
            img_path = os.path.join(self.image_dirs[class_id], fname)
            if os.path.exists(img_path):
                image_class = class_id
                break
        
        if image_class is None:
            raise ValueError(f"Image {fname} not found in any directory")

        # Load the image
        img_path = os.path.join(self.image_dirs[image_class], fname)
        final_image = Image.open(img_path).convert("RGB")
        
        # Load the corresponding mask
        mask_fname = 'e_' + fname[2:]  # c_XXXXX.png -> e_XXXXX.png
        mask_path = os.path.join(self.mask_dirs[image_class], mask_fname)
        
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
            # Convert to binary mask (stitch vs background)
            mask_arr = (np.array(mask_img) > 0).astype(np.uint8)
            
            # Assign class ID: 0 for background, class_id+1 for stitch type
            # Class 0: Background, Class 1: Flat stitch, Class 2: Satin stitch, Class 3: Tatami stitch
            final_mask = mask_arr * (image_class + 1)
        else:
            # No mask → all background (class 0)
            final_mask = np.zeros((final_image.height, final_image.width), dtype=np.uint8)

        # Convert to tensor
        image = T.ToTensor()(final_image)
        mask = torch.as_tensor(final_mask, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask
