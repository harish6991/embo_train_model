import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AlignedEmbroideryDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".png")]
        self.image_files.sort()  # Keep order consistent

        # Define transforms: Resize → Tensor → Normalize [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        mid = w // 2

        # Split left (target embroidery) and right (input logo)
        target_img = img.crop((0, 0, mid, h))
        input_img = img.crop((mid, 0, w, h))

        # Apply transforms
        target_tensor = self.transform(target_img)
        input_tensor = self.transform(input_img)

        return input_tensor, target_tensor
