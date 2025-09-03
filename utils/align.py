import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def make_white_mask(img_tensor, threshold=0.95):
    """
    Detect white background pixels.
    img_tensor: [3, H, W] in [0,1]
    threshold: pixels > threshold considered white
    Returns: [1, H, W] mask (1 = embroidery/foreground, 0 = background)
    """
    # Identify white pixels (all channels close to 1)
    is_white = (img_tensor > threshold).all(dim=0)  # [H,W] boolean

    # Invert: embroidery = 1, background = 0
    mask = (~is_white).float().unsqueeze(0)  # [1,H,W]

    return mask


class AlignedEmbroideryDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_files = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".png")
        ]
        self.image_files.sort()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # converts to [0,1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")

        w, h = img.size
        mid = w // 2

        target_img = img.crop((0, 0, mid, h))
        input_img  = img.crop((mid, 0, w, h))

        target_tensor = self.transform(target_img)
        input_tensor  = self.transform(input_img)

        # Build white-mask from target
        mask = make_white_mask(input_tensor)  # [1, H, W]

        return input_tensor, target_tensor, mask