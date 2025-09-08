import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2, numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from utils.align import AlignedEmbroideryDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# Auto-mask function (entropy + edges)
# ----------------------------
def auto_mask(img):
    """Generate mask from image using entropy and edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_uint8 = (gray / gray.max() * 255).astype(np.uint8)

    ent = entropy(gray_uint8, disk(5)).astype(np.float32)
    ent = (ent - ent.min()) / (ent.max() - ent.min() + 1e-9)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.abs(lap)
    lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-9)

    mask = 0.6 * ent + 0.4 * lap
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return np.clip(mask, 0, 1)


# ----------------------------
# Mask Generator (Fixed Architecture)
# ----------------------------
class MaskGenerator(nn.Module):
    """Generator that produces masks from input images"""
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # Encoder
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Decoder
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1),  # 256x256
            nn.Sigmoid()  # Output mask values between 0 and 1
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Custom Dataset for Mask Training
# ----------------------------
class MaskTrainingDataset:
    """Dataset that generates training pairs of (image, auto_generated_mask)"""
    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        self.image_size = image_size

        # Get all image files
        self.image_files = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.image_files.sort()

        # Transform for images
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])

        # Transform for masks
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate auto mask
        mask_np = auto_mask(img)  # [H, W]

        # Convert to PIL for transforms
        from PIL import Image
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

        # Apply transforms
        img_tensor = self.transform(img_pil)
        mask_tensor = self.mask_transform(mask_pil)

        return img_tensor, mask_tensor


# ----------------------------
# Loss Functions
# ----------------------------
def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for mask prediction"""
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice

def combined_mask_loss(pred, target):
    """Combined loss for mask prediction"""
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# ----------------------------
# Visualization Functions
# ----------------------------
def save_mask_comparison(input_img, target_mask, pred_mask, epoch, batch_idx, sample_idx, save_dir):
    """Save comparison of input image, target mask, and predicted mask"""
    # Convert tensors to numpy and denormalize
    input_np = input_img.cpu().detach().numpy().transpose(1, 2, 0)
    input_np = (input_np * 0.5 + 0.5)  # Denormalize from [-1,1] to [0,1]
    input_np = np.clip(input_np, 0, 1)

    target_np = target_mask.cpu().detach().numpy().squeeze()
    pred_np = pred_mask.cpu().detach().numpy().squeeze()

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(target_np, cmap='gray')
    axes[1].set_title('Target Mask (Auto-generated)')
    axes[1].axis('off')

    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}_batch_{batch_idx}_sample_{sample_idx}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

def save_individual_masks(pred_masks, epoch, batch_idx, save_dir):
    """Save individual predicted masks as separate images"""
    mask_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(mask_dir, exist_ok=True)

    for i, mask in enumerate(pred_masks):
        mask_np = mask.cpu().detach().numpy().squeeze()
        mask_img = (mask_np * 255).astype(np.uint8)

        # Save as PIL image
        mask_pil = Image.fromarray(mask_img, mode='L')
        mask_pil.save(f'{mask_dir}/batch_{batch_idx}_mask_{i}.png')


# ----------------------------
# Training Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
mask_generator = MaskGenerator().to(device)

# Create output directories
os.makedirs("mask_checkpoints", exist_ok=True)
os.makedirs("generated_masks", exist_ok=True)
os.makedirs("mask_comparisons", exist_ok=True)

# Create dataset and dataloader
# Note: Update this path to your actual dataset path
dataset_path = "./MSEmb_DATASET/embs_s_unaligned/train/trainX_e"  # Update this path
if os.path.exists(dataset_path):
    dataset = MaskTrainingDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    print(f"Dataset loaded with {len(dataset)} images")
else:
    print(f"Dataset path {dataset_path} not found. Using dummy data for demonstration.")
    # Fallback to dummy data if dataset not found
    dataset = None
    dataloader = None

# Optimizer
optimizer = torch.optim.Adam(mask_generator.parameters(), lr=1e-4)

# Training loop
epochs = 200

print("Starting mask generation training...")
print(f"Generated masks will be saved in: ./generated_masks/")
print(f"Mask comparisons will be saved in: ./mask_comparisons/")

for epoch in range(epochs):
    if dataloader is None:
        # Dummy training loop for demonstration
        print(f"Epoch {epoch+1}/{epochs} - Using dummy data")

        # Create dummy input and target
        dummy_img = torch.randn(2, 3, 256, 256).to(device)
        dummy_mask = torch.rand(2, 1, 256, 256).to(device)

        # Forward pass
        pred_mask = mask_generator(dummy_img)
        loss = combined_mask_loss(pred_mask, dummy_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

        # Save dummy masks for visualization
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            save_individual_masks(pred_mask, epoch+1, 0, "generated_masks")
            print(f"Dummy masks saved for epoch {epoch+1}")

    else:
        # Real training loop
        mask_generator.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, (images, target_masks) in enumerate(progress_bar):
            images = images.to(device)
            target_masks = target_masks.to(device)

            # Forward pass
            pred_masks = mask_generator(images)
            loss = combined_mask_loss(pred_masks, target_masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Save masks for visualization (first batch of every 5th epoch)
            if (epoch + 1) % 5 == 0 and batch_idx == 0:
                # Save individual masks
                save_individual_masks(pred_masks, epoch+1, batch_idx, "generated_masks")

                # Save comparison images (first 3 samples)
                num_samples = min(3, images.size(0))
                for i in range(num_samples):
                    save_mask_comparison(
                        images[i], target_masks[i], pred_masks[i],
                        epoch+1, batch_idx, i, "mask_comparisons"
                    )

                print(f"\nMasks saved for epoch {epoch+1}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

    # Save checkpoint every 25 epochs
    if (epoch + 1) % 25 == 0:
        torch.save(mask_generator.state_dict(), f'mask_checkpoints/mask_generator_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}")

print("Training completed!")

# Save final model
torch.save(mask_generator.state_dict(), 'mask_checkpoints/mask_generator_final.pth')
print("Final model saved!")

print("\n" + "="*50)
print("TRAINING SUMMARY:")
print("="*50)
print(f"✅ Model checkpoints saved in: ./mask_checkpoints/")
print(f"✅ Generated masks saved in: ./generated_masks/")
print(f"✅ Mask comparisons saved in: ./mask_comparisons/")
print("="*50)
