import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image
from models.unetModel import U2NETP

class RealMaskDataset(Dataset):
    """Dataset that uses real generated masks from create_masks.py"""

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Get all image files
        self.image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            self.image_files.extend([
                f for f in os.listdir(images_dir)
                if f.lower().endswith(ext.lower())
            ])
        self.image_files.sort()

        print(f"📁 Found {len(self.image_files)} images in {images_dir}")
        print(f"📁 Looking for masks in {masks_dir}")

        # Verify masks exist
        missing_masks = 0
        for img_file in self.image_files:
            mask_file = os.path.splitext(img_file)[0] + '.png'
            mask_path = os.path.join(masks_dir, mask_file)
            if not os.path.exists(mask_path):
                missing_masks += 1

        if missing_masks > 0:
            print(f"⚠️  Warning: {missing_masks} masks are missing!")
        else:
            print(f"✅ All {len(self.image_files)} masks found!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')

        # Load corresponding mask
        mask_file = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_file)

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # Create empty mask if missing
            mask = Image.new('L', image.size, 0)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def dice_loss(pred, target, smooth=1.0):
    """Dice loss for better overlap optimization"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return 1.0 - dice

def u2net_loss(preds, target):
    """
    U2-Net loss with deep supervision
    preds: list of 7 predictions from U2-Net
    target: ground truth mask
    """
    bce_loss = nn.BCELoss()
    total_loss = 0.0

    # Main prediction (d0) gets highest weight
    main_pred = preds[0]
    main_bce = bce_loss(main_pred, target)
    main_dice = dice_loss(main_pred, target)
    main_loss = main_bce + main_dice
    total_loss += main_loss

    # Side predictions get lower weights
    for i, pred in enumerate(preds[1:], 1):
        # Resize prediction to match target if needed
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

        side_bce = bce_loss(pred, target)
        side_dice = dice_loss(pred, target)
        side_loss = side_bce + side_dice

        # Decreasing weights for deeper side outputs
        weight = 0.5 / (2 ** (i-1))  # 0.5, 0.25, 0.125, etc.
        total_loss += weight * side_loss

    return total_loss

def save_sample_predictions(model, dataloader, device, epoch, save_dir="./training_samples/"):
    """Save sample predictions for visual inspection"""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Get one batch
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            preds = model(images)
            main_pred = preds[0]  # Main prediction

            # Save first 3 samples
            for i in range(min(3, images.size(0))):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                img_np = images[i].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                axes[0].imshow(img_np)
                axes[0].set_title('Input Image')
                axes[0].axis('off')

                # Ground truth mask
                target_np = targets[i, 0].cpu().numpy()
                axes[1].imshow(target_np, cmap='gray')
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')

                # Predicted mask
                pred_np = main_pred[i, 0].cpu().numpy()
                axes[2].imshow(pred_np, cmap='gray')
                axes[2].set_title('Predicted Mask')
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(f"{save_dir}/epoch_{epoch:03d}_sample_{i+1}.png",
                           dpi=150, bbox_inches='tight')
                plt.close()

            break  # Only process first batch

    model.train()

def main():
    print("🎯 U2-NET MASK TRAINING WITH REAL MASKS")
    print("="*60)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Dataset paths
    images_dir = r"./MSEmb_DATASET/embs_s_unaligned/train/trainX_e"
    masks_dir = r"./MSEmb_DATASET/embs_s_unaligned/train/masks"

    # Check paths
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return

    if not os.path.exists(masks_dir):
        print(f"❌ Masks directory not found: {masks_dir}")
        return

    # Data transforms
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    # Create dataset and dataloader
    print("\n📊 Setting up dataset...")
    dataset = RealMaskDataset(images_dir, masks_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)  # Optimized for 32GB GPU

    print(f"📈 Dataset size: {len(dataset)} images")
    print(f"📈 Batch size: 16 (optimized for 32GB GPU)")
    print(f"📈 Batches per epoch: {len(dataloader)}")
    print(f"🚀 Expected GPU memory usage: ~8-12GB")

    # Create model
    print("\n🧠 Initializing U2-Net model...")
    model = U2NETP(3, 1).to(device)  # 3 input channels (RGB), 1 output channel (mask)

    # Optimizer and scheduler (adjusted for larger batch size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)  # Higher LR for batch_size=16
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # Training parameters
    num_epochs = 100
    best_loss = float('inf')

    print(f"🚀 Starting training for {num_epochs} epochs...")
    print("="*60)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(images)

            # Calculate loss
            loss = u2net_loss(predictions, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{epoch_loss/(batch_idx+1):.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)

        # Update learning rate
        scheduler.step()

        # Save sample predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n📸 Saving sample predictions for epoch {epoch+1}")
            save_sample_predictions(model, dataloader, device, epoch+1)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_u2net_mask_model.pth')
            print(f"✅ New best model saved! Loss: {best_loss:.4f}")

        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint_path = f'u2net_mask_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"📁 Checkpoint saved: {checkpoint_path}")

        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    print("✅ Best model saved as: best_u2net_mask_model.pth")
    print("✅ Training samples saved in: ./training_samples/")
    print("✅ Ready for mask generation!")
    print("="*60)

if __name__ == "__main__":
    main()
