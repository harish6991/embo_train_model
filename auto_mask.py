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
# IMPROVED Auto-mask function (better for embroidery details)
# ----------------------------
def auto_mask_improved(img):
    """Improved mask generation with better embroidery detail capture"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_uint8 = (gray / gray.max() * 255).astype(np.uint8)

    # Multiple entropy scales for different detail levels
    ent_fine = entropy(gray_uint8, disk(3)).astype(np.float32)  # Fine details
    ent_coarse = entropy(gray_uint8, disk(7)).astype(np.float32)  # Coarse patterns
    
    # Normalize entropy maps
    ent_fine = (ent_fine - ent_fine.min()) / (ent_fine.max() - ent_fine.min() + 1e-9)
    ent_coarse = (ent_coarse - ent_coarse.min()) / (ent_coarse.max() - ent_coarse.min() + 1e-9)

    # Multiple edge detection methods
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.abs(lap)
    lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-9)
    
    # Sobel edges for better embroidery boundaries
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min() + 1e-9)

    # Combine features with better weights for embroidery
    mask = 0.4 * ent_fine + 0.3 * ent_coarse + 0.2 * lap + 0.1 * sobel
    
    # Less aggressive smoothing to preserve details
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Enhanced contrast for sharper masks
    mask = np.power(mask, 0.8)  # Gamma correction for better contrast
    
    return np.clip(mask, 0, 1)


# ----------------------------
# IMPROVED Mask Generator with Skip Connections
# ----------------------------
class ImprovedMaskGenerator(nn.Module):
    """Improved generator with skip connections for better detail preservation"""
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        
        # Encoder with skip connections
        self.enc1 = self._make_layer(input_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Bottleneck
        self.bottleneck = self._make_layer(512, 1024)
        
        # Decoder with skip connections
        self.dec4 = self._make_decoder_layer(1024 + 512, 512)
        self.dec3 = self._make_decoder_layer(512 + 256, 256)
        self.dec2 = self._make_decoder_layer(256 + 128, 128)
        self.dec1 = self._make_decoder_layer(128 + 64, 64)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, 1),
            nn.Sigmoid()
        )
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return self.final(d1)


# ----------------------------
# IMPROVED Dataset
# ----------------------------
class ImprovedMaskDataset:
    """Dataset with improved mask generation"""
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
        
        # Generate improved auto mask
        mask_np = auto_mask_improved(img)  # [H, W]
        
        # Convert to PIL for transforms
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        
        # Apply transforms
        img_tensor = self.transform(img_pil)
        mask_tensor = self.mask_transform(mask_pil)
        
        return img_tensor, mask_tensor


# ----------------------------
# IMPROVED Loss Functions
# ----------------------------
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for better handling of hard examples"""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

def edge_loss(pred, target):
    """Edge-aware loss to preserve boundaries"""
    # Compute gradients
    pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    
    target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    
    edge_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
    edge_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
    
    return edge_loss_x + edge_loss_y

def improved_mask_loss(pred, target):
    """Improved combined loss for better embroidery masks"""
    # Standard losses
    bce = F.binary_cross_entropy(pred, target)
    
    # Dice loss
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    dice_loss = 1 - dice
    
    # Focal loss for hard examples
    focal = focal_loss(pred, target)
    
    # Edge-aware loss
    edge = edge_loss(pred, target)
    
    # Combined loss with better weights
    total_loss = 0.3 * bce + 0.3 * dice_loss + 0.2 * focal + 0.2 * edge
    
    return total_loss


# ----------------------------
# Visualization Functions
# ----------------------------
def save_improved_comparison(input_img, target_mask, pred_mask, epoch, batch_idx, sample_idx, save_dir):
    """Save comparison with better visualization"""
    # Convert tensors to numpy and denormalize
    input_np = input_img.cpu().detach().numpy().transpose(1, 2, 0)
    input_np = (input_np * 0.5 + 0.5)  # Denormalize from [-1,1] to [0,1]
    input_np = np.clip(input_np, 0, 1)
    
    target_np = target_mask.cpu().detach().numpy().squeeze()
    pred_np = pred_mask.cpu().detach().numpy().squeeze()
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(target_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Target (Improved Auto-mask)')
    axes[1].axis('off')
    
    axes[2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('AI Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/improved_epoch_{epoch}_batch_{batch_idx}_sample_{sample_idx}.png', 
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
# MAIN TRAINING SCRIPT
# ----------------------------
def main():
    print("🚀 IMPROVED MASK GENERATOR TRAINING")
    print("="*60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs("improved_checkpoints", exist_ok=True)
    os.makedirs("improved_masks", exist_ok=True)
    os.makedirs("improved_comparisons", exist_ok=True)
    
    # Dataset
    dataset_path = "./MSEmb_DATASET/embs_s_unaligned/train/trainX_e"
    if os.path.exists(dataset_path):
        dataset = ImprovedMaskDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        print(f"✅ Dataset loaded: {len(dataset)} images")
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Model
    model = ImprovedMaskGenerator().to(device)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer with better settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training parameters
    epochs = 100  # Reduced since this model should converge faster
    
    print("\n🎯 Key Improvements:")
    print("✅ U-Net architecture with skip connections")
    print("✅ Multi-scale entropy detection")
    print("✅ Edge-aware + Focal + Dice loss")
    print("✅ Better contrast enhancement")
    print("✅ AdamW optimizer with cosine scheduling")
    
    print(f"\n🏃 Starting training for {epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (images, target_masks) in enumerate(progress_bar):
            images = images.to(device)
            target_masks = target_masks.to(device)
            
            # Forward pass
            pred_masks = model(images)
            loss = improved_mask_loss(pred_masks, target_masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Save visualization every 10 epochs
            if (epoch + 1) % 10 == 0 and batch_idx == 0:
                # Save individual masks
                save_individual_masks(pred_masks, epoch+1, batch_idx, "improved_masks")
                
                # Save first 3 samples comparison
                for i in range(min(3, images.size(0))):
                    save_improved_comparison(
                        images[i], target_masks[i], pred_masks[i], 
                        epoch+1, batch_idx, i, "improved_comparisons"
                    )
                print(f"\n✅ Improved masks and comparisons saved for epoch {epoch+1}")
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'improved_checkpoints/best_improved_mask_generator.pth')
            print(f"✅ New best model saved! Loss: {best_loss:.4f}")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), f'improved_checkpoints/improved_mask_epoch_{epoch+1}.pth')
            print(f"📁 Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), 'improved_checkpoints/improved_mask_final.pth')
    
    print("\n" + "="*60)
    print("🎉 IMPROVED TRAINING COMPLETED!")
    print("="*60)
    print(f"✅ Best loss achieved: {best_loss:.4f}")
    print(f"📁 Models saved in: ./improved_checkpoints/")
    print(f"🖼️ Comparisons saved in: ./improved_comparisons/")
    print("="*60)

if __name__ == "__main__":
    main() 