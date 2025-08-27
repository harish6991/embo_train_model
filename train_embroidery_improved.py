import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import logging
import torch.nn.functional as F
import functools
import random
from math import exp
from sklearn.model_selection import train_test_split
from PIL import Image

# Import custom modules
from utils.align import AlignedEmbroideryDataset
from utils.prefixDataset import PrefixStitchDataset
from models.test_model import UnetGenerator, UnetSkipConnectionBlock
from models.patchGAN import NLayerDiscriminator, MultiScaleDiscriminator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16  # Reduced for better stability
epochs = 350  # Increased epochs for better convergence
lambda_L1 = 200  # Increased L1 weight for sharper images
lambda_perceptual = 30  # Further increased perceptual loss weight
lambda_ssim = 20  # Increased SSIM loss weight for better structural similarity
lambda_adversarial = 0.3  # Reduced adversarial weight further for stability
lr = 0.00003  # Further reduced learning rate for finer tuning
lr_discriminator = 0.00008  # Reduced discriminator learning rate
beta1 = 0.5
beta2 = 0.999
weight_decay = 2e-4  # Increased L2 regularization to reduce artifacts
patience = 25  # Increased patience for better convergence

# Create output directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)

# Enhanced data loading with comprehensive augmentation
class AugmentedEmbroideryDataset(AlignedEmbroideryDataset):
    def __init__(self, root_dir, image_size=256, augment=True):
        super().__init__(root_dir, image_size)
        self.augment = augment

    def rotate_tensor(self, tensor, angle):
        """Rotate tensor by given angle"""
        if angle == 0:
            return tensor

        # Convert to PIL, rotate, convert back
        tensor_np = tensor.cpu().numpy().transpose(1, 2, 0)
        tensor_np = (tensor_np + 1) / 2  # Denormalize to [0, 1]
        tensor_pil = Image.fromarray((tensor_np * 255).astype(np.uint8))
        rotated_pil = tensor_pil.rotate(angle, resample=Image.BILINEAR)
        rotated_np = np.array(rotated_pil).astype(np.float32) / 255.0
        rotated_np = rotated_np * 2 - 1  # Normalize to [-1, 1]
        return torch.from_numpy(rotated_np.transpose(2, 0, 1)).to(tensor.device)

    def scale_tensor(self, tensor, scale):
        """Scale tensor by given factor"""
        if scale == 1.0:
            return tensor

        # Use F.interpolate for scaling
        h, w = tensor.shape[1], tensor.shape[2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Pad or crop to original size
        if scale > 1.0:
            # Crop from center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            scaled = scaled[:, :, start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad with zeros
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            padded = torch.zeros(1, 3, h, w, device=tensor.device)
            padded[:, :, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled
            scaled = padded

        return scaled.squeeze(0)

    def __getitem__(self, idx):
        input_tensor, target_tensor = super().__getitem__(idx)

        if self.augment and random.random() > 0.5:
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                input_tensor = self.rotate_tensor(input_tensor, angle)
                target_tensor = self.rotate_tensor(target_tensor, angle)

            # Random scaling
            if random.random() > 0.5:
                scale = random.uniform(0.8, 1.2)
                input_tensor = self.scale_tensor(input_tensor, scale)
                target_tensor = self.scale_tensor(target_tensor, scale)

            # Random horizontal flip
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, [2])
                target_tensor = torch.flip(target_tensor, [2])

            # Random brightness adjustment
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                input_tensor = torch.clamp(input_tensor * brightness_factor, -1, 1)
                target_tensor = torch.clamp(target_tensor * brightness_factor, -1, 1)

            # Random contrast adjustment
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                input_tensor = torch.clamp((input_tensor - 0.5) * contrast_factor + 0.5, -1, 1)
                target_tensor = torch.clamp((target_tensor - 0.5) * contrast_factor + 0.5, -1, 1)

        return input_tensor, target_tensor

# Enhanced loss functions
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use VGG features for perceptual loss
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).to(device)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.features(x)
        y_features = self.features(y)
        return F.mse_loss(x_features, y_features)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# Data loading with train/validation split
train_dataset = AugmentedEmbroideryDataset("./MSEmb_DATASET/embs_s_aligned/train", augment=True)

# Split dataset into train and validation
train_indices, val_indices = train_test_split(
    range(len(train_dataset)),
    test_size=0.2,
    random_state=42
)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)

logger.info(f"Training samples: {len(train_indices)}")
logger.info(f"Validation samples: {len(val_indices)}")

# Model initialization
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

# Initialize models
generator = UnetGenerator(
    input_nc=3,
    output_nc=3,
    num_downs=8,
    ngf=64,
    norm_layer=nn.BatchNorm2d,
    use_dropout=True
).to(device)

discriminator = MultiScaleDiscriminator(
    input_nc=6
).to(device)

# Load existing best model if available - Updated for model continuation
existing_generator_path = "checkpoints/best_generator.pth"
existing_discriminator_path = "checkpoints/embroidery_improved_best.pth"  # Use the best checkpoint

# Check if we want to start fresh or continue training
start_fresh = True  # Set to False if you want to continue from existing checkpoints

if start_fresh or not os.path.exists(existing_generator_path):
    logger.info("Starting fresh training - initializing generator with random weights")
    init_weights(generator)
else:
    logger.info(f"Loading existing generator from: {existing_generator_path}")
    try:
        generator.load_state_dict(torch.load(existing_generator_path, map_location=device, weights_only=False))
        logger.info("Existing generator loaded successfully for continued training!")
    except Exception as e:
        logger.warning(f"Could not load existing generator: {e}")
        logger.info("Initializing generator with random weights")
        init_weights(generator)

if start_fresh or not os.path.exists(existing_discriminator_path):
    logger.info("Starting fresh training - initializing discriminator with random weights")
    init_weights(discriminator)
else:
    logger.info(f"Loading existing discriminator from: {existing_discriminator_path}")
    try:
        checkpoint = torch.load(existing_discriminator_path, map_location=device, weights_only=False)
        if 'discriminator_state_dict' in checkpoint:
            # Try to load the state dict, but handle potential size mismatch
            try:
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                logger.info("Existing discriminator loaded successfully for continued training!")
            except RuntimeError as e:
                logger.warning(f"Could not load existing discriminator due to size mismatch: {e}")
                logger.info("Initializing discriminator with random weights")
                init_weights(discriminator)
        else:
            logger.info("No discriminator state dict found in checkpoint")
            init_weights(discriminator)
    except Exception as e:
        logger.warning(f"Could not load existing discriminator checkpoint: {e}")
        logger.info("Initializing discriminator with random weights")
        init_weights(discriminator)

# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
criterion_perceptual = PerceptualLoss()
criterion_ssim = SSIMLoss()

# Optimizers with weight decay (L2 regularization)
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(beta1, beta2), weight_decay=weight_decay)

# Enhanced learning rate schedulers
def lambda_rule(epoch):
    # More aggressive learning rate reduction
    if epoch < 50:
        return 1.0
    elif epoch < 100:
        return 0.5
    elif epoch < 150:
        return 0.2
    elif epoch < 200:
        return 0.1
    else:
        return 0.05

scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

# Training function with enhanced losses and validation
def train_epoch(epoch):
    generator.train()
    discriminator.train()

    total_g_loss = 0
    total_d_loss = 0
    total_l1_loss = 0
    total_perceptual_loss = 0
    total_ssim_loss = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

    for batch_idx, (input_images, target_images) in enumerate(progress_bar):
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        # Create labels for GAN loss - we'll create them dynamically based on actual output sizes
        batch_size = input_images.size(0)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real images
        real_output_16, real_output_35, real_output_70 = discriminator(torch.cat([input_images, target_images], dim=1))

        # Create labels dynamically based on actual output sizes - LSGAN labels with smoothing
        real_labels_16 = torch.ones_like(real_output_16) * 0.9  # Label smoothing
        fake_labels_16 = torch.zeros_like(real_output_16) + 0.1  # Label smoothing
        real_labels_35 = torch.ones_like(real_output_35) * 0.9  # Label smoothing
        fake_labels_35 = torch.zeros_like(real_output_35) + 0.1  # Label smoothing
        real_labels_70 = torch.ones_like(real_output_70) * 0.9  # Label smoothing
        fake_labels_70 = torch.zeros_like(real_output_70) + 0.1  # Label smoothing

        d_loss_real_16 = criterion_GAN(real_output_16, real_labels_16)
        d_loss_real_35 = criterion_GAN(real_output_35, real_labels_35)
        d_loss_real_70 = criterion_GAN(real_output_70, real_labels_70)
        d_loss_real = (d_loss_real_16 + d_loss_real_35 + d_loss_real_70) / 3

        # Fake images
        fake_images = generator(input_images)
        fake_output_16, fake_output_35, fake_output_70 = discriminator(torch.cat([input_images, fake_images.detach()], dim=1))
        d_loss_fake_16 = criterion_GAN(fake_output_16, fake_labels_16)
        d_loss_fake_35 = criterion_GAN(fake_output_35, fake_labels_35)
        d_loss_fake_70 = criterion_GAN(fake_output_70, fake_labels_70)
        d_loss_fake = (d_loss_fake_16 + d_loss_fake_35 + d_loss_fake_70) / 3

        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        # Adversarial loss with reduced weight and emphasis on fine details
        fake_output_16, fake_output_35, fake_output_70 = discriminator(torch.cat([input_images, fake_images], dim=1))
        g_loss_gan_16 = criterion_GAN(fake_output_16, real_labels_16)  # Use smoothed labels
        g_loss_gan_35 = criterion_GAN(fake_output_35, real_labels_35)  # Use smoothed labels
        g_loss_gan_70 = criterion_GAN(fake_output_70, real_labels_70)  # Use smoothed labels
        # Give slightly more weight to 16x16 scale for fine detail emphasis
        g_loss_gan = (1.2 * g_loss_gan_16 + g_loss_gan_35 + g_loss_gan_70) / 3.2

        # Feature matching loss for better training stability
        with torch.no_grad():
            real_output_16, real_output_35, real_output_70 = discriminator(torch.cat([input_images, target_images], dim=1))

        # Feature matching loss - encourage generator to match discriminator features at all scales
        feature_matching_loss = F.mse_loss(fake_output_16, real_output_16.detach()) + \
                               F.mse_loss(fake_output_35, real_output_35.detach()) + \
                               F.mse_loss(fake_output_70, real_output_70.detach())

        # L1 loss
        g_loss_l1 = criterion_L1(fake_images, target_images)

        # Perceptual loss
        g_loss_perceptual = criterion_perceptual(fake_images, target_images)

        # SSIM loss
        g_loss_ssim = criterion_ssim(fake_images, target_images)

        # Total generator loss with enhanced weights including feature matching
        g_loss = lambda_adversarial * g_loss_gan + lambda_L1 * g_loss_l1 + \
                 lambda_perceptual * g_loss_perceptual + lambda_ssim * g_loss_ssim + \
                 0.05 * feature_matching_loss  # Reduced weight for feature matching to reduce artifacts
        g_loss.backward()
        optimizer_G.step()

        # Update progress bar
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_l1_loss += g_loss_l1.item()
        total_perceptual_loss += g_loss_perceptual.item()
        total_ssim_loss += g_loss_ssim.item()

        progress_bar.set_postfix({
            'G_Loss': f'{g_loss.item():.4f}',
            'D_Loss': f'{d_loss.item():.4f}',
            'L1': f'{g_loss_l1.item():.4f}',
            'Perceptual': f'{g_loss_perceptual.item():.4f}',
            'SSIM': f'{g_loss_ssim.item():.4f}'
        })

    # Update learning rates
    scheduler_G.step()
    scheduler_D.step()

    return (total_g_loss / len(train_loader), total_d_loss / len(train_loader),
            total_l1_loss / len(train_loader), total_perceptual_loss / len(train_loader),
            total_ssim_loss / len(train_loader))

# Validation function
def validate_epoch(generator, val_loader):
    generator.eval()
    total_val_loss = 0

    with torch.no_grad():
        for input_images, target_images in val_loader:
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            fake_images = generator(input_images)
            val_loss = criterion_L1(fake_images, target_images)
            total_val_loss += val_loss.item()

    generator.train()
    return total_val_loss / len(val_loader)

def save_sample_images(input_images, target_images, fake_images, epoch, batch_idx):
    """Save sample images for visualization"""
    # Convert tensors to numpy arrays
    input_np = input_images[0].cpu().detach().numpy().transpose(1, 2, 0)
    target_np = target_images[0].cpu().detach().numpy().transpose(1, 2, 0)
    fake_np = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)

    # Normalize to [0, 1] range
    input_np = (input_np + 1) / 2
    target_np = (target_np + 1) / 2
    fake_np = (fake_np + 1) / 2

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_np)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(target_np)
    axes[1].set_title('Target')
    axes[1].axis('off')

    axes[2].imshow(fake_np)
    axes[2].set_title('Generated')
    axes[2].axis('off')

    plt.savefig(f'sample_images/epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

def save_sample_images_epoch(generator, dataset, epoch):
    """Save sample images after each epoch completion"""
    generator.eval()
    with torch.no_grad():
        # Get a few sample images from the dataset
        num_samples = min(3, len(dataset))
        sample_indices = [0, len(dataset)//2, len(dataset)-1] if len(dataset) >= 3 else [0]

        for i, idx in enumerate(sample_indices):
            input_img, target_img = dataset[idx]
            input_img = input_img.unsqueeze(0).to(device)

            # Generate fake image
            fake_img = generator(input_img)

            # Convert tensors to numpy arrays
            input_np = input_img[0].cpu().numpy().transpose(1, 2, 0)
            target_np = target_img.numpy().transpose(1, 2, 0)
            fake_np = fake_img[0].cpu().numpy().transpose(1, 2, 0)

            # Normalize to [0, 1] range
            input_np = (input_np + 1) / 2
            target_np = (target_np + 1) / 2
            fake_np = (fake_np + 1) / 2

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(input_np)
            axes[0].set_title('Input')
            axes[0].axis('off')

            axes[1].imshow(target_np)
            axes[1].set_title('Target')
            axes[1].axis('off')

            axes[2].imshow(fake_np)
            axes[2].set_title('Generated')
            axes[2].axis('off')

            plt.savefig(f'sample_images/epoch_{epoch}_sample_{i}.png')
            plt.close()

    generator.train()
    logger.info(f"Sample images saved for epoch {epoch+1}")

def save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, losses, prefix='epoch'):
    """Save model checkpoints"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'g_loss': losses[0],
        'd_loss': losses[1],
        'l1_loss': losses[2],
        'perceptual_loss': losses[3],
        'ssim_loss': losses[4]
    }

    filename = f'checkpoints/embroidery_improved_{prefix}_{epoch}.pth'
    torch.save(checkpoint, filename)
    logger.info(f'Checkpoint saved: {filename}')

def save_best_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, losses):
    """Save best model checkpoint with fixed filename (overwrites previous best)"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'g_loss': losses[0],
        'd_loss': losses[1],
        'l1_loss': losses[2],
        'perceptual_loss': losses[3],
        'ssim_loss': losses[4]
    }

    # Always use the same filename for best checkpoint
    filename = 'checkpoints/embroidery_improved_best.pth'
    torch.save(checkpoint, filename)
    logger.info(f'Best checkpoint saved: {filename}')

# Main training loop with early stopping
def main():
    logger.info(f"Starting improved training on device: {device}")
    logger.info(f"Dataset: embs_f_aligned/train")
    logger.info(f"Number of training samples: {len(train_indices)}")
    logger.info(f"Number of validation samples: {len(val_indices)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {epochs}")
    logger.info(f"Lambda L1: {lambda_L1}")
    logger.info(f"Lambda Perceptual: {lambda_perceptual}")
    logger.info(f"Lambda SSIM: {lambda_ssim}")
    logger.info(f"Lambda Adversarial: {lambda_adversarial}")
    logger.info(f"Generator Learning Rate: {lr}")
    logger.info(f"Discriminator Learning Rate: {lr_discriminator}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info("Using LSGAN with enhanced losses, label smoothing, elastic augmentation...")
    logger.info("Multi-scale discriminator: 16x16, 35x35, 70x70 receptive fields...")
    logger.info("Continuing training with existing model weights for refinement...")

    best_l1_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # Train one epoch
        avg_g_loss, avg_d_loss, avg_l1_loss, avg_perceptual_loss, avg_ssim_loss = train_epoch(epoch)

        # Validate
        val_loss = validate_epoch(generator, val_loader)

        logger.info(f"Epoch {epoch+1} - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
        logger.info(f"L1: {avg_l1_loss:.6f}, Perceptual: {avg_perceptual_loss:.6f}, SSIM: {avg_ssim_loss:.6f}")
        logger.info(f"Validation L1 Loss: {val_loss:.6f}")

        # Save sample images after each epoch completion
        save_sample_images_epoch(generator, train_dataset, epoch)

        # Early stopping logic
        if avg_l1_loss < best_l1_loss:
            best_l1_loss = avg_l1_loss
            patience_counter = 0
            # Save best model
            torch.save(generator.state_dict(), 'checkpoints/best_generator.pth')
            # Save best checkpoint with fixed filename (overwrites previous best)
            save_best_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D,
                               (avg_g_loss, avg_d_loss, avg_l1_loss, avg_perceptual_loss, avg_ssim_loss))
            logger.info(f"Updated best model with L1 loss: {best_l1_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} - no improvement for {patience} epochs")
            break

        # Save first epoch model
        if epoch == 0:
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D,
                          (avg_g_loss, avg_d_loss, avg_l1_loss, avg_perceptual_loss, avg_ssim_loss))
            logger.info(f"First epoch model saved")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D,
                          (avg_g_loss, avg_d_loss, avg_l1_loss, avg_perceptual_loss, avg_ssim_loss), prefix='epoch')
            logger.info(f"Checkpoint saved every 20 epochs at epoch {epoch + 1}")

        # Save last epoch model
        if epoch == epochs - 1:
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D,
                          (avg_g_loss, avg_d_loss, avg_l1_loss, avg_perceptual_loss, avg_ssim_loss), prefix='final')
            logger.info(f"Final epoch model saved")

if __name__ == "__main__":
    main()
