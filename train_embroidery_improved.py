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
import torchvision.models as models

# Import custom modules
from utils.align import AlignedEmbroideryDataset
from models.test_model import UnetGenerator, UnetSkipConnectionBlock
from models.patchGAN import NLayerDiscriminator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32  # Reduced for better stability
epochs =  375 # More epochs for better convergence
lr = 0.0001  # Slightly lower learning rate
beta1 = 0.5
beta2 = 0.999

# Create output directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)

# Data loading with augmentation
class AugmentedEmbroideryDataset(AlignedEmbroideryDataset):
    def __init__(self, root_dir, image_size=256, augment=True):
        super().__init__(root_dir, image_size)
        self.augment = augment

    def __getitem__(self, idx):
        input_tensor, target_tensor = super().__getitem__(idx)

        if self.augment and random.random() > 0.5:
            # Random horizontal flip
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, [2])
                target_tensor = torch.flip(target_tensor, [2])

            # Random brightness adjustment
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                input_tensor = torch.clamp(input_tensor * brightness_factor, -1, 1)
                target_tensor = torch.clamp(target_tensor * brightness_factor, -1, 1)

        return input_tensor, target_tensor

# Enhanced loss functions
class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features to capture high-level semantic similarity"""
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

        # Load pretrained VGG16 and freeze
        vgg = models.vgg16(pretrained=True).features.to(self.device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Extract features from multiple layers
        self.feature_layers = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[:9]),   # relu2_2
            nn.Sequential(*list(vgg.children())[:16]),  # relu3_3
            nn.Sequential(*list(vgg.children())[:23]),  # relu4_3
        ]).to(self.device)

    def forward(self, pred, target):
        # Normalize inputs to VGG expected range [0, 1]
        pred_norm = (pred + 1) / 2  # Convert from [-1, 1] to [0, 1]
        target_norm = (target + 1) / 2

        loss = 0
        for layer in self.feature_layers:
            pred_feat = layer(pred_norm)
            target_feat = layer(target_norm)

            # Normalize features to prevent explosion
            pred_feat = F.normalize(pred_feat.view(pred_feat.size(0), -1), dim=1)
            target_feat = F.normalize(target_feat.view(target_feat.size(0), -1), dim=1)

            layer_loss = F.mse_loss(pred_feat, target_feat)

            # Check for NaN and clip if necessary
            if torch.isnan(layer_loss) or torch.isinf(layer_loss):
                layer_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            loss += layer_loss

        # Clip the total loss to prevent explosion
        loss = torch.clamp(loss, 0, 10.0)
        return loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, device=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1  # default channel
        self.device = device if device is not None else torch.device("cpu")
        self.window = self.create_window(window_size, self.channel).to(self.device)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2*sigma**2))
                              for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # Recreate window if channels differ
        if channel != self.channel or self.window.device != img1.device:
            self.window = self.create_window(self.window_size, channel).to(img1.device)
            self.channel = channel

        ssim_result = self._ssim(img1, img2, self.window, self.window_size, channel, self.size_average)

        # Check for NaN and handle gracefully
        if torch.isnan(ssim_result) or torch.isinf(ssim_result):
            return torch.tensor(0.0, device=img1.device, requires_grad=True)

        return 1 - ssim_result


class EdgeAwareLoss(nn.Module):
    """Enhanced edge-aware loss to preserve sharp details and crispness"""
    def __init__(self):
        super().__init__()
        # Multiple edge detection kernels for better crispness detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Laplacian kernel for additional edge detection
        self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        # High-pass filter for fine detail preservation
        self.highpass = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, target):
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.laplacian = self.laplacian.to(device)
        self.highpass = self.highpass.to(device)

        # Convert to grayscale if needed
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target

        # Multiple edge detection methods for comprehensive crispness detection
        # 1. Sobel edges
        pred_edges_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_sobel = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-8)

        target_edges_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_sobel = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-8)

        # 2. Laplacian edges
        pred_laplacian = torch.abs(F.conv2d(pred_gray, self.laplacian, padding=1))
        target_laplacian = torch.abs(F.conv2d(target_gray, self.laplacian, padding=1))

        # 3. High-pass filter for fine details
        pred_highpass = torch.abs(F.conv2d(pred_gray, self.highpass, padding=1))
        target_highpass = torch.abs(F.conv2d(target_gray, self.highpass, padding=1))

        # Combine all edge detection methods
        pred_edges = pred_sobel + pred_laplacian + pred_highpass
        target_edges = target_sobel + target_laplacian + target_highpass

        # Normalize edges to prevent explosion but maintain sensitivity
        pred_edges = F.normalize(pred_edges.view(pred_edges.size(0), -1), dim=1)
        target_edges = F.normalize(target_edges.view(target_edges.size(0), -1), dim=1)

        # Multi-scale edge loss for better crispness detection
        edge_loss_sobel = F.mse_loss(pred_sobel.view(pred_sobel.size(0), -1), target_sobel.view(target_sobel.size(0), -1))
        edge_loss_laplacian = F.mse_loss(pred_laplacian.view(pred_laplacian.size(0), -1), target_laplacian.view(target_laplacian.size(0), -1))
        edge_loss_highpass = F.mse_loss(pred_highpass.view(pred_highpass.size(0), -1), target_highpass.view(target_highpass.size(0), -1))

        # Combined edge loss with weights
        edge_loss = edge_loss_sobel + 1.5 * edge_loss_laplacian + 2.0 * edge_loss_highpass

        # Check for NaN and clip if necessary
        if torch.isnan(edge_loss) or torch.isinf(edge_loss):
            edge_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Clip the loss to prevent explosion but allow meaningful gradients
        edge_loss = torch.clamp(edge_loss, 0, 10.0)
        return edge_loss


class ComprehensiveEmbroideryLoss(nn.Module):
    """Combined loss function for embroidery generation that captures both structure and details"""
    def __init__(self, device=None, ssim_weight=0.1, perceptual_weight=0.1,
                 edge_weight=0.4, l1_weight=0.3, color_weight=0.1):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

        # Individual loss functions
        self.ssim_loss = SSIMLoss(device=self.device)
        self.perceptual_loss = PerceptualLoss(device=self.device)
        self.edge_loss = EdgeAwareLoss()
        self.l1_loss = nn.L1Loss()

        # NEW: Color preservation loss
        self.color_loss = nn.MSELoss()

        # Weights for combining losses - PRIORITIZE EDGE LOSS AND COLOR PRESERVATION
        self.ssim_weight = ssim_weight      # Reduced from 0.1 to 0.1
        self.perceptual_weight = perceptual_weight  # Reduced from 0.1 to 0.1
        self.edge_weight = edge_weight      # Reduced from 0.6 to 0.4
        self.l1_weight = l1_weight         # Increased from 0.2 to 0.3
        self.color_weight = color_weight    # NEW: 0.1 for color preservation

    def forward(self, pred, target):
        # Calculate individual losses with error handling
        try:
            ssim_loss = self.ssim_loss(pred, target)
            perceptual_loss = self.perceptual_loss(pred, target)
            edge_loss = self.edge_loss(pred, target)
            l1_loss = self.l1_loss(pred, target)

            # NEW: Color preservation loss - separate RGB channels
            if pred.size(1) == 3:  # RGB image
                # Calculate color loss for each channel separately
                color_loss_r = F.mse_loss(pred[:, 0:1], target[:, 0:1])
                color_loss_g = F.mse_loss(pred[:, 1:2], target[:, 1:2])
                color_loss_b = F.mse_loss(pred[:, 2:3], target[:, 2:3])
                color_loss = color_loss_r + color_loss_g + color_loss_b
            else:
                color_loss = F.mse_loss(pred, target)

            # DEBUG: Print raw loss values to identify the issue
            if torch.rand(1).item() < 0.1:  # Print 10% of the time
                print(f"DEBUG - Raw losses: SSIM={ssim_loss.item():.6f}, Perceptual={perceptual_loss.item():.6f}, Edge={edge_loss.item():.6f}, L1={l1_loss.item():.6f}, Color={color_loss.item():.6f}")
                print(f"DEBUG - Pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}], Target range: [{target.min().item():.3f}, {target.max().item():.3f}]")
                print(f"DEBUG - Pred mean: {pred.mean().item():.3f}, Target mean: {target.mean().item():.3f}")
                if pred.size(1) == 3:
                    print(f"DEBUG - Pred RGB means: R={pred[:, 0].mean().item():.3f}, G={pred[:, 1].mean().item():.3f}, B={pred[:, 2].mean().item():.3f}")
                    print(f"DEBUG - Target RGB means: R={target[:, 0].mean().item():.3f}, G={target[:, 1].mean().item():.3f}, B={target[:, 2].mean().item():.3f}")

            # Check for NaN in individual losses
            if torch.isnan(ssim_loss) or torch.isinf(ssim_loss):
                ssim_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isnan(perceptual_loss) or torch.isinf(perceptual_loss):
                perceptual_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isnan(edge_loss) or torch.isinf(edge_loss):
                edge_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                l1_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isnan(color_loss) or torch.isinf(color_loss):
                color_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # AGGRESSIVE LOSS SCALING - PRIORITIZE COLOR AND CRISPNESS
            ssim_loss = ssim_loss * 30.0      # Reduced scaling
            perceptual_loss = perceptual_loss * 30.0  # Reduced scaling
            edge_loss = edge_loss * 300.0      # HIGH scaling for crispness
            l1_loss = l1_loss * 400.0         # VERY HIGH scaling for pixel accuracy
            color_loss = color_loss * 500.0    # HIGHEST scaling for color preservation

            # Combine losses with weights - COLOR AND EDGE DOMINATE
            total_loss = (self.ssim_weight * ssim_loss +
                         self.perceptual_weight * perceptual_loss +
                         self.edge_weight * edge_loss +
                         self.l1_weight * l1_loss +
                         self.color_weight * color_loss)

            # Final safety check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Clip total loss to prevent explosion
            total_loss = torch.clamp(total_loss, 0, 2000.0)  # Increased range for color + edge loss

        except Exception as e:
            print(f"Error in loss calculation: {e}")
            # Return safe default values
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            ssim_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            perceptual_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            edge_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            l1_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            color_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_loss, {
            'ssim_loss': ssim_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'edge_loss': edge_loss.item(),
            'l1_loss': l1_loss.item(),
            'color_loss': color_loss.item(),
            'total_loss': total_loss.item()
        }

# Data loading - Updated to use embs_t_aligned dataset
train_dataset = AugmentedEmbroideryDataset("./MSEmb_DATASET_SAMPLE/embs_t_aligned/train", augment=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

discriminator = NLayerDiscriminator(
    input_nc=6,
    ndf=64,
    n_layers=3,
    norm_layer=nn.BatchNorm2d
).to(device)

# Load existing best model if available - Updated for model continuation
existing_generator_path = "checkpoints/best_generator.pth"
existing_discriminator_path = "checkpoints/embroidery_improved_final_199.pth"  # Use the final checkpoint

if os.path.exists(existing_generator_path):
    logger.info(f"Loading existing generator from: {existing_generator_path}")
    generator.load_state_dict(torch.load(existing_generator_path, map_location=device))
    logger.info("Existing generator loaded successfully for continued training!")
else:
    logger.info("No existing generator found, initializing with random weights")
    init_weights(generator)

if os.path.exists(existing_discriminator_path):
    logger.info(f"Loading existing discriminator from: {existing_discriminator_path}")
    checkpoint = torch.load(existing_discriminator_path, map_location=device)
    if 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        logger.info("Existing discriminator loaded successfully for continued training!")
    else:
        logger.info("No discriminator state dict found in checkpoint")
        init_weights(discriminator)
else:
    logger.info("No existing discriminator found, initializing with random weights")
    init_weights(discriminator)

# Loss functions - Now using the comprehensive loss
criterion_GAN = nn.BCEWithLogitsLoss()
comprehensive_loss = ComprehensiveEmbroideryLoss(device=device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Learning rate schedulers with better scheduling
def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch - 100) / 100
    return max(lr_l, 0.1)

scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

# Training function with enhanced losses
def train_epoch(epoch):
    generator.train()
    discriminator.train()

    total_g_loss = 0
    total_d_loss = 0
    total_comprehensive_loss = 0
    total_ssim_loss = 0
    total_perceptual_loss = 0
    total_edge_loss = 0
    total_l1_loss = 0
    total_color_loss = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

    for batch_idx, (input_images, target_images) in enumerate(progress_bar):
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        # Create labels for GAN loss
        batch_size = input_images.size(0)
        real_labels = torch.ones(batch_size, 1, 30, 30).to(device)
        fake_labels = torch.zeros(batch_size, 1, 30, 30).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real images
        real_output = discriminator(torch.cat([input_images, target_images], dim=1))
        d_loss_real = criterion_GAN(real_output, real_labels)

        # Fake images
        fake_images = generator(input_images)
        fake_output = discriminator(torch.cat([input_images, fake_images.detach()], dim=1))
        d_loss_fake = criterion_GAN(fake_output, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) * 0.5

        # Check for NaN in discriminator loss
        if torch.isnan(d_loss) or torch.isinf(d_loss):
            print(f"Warning: NaN/Inf in discriminator loss at batch {batch_idx}")
            continue

        d_loss.backward()

        # Clip discriminator gradients
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        # Adversarial loss
        fake_output = discriminator(torch.cat([input_images, fake_images], dim=1))
        g_loss_gan = criterion_GAN(fake_output, real_labels)

        # Comprehensive loss (includes SSIM, Perceptual, Edge, L1, and Color)
        g_loss_comprehensive, loss_breakdown = comprehensive_loss(fake_images, target_images)

        # DEBUG: Check if generator is actually learning and producing different outputs
        if batch_idx % 10 == 0:  # Check every 10 batches
            with torch.no_grad():
                # Check if fake_images are different from target_images
                pixel_diff = torch.abs(fake_images - target_images).mean().item()
                print(f"DEBUG - Batch {batch_idx}: Pixel difference = {pixel_diff:.6f}")

                # Check if fake_images are changing between batches
                if hasattr(train_epoch, 'last_fake_mean'):
                    change = abs(fake_images.mean().item() - train_epoch.last_fake_mean)
                    print(f"DEBUG - Batch {batch_idx}: Change from last batch = {change:.6f}")
                train_epoch.last_fake_mean = fake_images.mean().item()

        # Check for NaN in comprehensive loss
        if torch.isnan(g_loss_comprehensive) or torch.isinf(g_loss_comprehensive):
            print(f"Warning: NaN/Inf in comprehensive loss at batch {batch_idx}")
            print(f"Loss breakdown: {loss_breakdown}")
            continue

        # Total generator loss
        g_loss = g_loss_gan + g_loss_comprehensive

        # Check for NaN in total generator loss
        if torch.isnan(g_loss) or torch.isinf(g_loss):
            print(f"Warning: NaN/Inf in total generator loss at batch {batch_idx}")
            continue

        g_loss.backward()

        # Clip generator gradients
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

        optimizer_G.step()

        # Update progress bar
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_comprehensive_loss += g_loss_comprehensive.item()
        total_ssim_loss += loss_breakdown['ssim_loss']
        total_perceptual_loss += loss_breakdown['perceptual_loss']
        total_edge_loss += loss_breakdown['edge_loss']
        total_l1_loss += loss_breakdown['l1_loss']
        total_color_loss += loss_breakdown['color_loss']

        progress_bar.set_postfix({
            'G_Loss': f'{g_loss.item():.4f}',
            'D_Loss': f'{d_loss.item():.4f}',
            'Comprehensive': f'{g_loss_comprehensive.item():.4f}',
            'Edge': f'{loss_breakdown["edge_loss"]:.4f}',
            'Color': f'{loss_breakdown["color_loss"]:.4f}',
            'L1': f'{loss_breakdown["l1_loss"]:.4f}'
        })

        # Save sample images every 50 batches
        if batch_idx % 50 == 0:
            save_sample_images(input_images, target_images, fake_images, epoch, batch_idx)

    # Update learning rates
    scheduler_G.step()
    scheduler_D.step()

    return (total_g_loss / len(train_loader), total_d_loss / len(train_loader),
            total_comprehensive_loss / len(train_loader), total_ssim_loss / len(train_loader),
            total_perceptual_loss / len(train_loader), total_edge_loss / len(train_loader),
            total_l1_loss / len(train_loader), total_color_loss / len(train_loader))

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
        'comprehensive_loss': losses[2],
        'ssim_loss': losses[3],
        'perceptual_loss': losses[4],
        'edge_loss': losses[5],
        'l1_loss': losses[6],
        'color_loss': losses[7]
    }

    filename = f'checkpoints/embroidery_improved_{prefix}_{epoch}.pth'
    torch.save(checkpoint, filename)
    logger.info(f'Checkpoint saved: {filename}')

# Main training loop
def main():
    logger.info(f"Starting improved training on device: {device}")
    logger.info(f"Dataset: embs_t_aligned/train")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {epochs}")
    logger.info(f"Using ComprehensiveEmbroideryLoss with weights:")
    logger.info(f"  Edge: {comprehensive_loss.edge_weight}")
    logger.info(f"  Perceptual: {comprehensive_loss.perceptual_weight}")
    logger.info(f"  SSIM: {comprehensive_loss.ssim_weight}")
    logger.info(f"  L1: {comprehensive_loss.l1_weight}")
    logger.info("Continuing training with existing model weights...")

    best_comprehensive_loss = float('inf')

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # Train one epoch
        avg_g_loss, avg_d_loss, avg_comprehensive_loss, avg_ssim_loss, avg_perceptual_loss, avg_edge_loss, avg_l1_loss, avg_color_loss = train_epoch(epoch)

        logger.info(f"Epoch {epoch+1} - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
        logger.info(f"Comprehensive: {avg_comprehensive_loss:.6f}")
        logger.info(f"SSIM: {avg_ssim_loss:.6f}, Perceptual: {avg_perceptual_loss:.6f}")
        logger.info(f"Edge: {avg_edge_loss:.6f}, L1: {avg_l1_loss:.6f}, Color: {avg_color_loss:.6f}")

        # Save first epoch model
        if epoch == 0:
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D,
                          (avg_g_loss, avg_d_loss, avg_comprehensive_loss, avg_ssim_loss, avg_perceptual_loss, avg_edge_loss, avg_l1_loss, avg_color_loss))
            logger.info(f"First epoch model saved")

        # Save best model based on comprehensive loss
        if avg_comprehensive_loss < best_comprehensive_loss:
            best_comprehensive_loss = avg_comprehensive_loss
            torch.save(generator.state_dict(), 'checkpoints/best_generator.pth')
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D,
                          (avg_g_loss, avg_d_loss, avg_comprehensive_loss, avg_ssim_loss, avg_perceptual_loss, avg_edge_loss, avg_l1_loss, avg_color_loss), prefix='best')
            logger.info(f"Updated best model with comprehensive loss: {best_comprehensive_loss:.6f}")

        # Save last epoch model
        if epoch == epochs - 1:
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D,
                          (avg_g_loss, avg_d_loss, avg_comprehensive_loss, avg_ssim_loss, avg_perceptual_loss, avg_edge_loss, avg_l1_loss, avg_color_loss), prefix='final')
            logger.info(f"Final epoch model saved")

if __name__ == "__main__":
    main()
