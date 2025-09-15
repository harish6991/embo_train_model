#!/usr/bin/env python3
"""
Direct CycleGAN Training Script for Aligned Embroidery Dataset
This script directly imports and trains the CycleGAN model without subprocess calls.
"""

import os
import sys
import torch
import psutil
from pathlib import Path
from torch.utils.data import DataLoader


# Add current directory to path
sys.path.append('.')
# Import CycleGAN components
from models.cycle_gan_model import CycleGANModel
from utils.align import AlignedEmbroideryDataset


train_dataset = AlignedEmbroideryDataset("./MSEmb_DATASET/embs_all_aligned/train")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

class TrainOptions:
    def __init__(self):
        # Basic parameters
        self.isTrain = True
        self.name = "embroidery_direct"
        self.checkpoints_dir = "./checkpoints"
        
        # Model parameters
        self.input_nc = 3   # input channels (RGB)
        self.output_nc = 3  # output channels (RGB)
        self.ngf = 64       # num of gen filters
        self.ndf = 64       # num of disc filters
        self.netG = 'resnet_9blocks'  # generator arch
        self.netD = 'basic'           # discriminator arch
        self.n_layers_D = 3
        self.norm = 'instance'
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_dropout = True
        
        # Training parameters
        self.gan_mode = 'lsgan'       # or 'vanilla'
        self.lr = 0.0002
        self.beta1 = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch_count = 1
        self.n_epochs = 150
        self.n_epochs_decay = 150
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_identity = 0.5
        self.pool_size = 50
        self.batch_size = 32
        self.direction = 'AtoB'  # you can swap this
        
        # Additional required parameters
        self.phase = 'train'
        self.epoch = 'latest'
        self.load_iter = 0
        self.verbose = False
        self.suffix = ''
        self.use_wandb = False
        self.wandb_project_name = 'CycleGAN-and-pix2pix'
        self.display_freq = 400
        self.update_html_freq = 1000
        self.print_freq = 100
        self.no_html = False
        self.save_latest_freq = 5000
        self.save_epoch_freq = 50
        self.save_by_iter = False
        self.continue_train = False
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.gpu_ids = "0" if torch.cuda.is_available() else "-1"
        
        # Dataset parameters
        self.preprocess = 'resize_and_crop'
        self.load_size = 286
        self.crop_size = 256
        self.no_flip = False
        self.serial_batches = False
        self.max_dataset_size = float("inf")
        self.display_winsize = 256
opt = TrainOptions()

# Create checkpoints directory if it doesn't exist
import os
checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
web_dir = os.path.join(checkpoint_dir, "web")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(web_dir, exist_ok=True)
print(f"✓ Checkpoints directory: {checkpoint_dir}")
print(f"✓ Web visualization directory: {web_dir}")

model = CycleGANModel(opt)
model.setup(opt)


print("=" * 60)
print("STARTING CYCLEGAN TRAINING")
print("=" * 60)
print(f"Dataset size: {len(train_dataset)} images")
print(f"Batch size: {opt.batch_size}")
print(f"Epochs: {opt.n_epochs} + {opt.n_epochs_decay} decay")
print(f"Device: {opt.device}")
print("=" * 60)

total_iters = 0

try:
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        import time
        epoch_start_time = time.time()
        
        # Update learning rates
        model.update_learning_rate()
        
        for i, data in enumerate(train_loader):
            total_iters += opt.batch_size
            
            # Set input and optimize
            model.set_input(data)
            model.optimize_parameters()
            
            # Print progress
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(f'Epoch: {epoch:3d}, Iter: {total_iters:6d}, '
                      f'G_A: {losses.get("G_A", 0):.4f}, '
                      f'D_A: {losses.get("D_A", 0):.4f}, '
                      f'cycle_A: {losses.get("cycle_A", 0):.4f}')
            
            # Save latest model
            if total_iters % opt.save_latest_freq == 0:
                print(f'Saving latest model (epoch {epoch}, iter {total_iters})')
                model.save_networks('latest')
        
        # Save epoch model
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving model at epoch {epoch}')
            model.save_networks('latest')
            model.save_networks(epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f'End of epoch {epoch:3d} / {opt.n_epochs + opt.n_epochs_decay} '
              f'Time: {epoch_time:.1f}s')

except KeyboardInterrupt:
    print(f"\nTraining interrupted by user at epoch {epoch}")
    print("Saving current model...")
    model.save_networks('latest')

except Exception as e:
    print(f"\nTraining failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)
print(f"Model saved in: ./checkpoints/{opt.name}/")
