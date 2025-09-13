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

# Add current directory to path
sys.path.append('.')

# Import CycleGAN components
from models.cycle_gan_model import CycleGANModel
from data import create_dataset
from options.train_options import TrainOptions

def create_options():
    """Create training options for aligned embroidery dataset."""
    # Create a custom options object without command line parsing
    class CustomOptions:
        def __init__(self):
            # Basic parameters
            self.dataroot = r"MSEmb_DATASET/embs_s_aligned/"
            self.name = "embroidery_direct"
            self.checkpoints_dir = "./checkpoints"
            
            # Model parameters
            self.model = "cycle_gan"
            self.input_nc = 3
            self.output_nc = 3
            self.ngf = 64
            self.ndf = 64
            self.netD = "basic"
            self.netG = "resnet_9blocks"
            self.n_layers_D = 3
            self.norm = "instance"
            self.init_type = "normal"
            self.init_gain = 0.02
            self.no_dropout = True
            
            # Dataset parameters
            self.dataset_mode = "aligned"
            self.direction = "AtoB"
            self.serial_batches = False
            self.num_threads = 8
            self.batch_size = 4
            self.load_size = 572
            self.crop_size = 512
            self.max_dataset_size = float("inf")
            self.preprocess = "scale_width_and_crop"
            self.no_flip = False
            self.display_winsize = 512
            
            # Additional parameters
            self.epoch = "latest"
            self.load_iter = 0
            self.verbose = False
            self.suffix = ""
            self.use_wandb = False
            self.wandb_project_name = "CycleGAN-and-pix2pix"
            
            # Training parameters
            self.display_freq = 50
            self.update_html_freq = 500
            self.print_freq = 50
            self.no_html = False
            self.save_latest_freq = 500
            self.save_epoch_freq = 30
            self.save_by_iter = False
            self.continue_train = False
            self.epoch_count = 1
            self.phase = "train"
            self.n_epochs = 200
            self.n_epochs_decay = 200
            self.beta1 = 0.5
            self.lr = 0.0002
            self.gan_mode = "lsgan"
            self.pool_size = 100
            self.lr_policy = "linear"
            self.lr_decay_iters = 50
            
            # CycleGAN specific parameters
            self.lambda_A = 15.0
            self.lambda_B = 15.0
            self.lambda_identity = 1.0
            
            # Device settings
            self.isTrain = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.gpu_ids = "0" if torch.cuda.is_available() else "-1"
    
    opt = CustomOptions()
    
    return opt

def train_cyclegan():
    """Main training function."""
    print("=" * 60)
    print("DIRECT CYCLEGAN TRAINING FOR EMBROIDERY")
    print("=" * 60)
    
    # Create options
    opt = create_options()
    
    # Check system resources
    print(f"✓ System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"✓ Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("⚠ CUDA not available, using CPU")
        opt.gpu_ids = "-1"
    
    # Create dataset
    print(f"\n1. Creating dataset from: {opt.dataroot}")
    try:
        dataset = create_dataset(opt)
        dataset_size = len(dataset)
        print(f"✓ Dataset created: {dataset_size} images")
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return
    
    # Create model
    print(f"\n2. Creating CycleGAN model...")
    try:
        model = CycleGANModel(opt)
        model.setup(opt)
        print("✓ Model created and setup complete")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return
    
    # Training loop
    print(f"\n3. Starting training...")
    print(f"   Epochs: {opt.n_epochs} + {opt.n_epochs_decay} decay")
    print(f"   Batch size: {opt.batch_size} (memory optimized for 512x512)")
    print(f"   Image size: {opt.crop_size}x{opt.crop_size} (high resolution)")
    print(f"   Data threads: {opt.num_threads}")
    print(f"   Preprocessing: {opt.preprocess}")
    print("=" * 60)
    
    total_iters = 0
    
    try:
        for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1):
            epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if epoch_start_time:
                epoch_start_time.record()
            
            # Update learning rates
            model.update_learning_rate()
            
            for i, data in enumerate(dataset):
                total_iters += opt.batch_size
                epoch_iter = total_iters - dataset_size * (epoch - 1)
                
                # Set input and optimize
                model.set_input(data)
                model.optimize_parameters()
                
                # Print progress with memory monitoring
                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    memory_used = psutil.virtual_memory().used / (1024**3)
                    memory_percent = psutil.virtual_memory().percent
                    print(f'Epoch: {epoch:3d}, Iter: {total_iters:6d}, '
                          f'G_A: {losses.get("G_A", 0):.4f}, '
                          f'D_A: {losses.get("D_A", 0):.4f}, '
                          f'cycle_A: {losses.get("cycle_A", 0):.4f}, '
                          f'RAM: {memory_used:.1f}GB ({memory_percent:.1f}%)')
                
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
            if epoch_start_time and torch.cuda.is_available():
                epoch_time = epoch_start_time.elapsed_time(torch.cuda.Event(enable_timing=True)) / 1000
                print(f'End of epoch {epoch:3d} / {opt.n_epochs + opt.n_epochs_decay} '
                      f'Time: {epoch_time:.1f}s')
            else:
                print(f'End of epoch {epoch:3d} / {opt.n_epochs + opt.n_epochs_decay}')
    
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user at epoch {epoch}")
        print("Saving current model...")
        model.save_networks('latest')
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model saved in: ./checkpoints/{opt.name}/")
    print(f"Visualization: ./checkpoints/{opt.name}/web/index.html")

if __name__ == "__main__":
    train_cyclegan()
