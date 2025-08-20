"""
Data loader utilities for MSEmbGAN training.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from typing import Optional, Tuple


class ImageDataset(Dataset):
    """Custom dataset for image loading and preprocessing."""
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None, 
                 image_size: int = 256):
        self.root_dir = root_dir
        self.image_size = image_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(
                [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                 if f.lower().endswith(ext.split('*')[1])]
            )
        
        if not self.image_files:
            raise ValueError(f"No image files found in {root_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def create_dataloader(data_path: str, batch_size: int = 16, image_size: int = 256, 
                     num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    """
    Create a data loader for training.
    
    Args:
        data_path: Path to the dataset directory
        batch_size: Batch size for training
        image_size: Size to resize images to
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader instance
    """
    # Data augmentation for training
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset
    if os.path.isdir(data_path):
        dataset = ImageDataset(data_path, transform=transform, image_size=image_size)
    else:
        raise ValueError(f"Data path {data_path} does not exist")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def create_celeba_dataloader(data_path: str, batch_size: int = 16, image_size: int = 256,
                           num_workers: int = 4) -> DataLoader:
    """
    Create a data loader for CelebA dataset.
    
    Args:
        data_path: Path to CelebA dataset
        batch_size: Batch size for training
        image_size: Size to resize images to
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def create_cifar_dataloader(batch_size: int = 16, num_workers: int = 4, 
                          train: bool = True) -> DataLoader:
    """
    Create a data loader for CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        train: Whether to use training or test set
        
    Returns:
        DataLoader instance
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5) if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )
    
    return dataloader


def create_progressive_dataloader(data_path: str, current_size: int, target_size: int,
                                batch_size: int = 16, num_workers: int = 4) -> DataLoader:
    """
    Create a data loader for progressive training.
    
    Args:
        data_path: Path to the dataset
        current_size: Current image size for progressive training
        target_size: Target image size
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    transform = transforms.Compose([
        transforms.Resize((current_size, current_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(data_path, transform=transform, image_size=current_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_dataset_info(dataloader: DataLoader) -> Tuple[int, int, int]:
    """
    Get information about the dataset.
    
    Args:
        dataloader: DataLoader instance
        
    Returns:
        Tuple of (num_samples, num_channels, image_size)
    """
    sample_batch = next(iter(dataloader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    
    batch_size, channels, height, width = sample_batch.shape
    num_samples = len(dataloader.dataset)
    
    return num_samples, channels, height 