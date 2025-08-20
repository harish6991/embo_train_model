"""
Loss functions for MSEmbGAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""
    
    def __init__(self, gan_mode: str = 'vanilla', target_real_label: float = 1.0, 
                 target_fake_label: float = 0.0):
        super().__init__()
        self.gan_mode = gan_mode
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgangp':
            self.loss = None  # Wasserstein loss is computed differently
        else:
            raise ValueError(f'Unsupported GAN mode: {gan_mode}')
    
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Compute adversarial loss.
        
        Args:
            prediction: Discriminator predictions
            target_is_real: Whether the target is real (True) or fake (False)
            
        Returns:
            Adversarial loss
        """
        if self.gan_mode == 'wgangp':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        
        target_label = self.target_real_label if target_is_real else self.target_fake_label
        target = torch.full_like(prediction, target_label)
        
        return self.loss(prediction, target)


class EmbeddingLoss(nn.Module):
    """Embedding consistency loss for MSEmbGAN."""
    
    def __init__(self, embedding_dim: int, temperature: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Projection heads for embedding comparison
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, real_embeddings: torch.Tensor, fake_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding consistency loss.
        
        Args:
            real_embeddings: Embeddings from real images
            fake_embeddings: Embeddings from generated images
            
        Returns:
            Embedding consistency loss
        """
        # Project embeddings
        real_proj = self.projection(real_embeddings)
        fake_proj = self.projection(fake_embeddings)
        
        # Normalize projections
        real_proj = F.normalize(real_proj, dim=1)
        fake_proj = F.normalize(fake_proj, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(real_proj * fake_proj, dim=1) / self.temperature
        
        # Contrastive loss: maximize similarity between real and fake embeddings
        loss = -torch.log(torch.sigmoid(similarity)).mean()
        
        return loss


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for stable GAN training."""
    
    def __init__(self, feature_weights: List[float] = None):
        super().__init__()
        self.feature_weights = feature_weights if feature_weights is not None else [1.0, 1.0, 1.0, 1.0]
        self.l1_loss = nn.L1Loss()
    
    def forward(self, real_features: List[List[torch.Tensor]], 
                fake_features: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            real_features: Features from real images for each discriminator
            fake_features: Features from generated images for each discriminator
            
        Returns:
            Feature matching loss
        """
        total_loss = 0.0
        
        for disc_idx, (real_disc_features, fake_disc_features) in enumerate(zip(real_features, fake_features)):
            disc_weight = self.feature_weights[disc_idx] if disc_idx < len(self.feature_weights) else 1.0
            
            for scale_idx, (real_scale_features, fake_scale_features) in enumerate(zip(real_disc_features, fake_disc_features)):
                # Ensure features have the same shape
                if real_scale_features.shape != fake_scale_features.shape:
                    fake_scale_features = F.interpolate(
                        fake_scale_features, 
                        size=real_scale_features.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Compute L1 loss between features
                scale_loss = self.l1_loss(real_scale_features, fake_scale_features)
                total_loss += disc_weight * scale_loss
        
        return total_loss


class MultiScaleEmbeddingLoss(nn.Module):
    """Multi-scale embedding loss that considers embeddings at different scales."""
    
    def __init__(self, embedding_dims: List[int], temperature: float = 0.1):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.temperature = temperature
        
        # Create projection heads for each scale
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for dim in embedding_dims
        ])
        
    def forward(self, real_embeddings: List[torch.Tensor], 
                fake_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute multi-scale embedding loss.
        
        Args:
            real_embeddings: List of embeddings from real images at different scales
            fake_embeddings: List of embeddings from generated images at different scales
            
        Returns:
            Multi-scale embedding loss
        """
        total_loss = 0.0
        
        for i, (real_emb, fake_emb, projection) in enumerate(zip(real_embeddings, fake_embeddings, self.projections)):
            # Project embeddings
            real_proj = projection(real_emb)
            fake_proj = projection(fake_emb)
            
            # Normalize projections
            real_proj = F.normalize(real_proj, dim=1)
            fake_proj = F.normalize(fake_proj, dim=1)
            
            # Compute cosine similarity
            similarity = torch.sum(real_proj * fake_proj, dim=1) / self.temperature
            
            # Contrastive loss
            scale_loss = -torch.log(torch.sigmoid(similarity)).mean()
            total_loss += scale_loss
        
        return total_loss


class GradientPenaltyLoss(nn.Module):
    """Gradient penalty loss for Wasserstein GAN training."""
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, discriminator: nn.Module, real_images: torch.Tensor, 
                fake_images: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty loss.
        
        Args:
            discriminator: Discriminator network
            real_images: Real images
            fake_images: Generated images
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(real_images.device)
        
        # Ensure both real and fake images have the same size
        if real_images.shape[2:] != fake_images.shape[2:]:
            # Resize fake images to match real images
            fake_images = F.interpolate(
                fake_images, 
                size=real_images.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Interpolate between real and fake images
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated images
        disc_outputs = discriminator(interpolated)
        
        # For gradient penalty, we use the first discriminator's global score
        # The discriminator returns a list of (global_score, local_scores, features) tuples
        disc_output = disc_outputs[0][0]  # First discriminator, global score
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty 