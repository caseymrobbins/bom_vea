# models/vgg.py
# VGG features for perceptual and texture loss

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGGFeatures(nn.Module):
    """Extract multi-scale VGG features for perceptual/texture loss."""
    
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Split into feature blocks
        self.slice1 = nn.Sequential(*list(vgg.features)[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16]) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.features)[16:23])# relu4_3
        
        # Freeze
        for p in self.parameters():
            p.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.to(device)
        self.eval()

    def forward(self, x, return_all=False):
        x = torch.clamp(x, 0, 1)
        x = (x - self.mean) / self.std
        
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        if return_all:
            return [h1, h2, h3, h4]
        return h3  # Default: relu3_3 for perceptual loss


def gram_matrix(features):
    """Compute Gram matrix for texture representation."""
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (C * H * W)


def texture_distance(feat1, feat2):
    """Compute texture distance using Gram matrices across all layers."""
    dist = 0.0
    for f1, f2 in zip(feat1, feat2):
        g1 = gram_matrix(f1)
        g2 = gram_matrix(f2)
        dist += (g1 - g2).pow(2).sum(dim=[1, 2])
    return dist / len(feat1)
