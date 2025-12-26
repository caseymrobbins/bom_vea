# models/vgg.py
# v12: VGG features + LPIPS for texture

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGGFeatures(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.slice1 = nn.Sequential(*list(vgg.features)[:4])
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9])
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16])
        self.slice4 = nn.Sequential(*list(vgg.features)[16:23])
        for p in self.parameters():
            p.requires_grad = False
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
        return h3

def gram_matrix(features):
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (C * H * W)

def texture_distance(feat1, feat2):
    """Gram matrix texture distance."""
    dist = 0.0
    for f1, f2 in zip(feat1, feat2):
        g1, g2 = gram_matrix(f1), gram_matrix(f2)
        dist += (g1 - g2).pow(2).sum(dim=[1, 2])
    return dist / len(feat1)

class LPIPSTexture(nn.Module):
    """LPIPS-based texture distance - better perceptual alignment."""
    def __init__(self, device='cuda'):
        super().__init__()
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
            self.available = True
        except ImportError:
            print("LPIPS not available, falling back to Gram matrix")
            self.available = False
            self.vgg = VGGFeatures(device)
    
    def forward(self, img1, img2):
        """Returns per-sample texture distance."""
        if self.available:
            # LPIPS returns (B, 1, 1, 1), squeeze to (B,)
            return self.lpips_fn(img1, img2).view(-1)
        else:
            # Fallback to Gram
            f1 = self.vgg(img1, return_all=True)
            f2 = self.vgg(img2, return_all=True)
            return texture_distance(f1, f2)
