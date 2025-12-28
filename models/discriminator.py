# models/discriminator.py
# PatchGAN discriminator with spectral normalization for BOM v14

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization.

    Returns patch-wise predictions (real/fake) rather than single scalar.
    Spectral norm prevents discriminator from dominating training.
    """
    def __init__(self, image_channels=3):
        super().__init__()

        # All convs use spectral norm for stability
        self.model = nn.Sequential(
            # 64x64 -> 32x32
            spectral_norm(nn.Conv2d(image_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 4x4 (patch predictions)
            spectral_norm(nn.Conv2d(512, 1, 3, 1, 1)),
        )

    def forward(self, x):
        """Returns (B, 1, 4, 4) patch predictions."""
        return self.model(x)

def create_discriminator(image_channels=3, device='cuda'):
    """Create and initialize discriminator."""
    disc = PatchGANDiscriminator(image_channels).to(device)
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")
    return disc
