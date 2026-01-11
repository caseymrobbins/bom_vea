# models/vae.py
# ConvVAE with PixelShuffle decoder

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
        nn.init.constant_(self.block[-1].weight, 0)
        nn.init.constant_(self.block[-1].bias, 0)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.act(x + self.block(x))

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.ps(self.conv(x))))

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.enc = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True), ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True), ResidualBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True), ResidualBlock(256),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True), ResidualBlock(512),
            nn.Flatten()
        )
        
        self.mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(512 * 4 * 4, latent_dim)
        # ULTRA conservative init: zero mu + logvar.bias=-5.0 → std≈0.08, KL≈2.0/dim at start
        # Keep encoder outputs near zero to avoid massive KL spikes from untrained features.
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight); nn.init.constant_(self.logvar.bias, -5.0)  # Zero weights = pure bias
        
        self.dec_lin = nn.Linear(latent_dim, 512 * 4 * 4)
        self.dec = nn.Sequential(
            ResidualBlock(512),
            PixelShuffleUpsample(512, 256), ResidualBlock(256),
            PixelShuffleUpsample(256, 128), ResidualBlock(128),
            PixelShuffleUpsample(128, 64), ResidualBlock(64),
            PixelShuffleUpsample(64, 32),
            nn.Conv2d(32, image_channels, 3, 1, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        mu = torch.clamp(mu, -50, 50)  # Prevent numerical explosion (BOM goals still enforce actual bounds)
        logvar = self.logvar(h)
        # CRITICAL: Tighter clamp to prevent gradient explosion
        # exp(5)=148 std is still huge, but exp(10)=22k → z can be ±500+ → decoder NaN gradients
        # BOX constraints handle actual bounds, this is just numerical safety
        logvar = torch.clamp(logvar, -10, 5)  # exp(5/2)≈12 std max, prevents z > ±100
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        logvar_safe = torch.clamp(logvar, min=-30.0, max=20.0)
        return mu + torch.exp(0.5 * logvar_safe) * torch.randn_like(logvar)
    
    def decode(self, z):
        return self.dec(self.dec_lin(z).view(-1, 512, 4, 4))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def create_model(latent_dim=128, image_channels=3, device='cuda'):
    model = ConvVAE(latent_dim, image_channels).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    return model
