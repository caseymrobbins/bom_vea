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
        base_dim = latent_dim // 4
        self.latent_sizes = {
            "core": base_dim,
            "mid": base_dim,
            "detail": base_dim,
            "resid": latent_dim - (base_dim * 3),
        }
        self.latent_order = ["core", "mid", "detail", "resid"]
        self.structure_keys = ("core", "mid")
        self.appearance_keys = ("detail", "resid")
        
        self.enc = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True), ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True), ResidualBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True), ResidualBlock(256),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True), ResidualBlock(512),
            nn.Flatten()
        )
        
        self.mu_core = nn.Linear(512 * 4 * 4, self.latent_sizes["core"])
        self.mu_mid = nn.Linear(512 * 4 * 4, self.latent_sizes["mid"])
        self.mu_detail = nn.Linear(512 * 4 * 4, self.latent_sizes["detail"])
        self.mu_resid = nn.Linear(512 * 4 * 4, self.latent_sizes["resid"])

        self.logvar_core = nn.Linear(512 * 4 * 4, self.latent_sizes["core"])
        self.logvar_mid = nn.Linear(512 * 4 * 4, self.latent_sizes["mid"])
        self.logvar_detail = nn.Linear(512 * 4 * 4, self.latent_sizes["detail"])
        self.logvar_resid = nn.Linear(512 * 4 * 4, self.latent_sizes["resid"])
        # ULTRA conservative init: zero mu + logvar.bias=-5.0 → std≈0.08, KL≈2.0/dim at start
        # Keep encoder outputs near zero to avoid massive KL spikes from untrained features.
        for layer in [self.mu_core, self.mu_mid, self.mu_detail, self.mu_resid]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.logvar_core, self.logvar_mid, self.logvar_detail, self.logvar_resid]:
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, -5.0)  # Zero weights = pure bias
        
        self.dec_lin = nn.Linear(latent_dim, 512 * 4 * 4)
        self.dec = nn.Sequential(
            ResidualBlock(512),
            PixelShuffleUpsample(512, 256), ResidualBlock(256),
            PixelShuffleUpsample(256, 128), ResidualBlock(128),
            PixelShuffleUpsample(128, 64), ResidualBlock(64),
            PixelShuffleUpsample(64, 32),
            nn.Conv2d(32, image_channels, 3, 1, 1), nn.Sigmoid()
        )

    def concat_latents(self, latents, keys=None):
        if isinstance(latents, torch.Tensor):
            return latents
        if keys is None:
            keys = self.latent_order
        return torch.cat([latents[key] for key in keys], dim=1)

    @property
    def structure_dim(self):
        return sum(self.latent_sizes[key] for key in self.structure_keys)

    @property
    def appearance_dim(self):
        return sum(self.latent_sizes[key] for key in self.appearance_keys)

    def structure_latents(self, latents):
        return self.concat_latents(latents, self.structure_keys)

    def appearance_latents(self, latents):
        return self.concat_latents(latents, self.appearance_keys)

    def encode(self, x):
        h = self.enc(x)
        mu = {
            "core": torch.clamp(self.mu_core(h), -50, 50),
            "mid": torch.clamp(self.mu_mid(h), -50, 50),
            "detail": torch.clamp(self.mu_detail(h), -50, 50),
            "resid": torch.clamp(self.mu_resid(h), -50, 50),
        }
        # CRITICAL: Tighter clamp to prevent gradient explosion
        # exp(5)=148 std is still huge, but exp(10)=22k → z can be ±500+ → decoder NaN gradients
        # BOX constraints handle actual bounds, this is just numerical safety
        logvar = {
            "core": torch.clamp(self.logvar_core(h), -10, 5),
            "mid": torch.clamp(self.logvar_mid(h), -10, 5),
            "detail": torch.clamp(self.logvar_detail(h), -10, 5),
            "resid": torch.clamp(self.logvar_resid(h), -10, 5),
        }
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        z = {}
        for key in self.latent_order:
            logvar_safe = torch.clamp(logvar[key], min=-30.0, max=20.0)
            z[key] = mu[key] + torch.exp(0.5 * logvar_safe) * torch.randn_like(logvar[key])
        return z, self.concat_latents(z)
    
    def decode(self, z):
        z = self.concat_latents(z)
        return self.dec(self.dec_lin(z).view(-1, 512, 4, 4))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z_parts, z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z_parts, z

def create_model(latent_dim=128, image_channels=3, device='cuda'):
    model = ConvVAE(latent_dim, image_channels).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    return model
