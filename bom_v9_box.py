# bom_vae_v9_boxed.py
# Integrating proper constraint types: BOX, BOX_ASYMMETRIC, MINIMIZE_SOFT
# This should prevent degenerate solutions like exploding detail dim variance

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import zipfile
import copy
from torchvision.models import vgg16, VGG16_Weights
from enum import Enum
from typing import Callable, Optional, Dict

# ==================== CONFIG ====================
EPOCHS = 30
LATENT_DIM = 128
BATCH_SIZE = 128
NUM_TRAVERSE_DIMS = 15
OUTPUT_DIR = '/content/outputs_bom_v9'
EVAL_SAMPLES = 10000
CALIBRATION_BATCHES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== CONSTRAINT TYPES ====================
class ConstraintType(Enum):
    UPPER = "upper"
    LOWER = "lower"
    BOX = "box"
    BOX_ASYMMETRIC = "box_asymmetric"
    MINIMIZE_SOFT = "min_soft"
    MINIMIZE_HARD = "min_hard"

def make_normalizer(ctype: ConstraintType, **kwargs) -> Callable[[float], float]:
    """Create a normalizer function that maps raw values to [0, 1] goal scores."""
    if ctype == ConstraintType.UPPER:
        margin = kwargs["margin"]
        return lambda x: max(0.0, min(1.0, (margin - x) / margin)) if margin > 0 else 0.0
    
    if ctype == ConstraintType.LOWER:
        margin = kwargs["margin"]
        return lambda x: max(0.0, min(1.0, (x - margin) / margin)) if margin > 0 else 0.0
    
    if ctype == ConstraintType.BOX:
        lower, upper = kwargs["lower"], kwargs["upper"]
        mid = (lower + upper) / 2
        half_width = (upper - lower) / 2
        # Returns 1.0 at midpoint, 0.0 at edges
        return lambda x: max(0.0, 1.0 - abs(x - mid) / half_width) if half_width > 0 else 0.0
    
    if ctype == ConstraintType.BOX_ASYMMETRIC:
        lower, upper, healthy = kwargs["lower"], kwargs["upper"], kwargs["healthy"]
        dist_lower = healthy - lower
        dist_upper = upper - healthy
        if dist_lower <= 0 or dist_upper <= 0:
            raise ValueError("Healthy must be strictly between lower and upper")
        # Returns 1.0 at healthy point, 0.0 at edges
        def asymmetric_box(x):
            if x < lower or x > upper:
                return 0.0
            if x <= healthy:
                return (x - lower) / dist_lower
            else:
                return (upper - x) / dist_upper
        return asymmetric_box
    
    if ctype == ConstraintType.MINIMIZE_SOFT:
        scale = kwargs["scale"]
        # Returns 1.0 at x=0, approaches 0 as x increases
        # goal = 1 / (1 + x/scale) - same as before but explicit
        return lambda x: 1.0 / (1.0 + max(0.0, x) / scale) if scale > 0 else 0.0
    
    if ctype == ConstraintType.MINIMIZE_HARD:
        scale = kwargs.get("scale", 1.0)
        # More aggressive: goal = 1 / (1 + (x/scale)^2)
        return lambda x: 1.0 / (1.0 + (max(0.0, x) / scale) ** 2) if scale > 0 else 0.0
    
    raise ValueError(f"Unknown constraint type: {ctype}")

def make_normalizer_torch(ctype: ConstraintType, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
    """PyTorch-compatible version of make_normalizer."""
    if ctype == ConstraintType.UPPER:
        margin = kwargs["margin"]
        return lambda x: torch.clamp((margin - x) / margin, 0.0, 1.0) if margin > 0 else torch.zeros_like(x)
    
    if ctype == ConstraintType.LOWER:
        margin = kwargs["margin"]
        return lambda x: torch.clamp((x - margin) / margin, 0.0, 1.0) if margin > 0 else torch.zeros_like(x)
    
    if ctype == ConstraintType.BOX:
        lower, upper = kwargs["lower"], kwargs["upper"]
        mid = (lower + upper) / 2
        half_width = (upper - lower) / 2
        return lambda x: torch.clamp(1.0 - torch.abs(x - mid) / half_width, 0.0, 1.0) if half_width > 0 else torch.zeros_like(x)
    
    if ctype == ConstraintType.BOX_ASYMMETRIC:
        lower, upper, healthy = kwargs["lower"], kwargs["upper"], kwargs["healthy"]
        dist_lower = healthy - lower
        dist_upper = upper - healthy
        def asymmetric_box(x):
            below = (x - lower) / dist_lower
            above = (upper - x) / dist_upper
            score = torch.where(x <= healthy, below, above)
            return torch.clamp(score, 0.0, 1.0)
        return asymmetric_box
    
    if ctype == ConstraintType.MINIMIZE_SOFT:
        scale = kwargs["scale"]
        return lambda x: 1.0 / (1.0 + torch.clamp(x, min=0.0) / scale) if scale > 0 else torch.zeros_like(x)
    
    if ctype == ConstraintType.MINIMIZE_HARD:
        scale = kwargs.get("scale", 1.0)
        return lambda x: 1.0 / (1.0 + (torch.clamp(x, min=0.0) / scale) ** 2) if scale > 0 else torch.zeros_like(x)
    
    raise ValueError(f"Unknown constraint type: {ctype}")

# ==================== GOAL DEFINITIONS ====================
# Each goal has a constraint type and parameters
# These will be calibrated from data where marked "auto"

GOAL_SPECS = {
    # Reconstruction group - minimize losses
    'pixel': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'edge': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'perceptual': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    
    # Core group - minimize losses + contrastive texture
    'core_mse': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'core_edge': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'cross': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'texture_contrastive': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    
    # Latent group - KL should be in a healthy range, not too low or high
    'kl': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 100, 'upper': 5000, 'healthy': 1500},
    'cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'weak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.1},  # Fraction of weak dims
    
    # Split constraint - detail ratio should be in [0.1, 0.5]
    'detail_ratio': {'type': ConstraintType.BOX, 'lower': 0.10, 'upper': 0.50},
    
    # NEW: Dimension variance health - penalize both collapsed AND exploding dims
    'core_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0},
    'detail_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0},
}

# ==================== DATA ====================
root_path = '/content/celeba'
zip_path = '/content/img_align_celeba.zip'

if not os.path.exists(root_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('/content/')

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
])

celebA = ImageFolder(root=root_path, transform=transform)
train_loader = DataLoader(celebA, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
print(f"Loaded {len(celebA)} samples, {len(train_loader)} batches")

# ==================== VGG FOR PERCEPTUAL + TEXTURE ====================
class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        self.slice1 = nn.Sequential(*list(vgg.features)[:4])
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9])
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16])
        self.slice4 = nn.Sequential(*list(vgg.features)[16:23])
        
        for p in self.parameters():
            p.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

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

vgg_features = VGGFeatures().to(device).eval()

def gram_matrix(features):
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (C * H * W)

def texture_distance(img1_features, img2_features):
    dist = 0.0
    for f1, f2 in zip(img1_features, img2_features):
        g1 = gram_matrix(f1)
        g2 = gram_matrix(f2)
        dist += (g1 - g2).pow(2).sum(dim=[1, 2])
    return dist / len(img1_features)

# ==================== SSIM ====================
def compute_ssim(x, y, window_size=11):
    C1, C2 = 0.01**2, 0.03**2
    sigma = 1.5
    gauss = torch.exp(-torch.arange(window_size, device=x.device, dtype=torch.float32).sub(window_size//2).pow(2) / (2*sigma**2))
    gauss = gauss / gauss.sum()
    window = (gauss.unsqueeze(0) * gauss.unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(3,1,window_size,window_size)
    
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=3)
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=3)
    mu_x_sq, mu_y_sq, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    
    sigma_x_sq = F.conv2d(x*x, window, padding=window_size//2, groups=3) - mu_x_sq
    sigma_y_sq = F.conv2d(y*y, window, padding=window_size//2, groups=3) - mu_y_sq
    sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=3) - mu_xy
    
    ssim_map = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2) + 1e-8)
    return torch.clamp(ssim_map.mean(), -1, 1)

# ==================== MODEL ====================
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch))
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(x + self.block(x))

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2,True), ResidualBlock(64),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2,True), ResidualBlock(128),
            nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2,True), ResidualBlock(256),
            nn.Conv2d(256,512,4,2,1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2,True), ResidualBlock(512),
            nn.Flatten())
        self.mu = nn.Linear(512*4*4, latent_dim)
        self.logvar = nn.Linear(512*4*4, latent_dim)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight); nn.init.constant_(self.logvar.bias, -2.0)
        
        self.dec_lin = nn.Linear(latent_dim, 512*4*4)
        self.dec = nn.Sequential(
            ResidualBlock(512),
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2,True), ResidualBlock(256),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2,True), ResidualBlock(128),
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2,True), ResidualBlock(64),
            nn.ConvTranspose2d(64,3,4,2,1), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), torch.clamp(self.logvar(h), -10, 10)
    
    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5*logvar) * torch.randn_like(logvar)
    
    def decode(self, z):
        z = torch.clamp(z, -50, 50)
        return self.dec(self.dec_lin(z).view(-1,512,4,4))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

model = ConvVAE(LATENT_DIM).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ==================== GOAL SYSTEM ====================
class GoalSystem:
    """Manages goals with proper constraint types and calibration."""
    
    def __init__(self, goal_specs: Dict):
        self.specs = goal_specs
        self.scales = {}  # For MINIMIZE_SOFT goals with 'auto' scale
        self.normalizers = {}  # Torch-compatible normalizer functions
        self.samples = {name: [] for name in goal_specs.keys()}
        self.calibrated = False
        self.calibration_count = 0
    
    def collect(self, loss_dict: Dict[str, float]):
        """Collect samples for calibration."""
        for name, value in loss_dict.items():
            if name in self.samples:
                if not np.isnan(value) and not np.isinf(value):
                    self.samples[name].append(value)
    
    def calibrate(self, epoch: int = 0):
        """Calibrate 'auto' scales from collected samples."""
        self.calibration_count += 1
        print("\n" + "="*70)
        print(f"CALIBRATING GOALS (#{self.calibration_count}, epoch {epoch})")
        print("="*70)
        
        for name, spec in self.specs.items():
            ctype = spec['type']
            
            if ctype == ConstraintType.MINIMIZE_SOFT and spec.get('scale') == 'auto':
                if self.samples.get(name):
                    median = np.median(self.samples[name])
                    self.scales[name] = max(median, 1e-6)
                    self.normalizers[name] = make_normalizer_torch(ctype, scale=self.scales[name])
                    print(f"  {name:20s}: MINIMIZE_SOFT scale={self.scales[name]:.4f} (from median)")
                else:
                    self.scales[name] = 1.0
                    self.normalizers[name] = make_normalizer_torch(ctype, scale=1.0)
                    print(f"  {name:20s}: MINIMIZE_SOFT scale=1.0 (no samples)")
            
            elif ctype == ConstraintType.MINIMIZE_SOFT:
                scale = spec['scale']
                self.scales[name] = scale
                self.normalizers[name] = make_normalizer_torch(ctype, scale=scale)
                print(f"  {name:20s}: MINIMIZE_SOFT scale={scale:.4f} (fixed)")
            
            elif ctype == ConstraintType.BOX:
                lower, upper = spec['lower'], spec['upper']
                self.normalizers[name] = make_normalizer_torch(ctype, lower=lower, upper=upper)
                print(f"  {name:20s}: BOX [{lower:.2f}, {upper:.2f}]")
            
            elif ctype == ConstraintType.BOX_ASYMMETRIC:
                lower, upper, healthy = spec['lower'], spec['upper'], spec['healthy']
                self.normalizers[name] = make_normalizer_torch(ctype, lower=lower, upper=upper, healthy=healthy)
                print(f"  {name:20s}: BOX_ASYM [{lower:.0f}, {upper:.0f}] healthy={healthy:.0f}")
            
            else:
                print(f"  {name:20s}: {ctype.value} (using defaults)")
                self.normalizers[name] = make_normalizer_torch(ctype, **{k:v for k,v in spec.items() if k != 'type'})
        
        print("="*70 + "\n")
        self.calibrated = True
        self.samples = {name: [] for name in self.specs.keys()}
    
    def goal(self, value: torch.Tensor, name: str) -> torch.Tensor:
        """Compute goal score for a value."""
        if name not in self.normalizers:
            # Fallback to simple minimize
            return 1.0 / (1.0 + torch.clamp(value, min=0.0))
        
        score = self.normalizers[name](value)
        return torch.clamp(score, 0.001, 1.0)
    
    def start_recalibration(self):
        """Reset samples for recalibration."""
        self.samples = {name: [] for name in self.specs.keys()}

goal_system = GoalSystem(GOAL_SPECS)

# ==================== BOM LOSS ====================
split_idx = LATENT_DIM // 2
GROUP_NAMES = ['recon', 'core', 'latent', 'health']  # Renamed 'split' to 'health'

sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3).to(device)
sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3).to(device)

def edges(img):
    g = img.mean(1, keepdim=True)
    return (F.conv2d(g, sobel_x, padding=1).pow(2) + F.conv2d(g, sobel_y, padding=1).pow(2)).sqrt()

def geometric_mean(goals):
    goals = torch.stack(goals)
    goals = torch.clamp(goals, min=0.001)
    return goals.prod() ** (1.0 / len(goals))

def check_tensor(t):
    return not (torch.isnan(t).any() or torch.isinf(t).any())

def compute_contrastive_texture_loss(r_sw, x1, x2):
    r_sw_feat = vgg_features(r_sw, return_all=True)
    x1_feat = vgg_features(x1, return_all=True)
    x2_feat = vgg_features(x2, return_all=True)
    
    dist_to_x2 = texture_distance(r_sw_feat, x2_feat)
    dist_to_x1 = texture_distance(r_sw_feat, x1_feat)
    
    margin = 0.1
    contrastive_loss = F.relu(dist_to_x2 - dist_to_x1 + margin)
    
    return contrastive_loss.mean(), dist_to_x2.mean(), dist_to_x1.mean()

def compute_raw_losses(recon, x, mu, logvar, z, model):
    """Compute all raw losses for calibration."""
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core = logvar[:, :split_idx]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)
    recon = torch.clamp(recon, 0, 1)
    recon_core = torch.clamp(recon_core, 0, 1)

    with torch.no_grad():
        x_feat = vgg_features(x)
    recon_feat = vgg_features(recon)
    edges_x = edges(x)

    losses = {}
    
    # Pixel
    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val):
        ssim_val = torch.tensor(0.0, device=x.device)
    losses['pixel'] = (pixel_mse + 0.1 * (1.0 - ssim_val)).item()
    
    # Edge
    edges_recon = edges(recon)
    losses['edge'] = F.mse_loss(edges_recon, edges_x).item()
    
    # Perceptual
    losses['perceptual'] = F.mse_loss(recon_feat, x_feat).item()
    
    # Core MSE
    losses['core_mse'] = F.mse_loss(recon_core, x).item()
    
    # Core edge
    edges_core = edges(recon_core)
    losses['core_edge'] = F.mse_loss(edges_core, edges_x).item()
    
    # Cross-reconstruction and contrastive texture
    if B >= 4:
        h = B // 2
        z1_c, z2_d = z_core[:h], z_detail[h:2*h]
        x1, x2 = x[:h], x[h:2*h]
        
        z_sw = torch.cat([z1_c, z2_d], dim=1)
        r_sw = model.decode(z_sw)
        r_sw = torch.clamp(r_sw, 0, 1)
        
        e_x1, e_sw = edges(x1), edges(r_sw)
        losses['cross'] = (F.mse_loss(r_sw, x1) + F.mse_loss(e_sw, e_x1)).item()
        
        with torch.no_grad():
            tex_loss, _, _ = compute_contrastive_texture_loss(r_sw, x1, x2)
        losses['texture_contrastive'] = tex_loss.item()
    else:
        losses['cross'] = 0.1
        losses['texture_contrastive'] = 0.1
    
    # KL
    kl_per_dim = -0.5 * (1 + logvar_core - mu_core.pow(2) - logvar_core.exp())
    kl_per_dim = torch.clamp(kl_per_dim, 0, 50)
    losses['kl'] = kl_per_dim.sum(dim=1).mean().item()
    
    # Covariance
    z_c = torch.clamp(z_core, -10, 10)
    z_c = z_c - z_c.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    off_diag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    losses['cov'] = torch.clamp(off_diag_sq / diag.pow(2).sum(), 0, 50).item()
    
    # Weak dims
    mu_var = mu_core.var(0) + 1e-8
    losses['weak'] = (mu_var < 0.1).float().mean().item()
    
    # Detail ratio
    detail_contrib = (recon - recon_core).abs().mean()
    core_mag = recon_core.abs().mean() + 1e-8
    losses['detail_ratio'] = (detail_contrib / core_mag).item()
    
    # Dimension variance health (median variance of each split)
    losses['core_var_health'] = mu_core.var(0).median().item()
    losses['detail_var_health'] = mu_detail.var(0).median().item()
    
    losses['_ssim'] = ssim_val.item()
    
    return losses

def grouped_bom_loss(recon, x, mu, logvar, z, model, goals):
    """Compute BOM loss with proper constraint types."""
    if not all([check_tensor(t) for t in [recon, x, mu, logvar, z]]):
        return None
    
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core = logvar[:, :split_idx]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)
    
    if not check_tensor(recon_core):
        return None

    recon = torch.clamp(recon, 0, 1)
    recon_core = torch.clamp(recon_core, 0, 1)

    with torch.no_grad():
        x_feat = vgg_features(x)
    recon_feat = vgg_features(recon)
    
    if not all([check_tensor(t) for t in [x_feat, recon_feat]]):
        return None

    edges_x = edges(x)

    # ==================== GROUP A: RECONSTRUCTION ====================
    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val):
        ssim_val = torch.tensor(0.0, device=x.device)
    pixel_loss = pixel_mse + 0.1 * (1.0 - ssim_val)
    g_pixel = goals.goal(pixel_loss, 'pixel')
    
    edges_recon = edges(recon)
    edge_loss = F.mse_loss(edges_recon, edges_x)
    g_edge = goals.goal(edge_loss, 'edge')
    
    perceptual_loss = F.mse_loss(recon_feat, x_feat)
    g_perceptual = goals.goal(perceptual_loss, 'perceptual')

    # ==================== GROUP B: CORE STRUCTURE ====================
    core_mse = F.mse_loss(recon_core, x)
    g_core_mse = goals.goal(core_mse, 'core_mse')
    
    edges_core = edges(recon_core)
    core_edge_loss = F.mse_loss(edges_core, edges_x)
    g_core_edge = goals.goal(core_edge_loss, 'core_edge')
    
    if B >= 4:
        h = B // 2
        z1_c, z2_d = z_core[:h], z_detail[h:2*h]
        x1, x2 = x[:h], x[h:2*h]
        
        z_sw = torch.cat([z1_c, z2_d], dim=1)
        r_sw = model.decode(z_sw)
        r_sw = torch.clamp(r_sw, 0, 1)
        
        e_x1, e_sw = edges(x1), edges(r_sw)
        cross_loss = F.mse_loss(r_sw, x1) + F.mse_loss(e_sw, e_x1)
        g_cross = goals.goal(cross_loss, 'cross')
        
        texture_loss, dist_to_x2, dist_to_x1 = compute_contrastive_texture_loss(r_sw, x1, x2)
        g_texture = goals.goal(texture_loss, 'texture_contrastive')
    else:
        g_cross = torch.tensor(0.5, device=x.device)
        g_texture = torch.tensor(0.5, device=x.device)
        cross_loss = torch.tensor(0.0, device=x.device)
        texture_loss = torch.tensor(0.0, device=x.device)
        dist_to_x2 = torch.tensor(0.0, device=x.device)
        dist_to_x1 = torch.tensor(0.0, device=x.device)

    # ==================== GROUP C: LATENT QUALITY ====================
    kl_per_dim = -0.5 * (1 + logvar_core - mu_core.pow(2) - logvar_core.exp())
    kl_per_dim = torch.clamp(kl_per_dim, 0, 50)
    kl_core = kl_per_dim.sum(dim=1).mean()
    g_kl = goals.goal(kl_core, 'kl')  # BOX_ASYMMETRIC!
    
    z_c = torch.clamp(z_core, -10, 10)
    z_c = z_c - z_c.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    off_diag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    cov_penalty = torch.clamp(off_diag_sq / diag.pow(2).sum(), 0, 50)
    g_cov = goals.goal(cov_penalty, 'cov')
    
    mu_var = mu_core.var(0) + 1e-8
    weak_frac = (mu_var < 0.1).float().mean()
    g_weak = goals.goal(weak_frac, 'weak')

    # ==================== GROUP D: HEALTH (split + variance) ====================
    # Detail ratio - BOX constraint
    detail_contrib = (recon - recon_core).abs().mean()
    core_mag = recon_core.abs().mean() + 1e-8
    detail_ratio = detail_contrib / core_mag
    g_detail_ratio = goals.goal(detail_ratio, 'detail_ratio')  # BOX [0.1, 0.5]!
    
    # Dimension variance health - BOX constraint
    # Penalize collapsed dims (var < 0.5) AND exploding dims (var > 50)
    core_var_median = mu_core.var(0).median()
    detail_var_median = mu_detail.var(0).median()
    g_core_var = goals.goal(core_var_median, 'core_var_health')  # BOX!
    g_detail_var = goals.goal(detail_var_median, 'detail_var_health')  # BOX!

    # ==================== GROUPED BOM ====================
    group_recon = geometric_mean([g_pixel, g_edge, g_perceptual])
    group_core = geometric_mean([g_core_mse, g_core_edge, g_cross, g_texture])
    group_latent = geometric_mean([g_kl, g_cov, g_weak])
    group_health = geometric_mean([g_detail_ratio, g_core_var, g_detail_var])
    
    groups = torch.stack([group_recon, group_core, group_latent, group_health])
    
    if torch.isnan(groups).any() or torch.isinf(groups).any():
        return None
    
    groups = torch.clamp(groups, min=0.001)
    
    min_group = groups.min()
    min_group_idx = groups.argmin()
    
    loss = -torch.log(min_group)
    
    if torch.isnan(loss) or torch.isinf(loss) or loss > 20:
        return None

    individual_goals = {
        'pixel': g_pixel.item(),
        'edge': g_edge.item(),
        'perceptual': g_perceptual.item(),
        'core_mse': g_core_mse.item(),
        'core_edge': g_core_edge.item(),
        'cross': g_cross.item() if isinstance(g_cross, torch.Tensor) else g_cross,
        'texture': g_texture.item() if isinstance(g_texture, torch.Tensor) else g_texture,
        'kl': g_kl.item(),
        'cov': g_cov.item(),
        'weak': g_weak.item(),
        'detail_ratio': g_detail_ratio.item(),
        'core_var': g_core_var.item(),
        'detail_var': g_detail_var.item(),
    }
    
    group_values = {
        'recon': group_recon.item(),
        'core': group_core.item(),
        'latent': group_latent.item(),
        'health': group_health.item(),
    }

    raw_values = {
        'kl_raw': kl_core.item(),
        'detail_ratio_raw': detail_ratio.item(),
        'core_var_raw': core_var_median.item(),
        'detail_var_raw': detail_var_median.item(),
        'texture_loss': texture_loss.item() if isinstance(texture_loss, torch.Tensor) else texture_loss,
        'dist_x2': dist_to_x2.item() if isinstance(dist_to_x2, torch.Tensor) else dist_to_x2,
        'dist_x1': dist_to_x1.item() if isinstance(dist_to_x1, torch.Tensor) else dist_to_x1,
    }

    return (loss, groups, min_group_idx, group_values, individual_goals, raw_values,
            ssim_val.item(), pixel_mse.item(), edge_loss.item())

# ==================== TRAINING ====================
RECALIBRATION_EPOCHS = [10, 20]

histories = {
    'loss': [], 'min_group': [], 'bottleneck': [],
    'ssim': [], 'mse': [], 'edge': [],
    'kl_raw': [], 'detail_ratio_raw': [], 'core_var_raw': [], 'detail_var_raw': [],
    'texture_loss': [], 'texture_dist_x2': [], 'texture_dist_x1': [],
    **{f'group_{n}': [] for n in GROUP_NAMES},
    'pixel': [], 'edge_goal': [], 'perceptual': [],
    'core_mse': [], 'core_edge': [], 'cross': [], 'texture': [],
    'kl_goal': [], 'cov_goal': [], 'weak': [],
    'detail_ratio_goal': [], 'core_var_goal': [], 'detail_var_goal': [],
    'core_active': [], 'detail_active': [],
    'core_effective': [], 'detail_effective': [],
}

dim_variance_history = {'core': [], 'detail': []}

print("\n" + "="*120)
print(f"BOM VAE v9 - BOXED CONSTRAINTS - {EPOCHS} EPOCHS")
print("NEW: Using proper constraint types:")
print("  - MINIMIZE_SOFT for losses (lower is better)")
print("  - BOX for detail_ratio [0.1, 0.5] and dim variance [0.5, 50]")
print("  - BOX_ASYMMETRIC for KL [100, 5000] healthy=1500")
print("Groups: recon(pixel,edge,perceptual) | core(core_mse,core_edge,cross,texture)")
print("        latent(kl,cov,weak) | health(detail_ratio,core_var,detail_var)")
print("="*120 + "\n")

last_good_state = copy.deepcopy(model.state_dict())
last_good_optimizer = copy.deepcopy(optimizer.state_dict())
nan_recovery_count = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    epoch_data = {k: [] for k in histories.keys()}
    bn_counts = {n: 0 for n in GROUP_NAMES}
    skip_count = 0
    
    all_mu_core = []
    all_mu_detail = []
    
    needs_recalibration = epoch in RECALIBRATION_EPOCHS or epoch == 1
    if needs_recalibration:
        goal_system.start_recalibration()
        print(f"\n📊 Epoch {epoch}: Collecting samples for calibration...")

    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch_idx, (x, _) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        
        if not check_tensor(x):
            skip_count += 1
            continue
        
        optimizer.zero_grad(set_to_none=True)
        
        recon, mu, logvar, z = model(x)
        
        # === CALIBRATION PHASE ===
        if needs_recalibration and batch_idx < CALIBRATION_BATCHES:
            with torch.no_grad():
                raw_losses = compute_raw_losses(recon, x, mu, logvar, z, model)
                goal_system.collect(raw_losses)
            
            if not goal_system.calibrated:
                calib_loss = F.mse_loss(recon, x)
                calib_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pbar.set_postfix({'phase': 'CALIBRATING', 'batch': f"{batch_idx}/{CALIBRATION_BATCHES}"})
                continue
        
        if needs_recalibration and batch_idx == CALIBRATION_BATCHES:
            goal_system.calibrate(epoch=epoch)
            needs_recalibration = False
        
        # === NORMAL BOM TRAINING ===
        result = grouped_bom_loss(recon, x, mu, logvar, z, model, goal_system)
        
        if result is None:
            skip_count += 1
            if skip_count > 10:
                print(f"\n⚠️  Too many NaN batches ({skip_count}), recovering...")
                model.load_state_dict(last_good_state)
                optimizer.load_state_dict(last_good_optimizer)
                nan_recovery_count += 1
                skip_count = 0
                if nan_recovery_count > 5:
                    print("❌ Too many recoveries, stopping")
                    break
            continue
        
        loss, groups, min_idx, group_vals, ind_goals, raw_vals, ssim_v, mse, edge = result

        loss.backward()
        
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if batch_idx % 10 == 0:
            with torch.no_grad():
                all_mu_core.append(mu[:, :split_idx].cpu())
                all_mu_detail.append(mu[:, split_idx:].cpu())
        
        if batch_idx % 200 == 0 and batch_idx > 0:
            last_good_state = copy.deepcopy(model.state_dict())
            last_good_optimizer = copy.deepcopy(optimizer.state_dict())
            skip_count = 0

        with torch.no_grad():
            g_np = groups.cpu().numpy()
            epoch_data['loss'].append(loss.item())
            epoch_data['min_group'].append(g_np.min())
            epoch_data['bottleneck'].append(min_idx.item())
            epoch_data['ssim'].append(ssim_v)
            epoch_data['mse'].append(mse)
            epoch_data['edge'].append(edge)
            
            epoch_data['kl_raw'].append(raw_vals['kl_raw'])
            epoch_data['detail_ratio_raw'].append(raw_vals['detail_ratio_raw'])
            epoch_data['core_var_raw'].append(raw_vals['core_var_raw'])
            epoch_data['detail_var_raw'].append(raw_vals['detail_var_raw'])
            epoch_data['texture_loss'].append(raw_vals['texture_loss'])
            epoch_data['texture_dist_x2'].append(raw_vals['dist_x2'])
            epoch_data['texture_dist_x1'].append(raw_vals['dist_x1'])
            
            for n in GROUP_NAMES:
                epoch_data[f'group_{n}'].append(group_vals[n])
            bn_counts[GROUP_NAMES[min_idx.item()]] += 1
            
            epoch_data['pixel'].append(ind_goals['pixel'])
            epoch_data['edge_goal'].append(ind_goals['edge'])
            epoch_data['perceptual'].append(ind_goals['perceptual'])
            epoch_data['core_mse'].append(ind_goals['core_mse'])
            epoch_data['core_edge'].append(ind_goals['core_edge'])
            epoch_data['cross'].append(ind_goals['cross'])
            epoch_data['texture'].append(ind_goals['texture'])
            epoch_data['kl_goal'].append(ind_goals['kl'])
            epoch_data['cov_goal'].append(ind_goals['cov'])
            epoch_data['weak'].append(ind_goals['weak'])
            epoch_data['detail_ratio_goal'].append(ind_goals['detail_ratio'])
            epoch_data['core_var_goal'].append(ind_goals['core_var'])
            epoch_data['detail_var_goal'].append(ind_goals['detail_var'])

        pbar.set_postfix({
            'loss': f"{loss.item():.2f}", 
            'min': f"{g_np.min():.3f}",
            'bn': GROUP_NAMES[min_idx.item()],
            'ssim': f"{ssim_v:.3f}",
            'dv': f"{raw_vals['detail_var_raw']:.1f}",
        })

        if batch_idx % 400 == 0 and batch_idx > CALIBRATION_BATCHES:
            g_str = " | ".join([f"{n}:{group_vals[n]:.2f}" for n in GROUP_NAMES])
            print(f"\n  B{batch_idx}: Groups: {g_str}")
            print(f"           KL={raw_vals['kl_raw']:.0f} det_ratio={raw_vals['detail_ratio_raw']:.3f} "
                  f"core_var={raw_vals['core_var_raw']:.1f} det_var={raw_vals['detail_var_raw']:.1f}")

    if nan_recovery_count > 5:
        break

    scheduler.step()
    
    last_good_state = copy.deepcopy(model.state_dict())
    last_good_optimizer = copy.deepcopy(optimizer.state_dict())
    
    # Dimension analysis
    if all_mu_core:
        all_mu_core_t = torch.cat(all_mu_core, dim=0)
        all_mu_detail_t = torch.cat(all_mu_detail, dim=0)
        
        core_var = all_mu_core_t.var(0)
        detail_var = all_mu_detail_t.var(0)
        
        dim_variance_history['core'].append(core_var.numpy())
        dim_variance_history['detail'].append(detail_var.numpy())
        
        core_active = (core_var > 0.1).sum().item()
        detail_active = (detail_var > 0.1).sum().item()
        
        core_var_norm = core_var / (core_var.sum() + 1e-8) + 1e-8
        core_effective = torch.exp(-torch.sum(core_var_norm * torch.log(core_var_norm))).item()
        
        detail_var_norm = detail_var / (detail_var.sum() + 1e-8) + 1e-8
        detail_effective = torch.exp(-torch.sum(detail_var_norm * torch.log(detail_var_norm))).item()
        
        epoch_data['core_active'] = [core_active]
        epoch_data['detail_active'] = [detail_active]
        epoch_data['core_effective'] = [core_effective]
        epoch_data['detail_effective'] = [detail_effective]

    for k in histories.keys():
        if k == 'bottleneck':
            histories[k].append(max(bn_counts, key=bn_counts.get) if bn_counts else 'none')
        elif epoch_data[k]:
            histories[k].append(np.mean(epoch_data[k]))
        else:
            histories[k].append(histories[k][-1] if histories[k] else 0)

    dt = time.time() - t0
    
    kl = histories['kl_raw'][-1] if histories['kl_raw'] else 0
    dr = histories['detail_ratio_raw'][-1] if histories['detail_ratio_raw'] else 0
    cv = histories['core_var_raw'][-1] if histories['core_var_raw'] else 0
    dv = histories['detail_var_raw'][-1] if histories['detail_var_raw'] else 0
    core_act = histories['core_active'][-1] if histories['core_active'] else 0
    det_act = histories['detail_active'][-1] if histories['detail_active'] else 0
    
    print(f"\nEpoch {epoch:2d} | Loss: {histories['loss'][-1]:.3f} | Min: {histories['min_group'][-1]:.3f} | "
          f"SSIM: {histories['ssim'][-1]:.3f}")
    print(f"         KL={kl:.0f} det_ratio={dr:.3f} core_var={cv:.1f} det_var={dv:.1f}")
    print(f"         Dims: core={core_act:.0f}/64 detail={det_act:.0f}/64")

    tot = sum(bn_counts.values())
    if tot > 0:
        bn_str = " | ".join([f"{n}:{c/tot*100:.0f}%" for n, c in sorted(bn_counts.items(), key=lambda x:-x[1]) if c > 0])
        print(f"         Bottlenecks: {bn_str}")

    g_str = " | ".join([f"{n}:{histories[f'group_{n}'][-1]:.2f}" for n in GROUP_NAMES])
    print(f"         Groups: {g_str}\n")

# ==================== SAVE ====================
torch.save({
    'model_state_dict': model.state_dict(), 
    'histories': histories,
    'goal_specs': GOAL_SPECS,
    'goal_scales': goal_system.scales,
    'dim_variance_history': dim_variance_history,
}, f'{OUTPUT_DIR}/bom_vae_v9.pt')
print(f"✓ Model saved")

# ==================== EVAL ====================
print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except:
    os.system("pip install torchmetrics[image] -q")
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image import StructuralSimilarityIndexMeasure

fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

model.eval()
ssim_s, lpips_s, mse_t, cnt = [], [], 0, 0

with torch.no_grad():
    for x, _ in tqdm(train_loader, desc="Eval"):
        if cnt >= EVAL_SAMPLES: break
        x = x.to(device)
        r, _, _, _ = model(x)
        r = torch.clamp(r, 0, 1)
        fid.update(x, real=True); fid.update(r, real=False)
        ssim_s.append(ssim_m(r, x).item())
        lpips_s.append(lpips_metric(x, r).item())
        mse_t += F.mse_loss(r, x, reduction='sum').item()
        cnt += x.shape[0]

print(f"\n  MSE:   {mse_t/(cnt*3*64*64):.6f}")
print(f"  SSIM:  {np.mean(ssim_s):.4f}")
print(f"  LPIPS: {np.mean(lpips_s):.4f}")
print(f"  FID:   {fid.compute().item():.2f}")

# ==================== VISUALIZATIONS ====================
print("\nGenerating visualizations...")

ep = range(1, len(histories['loss']) + 1)

# Group balance
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ep, histories['min_group'], 'k-', lw=3, label='Min Group')
colors = ['blue', 'red', 'green', 'purple']
for i, n in enumerate(GROUP_NAMES):
    ax.plot(ep, histories[f'group_{n}'], color=colors[i], lw=2, label=n)
ax.set_title('BOM VAE v9: Group Balance (Boxed Constraints)', fontsize=14)
ax.set_xlabel('Epoch'); ax.set_ylabel('Group Score (0-1)')
ax.legend()
ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/group_balance.png', dpi=150, bbox_inches='tight')
plt.close()

# Goal details
fig, axs = plt.subplots(2, 3, figsize=(16, 10))

axs[0,0].plot(ep, histories['pixel'], label='pixel')
axs[0,0].plot(ep, histories['edge_goal'], label='edge')
axs[0,0].plot(ep, histories['perceptual'], label='perceptual')
axs[0,0].plot(ep, histories['group_recon'], 'k--', lw=2, label='group')
axs[0,0].set_title('Reconstruction Group'); axs[0,0].legend(); axs[0,0].grid(True, alpha=0.3)
axs[0,0].set_ylim(0, 1)

axs[0,1].plot(ep, histories['core_mse'], label='core_mse')
axs[0,1].plot(ep, histories['core_edge'], label='core_edge')
axs[0,1].plot(ep, histories['cross'], label='cross')
axs[0,1].plot(ep, histories['texture'], label='texture', lw=2)
axs[0,1].plot(ep, histories['group_core'], 'k--', lw=2, label='group')
axs[0,1].set_title('Core Group'); axs[0,1].legend(); axs[0,1].grid(True, alpha=0.3)
axs[0,1].set_ylim(0, 1)

axs[0,2].plot(ep, histories['kl_goal'], label='kl (BOX_ASYM)')
axs[0,2].plot(ep, histories['cov_goal'], label='cov')
axs[0,2].plot(ep, histories['weak'], label='weak')
axs[0,2].plot(ep, histories['group_latent'], 'k--', lw=2, label='group')
axs[0,2].set_title('Latent Group'); axs[0,2].legend(); axs[0,2].grid(True, alpha=0.3)
axs[0,2].set_ylim(0, 1)

axs[1,0].plot(ep, histories['detail_ratio_goal'], label='detail_ratio (BOX)')
axs[1,0].plot(ep, histories['core_var_goal'], label='core_var (BOX)')
axs[1,0].plot(ep, histories['detail_var_goal'], label='detail_var (BOX)')
axs[1,0].plot(ep, histories['group_health'], 'k--', lw=2, label='group')
axs[1,0].set_title('Health Group (BOX constraints)'); axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)
axs[1,0].set_ylim(0, 1)

axs[1,1].plot(ep, histories['texture_dist_x2'], 'b-', lw=2, label='dist to x2')
axs[1,1].plot(ep, histories['texture_dist_x1'], 'r-', lw=2, label='dist to x1')
axs[1,1].set_title('Texture Distances (want blue < red)'); axs[1,1].legend(); axs[1,1].grid(True, alpha=0.3)

# Raw values for BOX constraints
ax2 = axs[1,2]
ax2.plot(ep, histories['detail_ratio_raw'], 'g-', lw=2, label='detail_ratio')
ax2.axhline(0.1, color='g', ls='--', alpha=0.5)
ax2.axhline(0.5, color='g', ls='--', alpha=0.5)
ax2.set_ylabel('Detail Ratio', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_ylim(0, 0.6)
ax2.legend(loc='upper left')
ax2.set_title('Raw BOX Values')
ax2.grid(True, alpha=0.3)

ax3 = ax2.twinx()
ax3.plot(ep, histories['kl_raw'], 'b-', lw=2, label='KL')
ax3.axhline(100, color='b', ls='--', alpha=0.3)
ax3.axhline(1500, color='b', ls='-', alpha=0.3)
ax3.axhline(5000, color='b', ls='--', alpha=0.3)
ax3.set_ylabel('KL', color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/goal_details.png', dpi=150, bbox_inches='tight')
plt.close()

# Reconstructions
model.eval()
with torch.no_grad():
    samp, _ = next(iter(train_loader))
    samp = samp.to(device)[:16]
    rec_f, _, _, z_b = model(samp)
    rec_f = torch.clamp(rec_f, 0, 1)
    z_co = z_b.clone(); z_co[:, split_idx:] = 0
    rec_c = model.decode(z_co)
    rec_c = torch.clamp(rec_c, 0, 1)
    
    detail_contrib = torch.clamp((rec_f - rec_c).abs() * 5, 0, 1)

    fig, axs = plt.subplots(4, 16, figsize=(20, 5.5))
    fig.suptitle("Original | Full | Core-Only | Detail×5", fontsize=14, y=1.02)
    for i in range(16):
        axs[0,i].imshow(samp[i].cpu().permute(1,2,0).numpy()); axs[0,i].axis('off')
        axs[1,i].imshow(rec_f[i].cpu().permute(1,2,0).numpy()); axs[1,i].axis('off')
        axs[2,i].imshow(rec_c[i].cpu().permute(1,2,0).numpy()); axs[2,i].axis('off')
        axs[3,i].imshow(detail_contrib[i].cpu().permute(1,2,0).numpy()); axs[3,i].axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/reconstructions.png', dpi=150, bbox_inches='tight')
    plt.close()

# Traversals
with torch.no_grad():
    s = samp[0:1]
    mu_b, _ = model.encode(s)
    scales = np.linspace(-3, 3, 11)
    
    fig, axs = plt.subplots(NUM_TRAVERSE_DIMS, 11, figsize=(22, NUM_TRAVERSE_DIMS*2))
    fig.suptitle("Core Traversals (Dims 0-14)", fontsize=16, y=1.01)
    for d in range(NUM_TRAVERSE_DIMS):
        for j, sc in enumerate(scales):
            z = mu_b.clone(); z[0, d] = mu_b[0, d] + sc
            g = model.decode(z)
            g = torch.clamp(g, 0, 1)[0].cpu().numpy()
            axs[d,j].imshow(np.transpose(g,(1,2,0))); axs[d,j].axis('off')
            if j == 0: axs[d,j].set_ylabel(f'D{d}', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/traversals_core.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(NUM_TRAVERSE_DIMS, 11, figsize=(22, NUM_TRAVERSE_DIMS*2))
    fig.suptitle("Detail Traversals (Dims 64-78)", fontsize=16, y=1.01)
    for do in range(NUM_TRAVERSE_DIMS):
        d = split_idx + do
        for j, sc in enumerate(scales):
            z = mu_b.clone(); z[0, d] = mu_b[0, d] + sc
            g = model.decode(z)
            g = torch.clamp(g, 0, 1)[0].cpu().numpy()
            axs[do,j].imshow(np.transpose(g,(1,2,0))); axs[do,j].axis('off')
            if j == 0: axs[do,j].set_ylabel(f'D{d}', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/traversals_detail.png', dpi=150, bbox_inches='tight')
    plt.close()

# Cross-reconstruction
with torch.no_grad():
    n = 8
    x1, x2 = samp[:n], samp[n:2*n] if samp.shape[0] >= 2*n else samp[:n]
    _, _, _, z1 = model(x1)
    _, _, _, z2 = model(x2)
    z_sw = torch.cat([z1[:,:split_idx], z2[:,split_idx:]], dim=1)
    r_sw = model.decode(z_sw)
    r_sw = torch.clamp(r_sw, 0, 1)

    fig, axs = plt.subplots(3, n, figsize=(n*2.5, 8))
    fig.suptitle("Cross-Recon: x1 Core + x2 Detail", fontsize=14)
    for i in range(n):
        axs[0,i].imshow(x1[i].cpu().permute(1,2,0).numpy()); axs[0,i].axis('off')
        if i == 0: axs[0,i].set_ylabel('x1 (structure)', fontsize=12)
        axs[1,i].imshow(x2[i].cpu().permute(1,2,0).numpy()); axs[1,i].axis('off')
        if i == 0: axs[1,i].set_ylabel('x2 (texture)', fontsize=12)
        axs[2,i].imshow(r_sw[i].cpu().permute(1,2,0).numpy()); axs[2,i].axis('off')
        if i == 0: axs[2,i].set_ylabel('x1_core+x2_detail', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cross_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()

# Dimension activity
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0,0].plot(ep, histories['core_active'], 'b-', lw=2, label='Core')
axs[0,0].plot(ep, histories['detail_active'], 'r-', lw=2, label='Detail')
axs[0,0].axhline(64, color='k', ls='--', alpha=0.3)
axs[0,0].set_title('Active Dimensions'); axs[0,0].legend()
axs[0,0].grid(True, alpha=0.3); axs[0,0].set_ylim(0, 70)

axs[0,1].plot(ep, histories['core_effective'], 'b-', lw=2, label='Core')
axs[0,1].plot(ep, histories['detail_effective'], 'r-', lw=2, label='Detail')
axs[0,1].set_title('Effective Dimensionality'); axs[0,1].legend()
axs[0,1].grid(True, alpha=0.3)

if dim_variance_history['core']:
    final_core_var = dim_variance_history['core'][-1]
    axs[1,0].bar(range(split_idx), final_core_var, color='blue', alpha=0.7)
    axs[1,0].axhline(0.5, color='g', ls='--', lw=2, label='BOX lower')
    axs[1,0].axhline(50, color='r', ls='--', lw=2, label='BOX upper')
    active_core = (final_core_var > 0.1).sum()
    axs[1,0].set_title(f'Core Dims Variance - {active_core}/64 active')
    axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)

if dim_variance_history['detail']:
    final_detail_var = dim_variance_history['detail'][-1]
    axs[1,1].bar(range(split_idx), final_detail_var, color='red', alpha=0.7)
    axs[1,1].axhline(0.5, color='g', ls='--', lw=2, label='BOX lower')
    axs[1,1].axhline(50, color='r', ls='--', lw=2, label='BOX upper')
    active_detail = (final_detail_var > 0.1).sum()
    axs[1,1].set_title(f'Detail Dims Variance - {active_detail}/64 active')
    axs[1,1].legend(); axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/dimension_activity.png', dpi=150, bbox_inches='tight')
plt.close()

# Training history
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs[0,0].plot(histories['loss'], 'b-', lw=2); axs[0,0].set_title('Loss'); axs[0,0].grid(True, alpha=0.3)
axs[0,1].plot(histories['ssim'], 'g-', lw=2); axs[0,1].set_title('SSIM'); axs[0,1].grid(True, alpha=0.3)
axs[0,2].plot(histories['texture'], 'r-', lw=2); axs[0,2].set_title('Texture Goal'); axs[0,2].grid(True, alpha=0.3)
axs[1,0].plot(histories['core_var_raw'], 'b-', lw=2, label='core')
axs[1,0].plot(histories['detail_var_raw'], 'r-', lw=2, label='detail')
axs[1,0].axhline(0.5, c='g', ls='--', alpha=0.5); axs[1,0].axhline(50, c='g', ls='--', alpha=0.5)
axs[1,0].set_title('Dim Variance (BOX [0.5, 50])'); axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)
axs[1,1].plot(histories['kl_raw'], 'c-', lw=2)
axs[1,1].axhline(100, c='r', ls='--', alpha=0.5); axs[1,1].axhline(1500, c='g', ls='-', alpha=0.5)
axs[1,1].axhline(5000, c='r', ls='--', alpha=0.5)
axs[1,1].set_title('KL (BOX_ASYM [100,5000] h=1500)'); axs[1,1].grid(True, alpha=0.3)
axs[1,2].plot(histories['min_group'], 'k-', lw=2); axs[1,2].set_title('Min Group'); axs[1,2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_history.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved to {OUTPUT_DIR}/")
print("="*60 + "\nCOMPLETE\n" + "="*60)
