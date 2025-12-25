# bom_vae_v7_texture.py
# Adding texture transfer goal - detail dims must carry texture that transfers between images

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

# ==================== CONFIG ====================
EPOCHS = 30
LATENT_DIM = 128
BATCH_SIZE = 128
NUM_TRAVERSE_DIMS = 15
OUTPUT_DIR = '/content/outputs_bom_v7'
EVAL_SAMPLES = 10000
CALIBRATION_BATCHES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    """Extract multiple layers for both perceptual loss and gram matrices"""
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # We'll extract from multiple layers for texture
        self.slice1 = nn.Sequential(*list(vgg.features)[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16]) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.features)[16:23]) # relu4_3
        
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
        return h3  # Default: relu3_3 for perceptual loss

vgg_features = VGGFeatures().to(device).eval()

def gram_matrix(features):
    """Compute gram matrix for texture representation"""
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (C * H * W)  # Normalize

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

# ==================== ADAPTIVE REFERENCE SYSTEM ====================
class AdaptiveReferences:
    def __init__(self, loss_names):
        self.loss_names = loss_names
        self.samples = {name: [] for name in loss_names}
        self.references = {name: 1.0 for name in loss_names}
        self.calibrated = False
        self.calibration_count = 0
    
    def collect(self, loss_dict):
        """Collect loss samples for calibration."""
        for name, value in loss_dict.items():
            if name in self.samples and not np.isnan(value) and not np.isinf(value) and value > 0:
                self.samples[name].append(value)
    
    def start_recalibration(self):
        """Reset samples to prepare for recalibration."""
        self.samples = {name: [] for name in self.loss_names}
    
    def calibrate(self, epoch=0):
        self.calibration_count += 1
        print("\n" + "="*60)
        print(f"CALIBRATING REFERENCES (#{self.calibration_count}, epoch {epoch})")
        print("="*60)
        for name in self.loss_names:
            if self.samples[name]:
                median = np.median(self.samples[name])
                old_ref = self.references[name]
                self.references[name] = max(median, 1e-6)
                change = (self.references[name] - old_ref) / (old_ref + 1e-8) * 100
                print(f"  {name:15s}: median={median:.4f} -> ref={self.references[name]:.4f} ({change:+.1f}%)")
            else:
                print(f"  {name:15s}: NO SAMPLES - keeping {self.references[name]:.4f}")
        print("="*60 + "\n")
        self.calibrated = True
        self.samples = {name: [] for name in self.loss_names}  # Reset for next calibration
    
    def goal(self, loss, name):
        ref = self.references.get(name, 1.0)
        loss = torch.clamp(loss, 0, 1e6)
        goal = 1.0 / (1.0 + loss / ref)
        return torch.clamp(goal, 0.001, 1.0)

RECALIBRATION_EPOCHS = [10, 20]  # Recalibrate at these epochs
RECALIBRATION_BATCHES = 100  # Batches to collect before recalibrating

# Loss names - added texture_transfer
LOSS_NAMES = [
    'pixel', 'edge', 'perceptual',           # Recon group
    'core_mse', 'core_edge', 'cross', 'texture_transfer',  # Core group (texture_transfer added!)
    'kl', 'cov', 'weak',                     # Latent group
    'split'                                   # Split constraint
]

adaptive_refs = AdaptiveReferences(LOSS_NAMES)

# ==================== BOM LOSS ====================
split_idx = LATENT_DIM // 2
GROUP_NAMES = ['recon', 'core', 'latent', 'split']

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

def compute_texture_loss(recon_features, target_features):
    """
    Compute texture similarity using gram matrices across multiple VGG layers.
    recon_features and target_features are lists of feature maps from VGG.
    """
    texture_loss = 0.0
    for rf, tf in zip(recon_features, target_features):
        gram_recon = gram_matrix(rf)
        gram_target = gram_matrix(tf)
        texture_loss += F.mse_loss(gram_recon, gram_target)
    return texture_loss / len(recon_features)

def compute_raw_losses(recon, x, mu, logvar, z, model):
    """Compute all raw losses for calibration."""
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, logvar_core = mu[:, :split_idx], logvar[:, :split_idx]

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
    
    # Cross-reconstruction and texture transfer
    if B >= 4:
        h = B // 2
        z1_c, z2_d = z_core[:h], z_detail[h:2*h]
        x1, x2 = x[:h], x[h:2*h]
        
        z_sw = torch.cat([z1_c, z2_d], dim=1)
        r_sw = model.decode(z_sw)
        r_sw = torch.clamp(r_sw, 0, 1)
        
        e_x1, e_sw = edges(x1), edges(r_sw)
        losses['cross'] = (F.mse_loss(r_sw, x1) + F.mse_loss(e_sw, e_x1)).item()
        
        # Texture transfer: does r_sw have x2's texture?
        with torch.no_grad():
            x2_features = vgg_features(x2, return_all=True)
            r_sw_features = vgg_features(r_sw, return_all=True)
        losses['texture_transfer'] = compute_texture_loss(r_sw_features, x2_features).item()
    else:
        losses['cross'] = 0.1
        losses['texture_transfer'] = 0.1
    
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
    
    # Split constraint
    detail_contrib = (recon - recon_core).abs().mean()
    core_mag = recon_core.abs().mean() + 1e-8
    detail_ratio = (detail_contrib / core_mag).item()
    below_penalty = max(0, 0.10 - detail_ratio) * 10
    above_penalty = max(0, detail_ratio - 0.50) * 5
    losses['split'] = below_penalty + above_penalty
    
    losses['_ssim'] = ssim_val.item()
    losses['_detail_ratio'] = detail_ratio
    
    return losses

def grouped_bom_loss(recon, x, mu, logvar, z, model, refs):
    """Compute BOM loss with texture transfer goal."""
    if not all([check_tensor(t) for t in [recon, x, mu, logvar, z]]):
        return None
    
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, logvar_core = mu[:, :split_idx], logvar[:, :split_idx]

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
    g_pixel = refs.goal(pixel_loss, 'pixel')
    
    edges_recon = edges(recon)
    edge_loss = F.mse_loss(edges_recon, edges_x)
    g_edge = refs.goal(edge_loss, 'edge')
    
    perceptual_loss = F.mse_loss(recon_feat, x_feat)
    g_perceptual = refs.goal(perceptual_loss, 'perceptual')

    # ==================== GROUP B: CORE STRUCTURE ====================
    core_mse = F.mse_loss(recon_core, x)
    g_core_mse = refs.goal(core_mse, 'core_mse')
    
    edges_core = edges(recon_core)
    core_edge_loss = F.mse_loss(edges_core, edges_x)
    g_core_edge = refs.goal(core_edge_loss, 'core_edge')
    
    # Cross-reconstruction and texture transfer
    if B >= 4:
        h = B // 2
        z1_c, z2_d = z_core[:h], z_detail[h:2*h]
        x1, x2 = x[:h], x[h:2*h]
        
        z_sw = torch.cat([z1_c, z2_d], dim=1)
        r_sw = model.decode(z_sw)
        r_sw = torch.clamp(r_sw, 0, 1)
        
        e_x1, e_sw = edges(x1), edges(r_sw)
        cross_loss = F.mse_loss(r_sw, x1) + F.mse_loss(e_sw, e_x1)
        g_cross = refs.goal(cross_loss, 'cross')
        
        # NEW: Texture transfer - r_sw should have x2's texture
        x2_features = vgg_features(x2, return_all=True)
        r_sw_features = vgg_features(r_sw, return_all=True)
        texture_loss = compute_texture_loss(r_sw_features, x2_features)
        g_texture = refs.goal(texture_loss, 'texture_transfer')
    else:
        g_cross = torch.tensor(0.5, device=x.device)
        g_texture = torch.tensor(0.5, device=x.device)
        cross_loss = torch.tensor(0.0, device=x.device)
        texture_loss = torch.tensor(0.0, device=x.device)

    # ==================== GROUP C: LATENT QUALITY ====================
    kl_per_dim = -0.5 * (1 + logvar_core - mu_core.pow(2) - logvar_core.exp())
    kl_per_dim = torch.clamp(kl_per_dim, 0, 50)
    kl_core = kl_per_dim.sum(dim=1).mean()
    g_kl = refs.goal(kl_core, 'kl')
    
    z_c = torch.clamp(z_core, -10, 10)
    z_c = z_c - z_c.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    off_diag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    cov_penalty = torch.clamp(off_diag_sq / diag.pow(2).sum(), 0, 50)
    g_cov = refs.goal(cov_penalty, 'cov')
    
    mu_var = mu_core.var(0) + 1e-8
    weak_frac = (mu_var < 0.1).float().mean()
    g_weak = refs.goal(weak_frac, 'weak')

    # ==================== GROUP D: SPLIT CONSTRAINT ====================
    detail_contrib = (recon - recon_core).abs().mean()
    core_mag = recon_core.abs().mean() + 1e-8
    detail_ratio = detail_contrib / core_mag
    
    below_penalty = F.relu(0.10 - detail_ratio) * 10
    above_penalty = F.relu(detail_ratio - 0.50) * 5
    split_loss = below_penalty + above_penalty
    g_split = refs.goal(split_loss, 'split')

    # ==================== GROUPED BOM ====================
    group_recon = geometric_mean([g_pixel, g_edge, g_perceptual])
    group_core = geometric_mean([g_core_mse, g_core_edge, g_cross, g_texture])  # Added g_texture!
    group_latent = geometric_mean([g_kl, g_cov, g_weak])
    group_split = g_split
    
    groups = torch.stack([group_recon, group_core, group_latent, group_split])
    
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
        'split': g_split.item(),
    }
    
    group_values = {
        'recon': group_recon.item(),
        'core': group_core.item(),
        'latent': group_latent.item(),
        'split': group_split.item(),
    }

    return (loss, groups, min_group_idx, group_values, individual_goals,
            kl_core.item(), cov_penalty.item(), ssim_val.item(), 
            pixel_mse.item(), edge_loss.item(), detail_ratio.item(),
            texture_loss.item() if isinstance(texture_loss, torch.Tensor) else texture_loss)

# ==================== TRAINING ====================
histories = {
    'loss': [], 'min_group': [], 'bottleneck': [],
    'kl': [], 'cov_penalty': [], 'ssim': [], 'mse': [], 
    'edge': [], 'detail_ratio': [], 'texture_loss': [],
    **{f'group_{n}': [] for n in GROUP_NAMES},
    'pixel': [], 'edge_goal': [], 'perceptual': [],
    'core_mse': [], 'core_edge': [], 'cross': [], 'texture': [],
    'kl_goal': [], 'cov_goal': [], 'weak': [], 'split': [],
    # NEW: Dimension activity tracking
    'core_active': [], 'detail_active': [],
    'core_effective': [], 'detail_effective': [],
}

# Store per-dimension variance at end of each epoch
dim_variance_history = {'core': [], 'detail': []}

print("\n" + "="*120)
print(f"BOM VAE v7 - TEXTURE TRANSFER GOAL - {EPOCHS} EPOCHS")
print(f"NEW: Core group now includes texture_transfer goal (gram matrix style loss)")
print("Groups: recon(pixel,edge,perceptual) | core(core_mse,core_edge,cross,TEXTURE) | latent(kl,cov,weak) | split")
print("="*120 + "\n")

last_good_state = copy.deepcopy(model.state_dict())
last_good_optimizer = copy.deepcopy(optimizer.state_dict())
nan_recovery_count = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    epoch_data = {k: [] for k in histories.keys()}
    bn_counts = {n: 0 for n in GROUP_NAMES}
    skip_count = 0
    
    # Track mu values for dimension analysis
    all_mu_core = []
    all_mu_detail = []
    
    # Check if we need to recalibrate this epoch
    needs_recalibration = epoch in RECALIBRATION_EPOCHS
    if needs_recalibration:
        adaptive_refs.start_recalibration()
        print(f"\n📊 Epoch {epoch}: Collecting samples for recalibration...")

    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch_idx, (x, _) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        
        if not check_tensor(x):
            skip_count += 1
            continue
        
        optimizer.zero_grad(set_to_none=True)
        
        recon, mu, logvar, z = model(x)
        
        # === INITIAL CALIBRATION PHASE (epoch 1) ===
        if epoch == 1 and batch_idx < CALIBRATION_BATCHES and not adaptive_refs.calibrated:
            with torch.no_grad():
                raw_losses = compute_raw_losses(recon, x, mu, logvar, z, model)
                adaptive_refs.collect(raw_losses)
            
            calib_loss = F.mse_loss(recon, x)
            calib_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix({'phase': 'CALIBRATING', 'batch': f"{batch_idx}/{CALIBRATION_BATCHES}"})
            continue
        
        if epoch == 1 and batch_idx == CALIBRATION_BATCHES and not adaptive_refs.calibrated:
            adaptive_refs.calibrate(epoch=1)
        
        # === RECALIBRATION PHASE (epochs 10, 20) ===
        if needs_recalibration and batch_idx < RECALIBRATION_BATCHES:
            with torch.no_grad():
                raw_losses = compute_raw_losses(recon, x, mu, logvar, z, model)
                adaptive_refs.collect(raw_losses)
        
        if needs_recalibration and batch_idx == RECALIBRATION_BATCHES:
            adaptive_refs.calibrate(epoch=epoch)
            needs_recalibration = False  # Done for this epoch
        
        # === NORMAL BOM TRAINING ===
        result = grouped_bom_loss(recon, x, mu, logvar, z, model, adaptive_refs)
        
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
        
        loss, groups, min_idx, group_vals, ind_goals, kl, cov_pen, ssim_v, mse, edge, det_ratio, tex_loss = result

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
        
        # Collect mu for dimension analysis (every 10th batch to save memory)
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
            epoch_data['kl'].append(kl)
            epoch_data['cov_penalty'].append(cov_pen)
            epoch_data['ssim'].append(ssim_v)
            epoch_data['mse'].append(mse)
            epoch_data['edge'].append(edge)
            epoch_data['detail_ratio'].append(det_ratio)
            epoch_data['texture_loss'].append(tex_loss)
            
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
            epoch_data['split'].append(ind_goals['split'])

        pbar.set_postfix({
            'loss': f"{loss.item():.2f}", 
            'min': f"{g_np.min():.3f}",
            'bn': GROUP_NAMES[min_idx.item()],
            'ssim': f"{ssim_v:.3f}",
            'tex': f"{ind_goals['texture']:.2f}",
            'det': f"{det_ratio:.3f}",
        })

        if batch_idx % 400 == 0 and batch_idx > CALIBRATION_BATCHES:
            g_str = " | ".join([f"{n}:{group_vals[n]:.2f}" for n in GROUP_NAMES])
            print(f"\n  B{batch_idx}: Groups: {g_str}")
            i_str = " | ".join([f"{k[:4]}:{v:.2f}" for k, v in ind_goals.items()])
            print(f"           Goals: {i_str}")

    if nan_recovery_count > 5:
        break

    scheduler.step()
    
    last_good_state = copy.deepcopy(model.state_dict())
    last_good_optimizer = copy.deepcopy(optimizer.state_dict())
    
    # === DIMENSION ANALYSIS ===
    if all_mu_core:
        all_mu_core_t = torch.cat(all_mu_core, dim=0)
        all_mu_detail_t = torch.cat(all_mu_detail, dim=0)
        
        core_var = all_mu_core_t.var(0)
        detail_var = all_mu_detail_t.var(0)
        
        # Store for visualization
        dim_variance_history['core'].append(core_var.numpy())
        dim_variance_history['detail'].append(detail_var.numpy())
        
        # Count active dims (variance > 0.1)
        core_active = (core_var > 0.1).sum().item()
        detail_active = (detail_var > 0.1).sum().item()
        
        # Effective dimensionality (exponential of entropy)
        core_var_norm = core_var / (core_var.sum() + 1e-8)
        core_var_norm = core_var_norm + 1e-8  # Prevent log(0)
        core_effective = torch.exp(-torch.sum(core_var_norm * torch.log(core_var_norm))).item()
        
        detail_var_norm = detail_var / (detail_var.sum() + 1e-8)
        detail_var_norm = detail_var_norm + 1e-8
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
    
    tex_goal = histories['texture'][-1] if histories['texture'] else 0
    core_act = histories['core_active'][-1] if histories['core_active'] else 0
    det_act = histories['detail_active'][-1] if histories['detail_active'] else 0
    
    print(f"\nEpoch {epoch:2d} | Loss: {histories['loss'][-1]:.3f} | Min: {histories['min_group'][-1]:.3f} | "
          f"SSIM: {histories['ssim'][-1]:.3f} | Texture: {tex_goal:.3f} | "
          f"Det: {histories['detail_ratio'][-1]:.3f} | Time: {dt:.1f}s")
    print(f"         Dims active: core={core_act:.0f}/64 | detail={det_act:.0f}/64")

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
    'references': adaptive_refs.references,
    'dim_variance_history': dim_variance_history,
}, f'{OUTPUT_DIR}/bom_vae_v7.pt')
print(f"✓ Model saved")

print("\nFinal Calibrated References:")
for name, ref in adaptive_refs.references.items():
    print(f"  {name}: {ref:.6f}")

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

# Group balance
fig, ax = plt.subplots(figsize=(12, 6))
ep = range(1, len(histories['loss']) + 1)
ax.plot(ep, histories['min_group'], 'k-', lw=3, label='Min Group')
colors = ['blue', 'red', 'green', 'orange']
for i, n in enumerate(GROUP_NAMES):
    ax.plot(ep, histories[f'group_{n}'], color=colors[i], lw=2, label=n)
ax.set_title('BOM VAE v7: Group Balance (With Texture Transfer)', fontsize=14)
ax.set_xlabel('Epoch'); ax.set_ylabel('Group Score (0-1)')
ax.legend()
ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/group_balance.png', dpi=150, bbox_inches='tight')
plt.close()

# Individual goals - now with texture
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0,0].plot(ep, histories['pixel'], label='pixel')
axs[0,0].plot(ep, histories['edge_goal'], label='edge')
axs[0,0].plot(ep, histories['perceptual'], label='perceptual')
axs[0,0].plot(ep, histories['group_recon'], 'k--', lw=2, label='group')
axs[0,0].set_title('Reconstruction Group'); axs[0,0].legend(); axs[0,0].grid(True, alpha=0.3)
axs[0,0].set_ylim(0, 1)

axs[0,1].plot(ep, histories['core_mse'], label='core_mse')
axs[0,1].plot(ep, histories['core_edge'], label='core_edge')
axs[0,1].plot(ep, histories['cross'], label='cross')
axs[0,1].plot(ep, histories['texture'], label='TEXTURE', lw=2)  # Highlight texture
axs[0,1].plot(ep, histories['group_core'], 'k--', lw=2, label='group')
axs[0,1].set_title('Core Group (+ Texture Transfer)'); axs[0,1].legend(); axs[0,1].grid(True, alpha=0.3)
axs[0,1].set_ylim(0, 1)

axs[1,0].plot(ep, histories['kl_goal'], label='kl')
axs[1,0].plot(ep, histories['cov_goal'], label='cov')
axs[1,0].plot(ep, histories['weak'], label='weak')
axs[1,0].plot(ep, histories['group_latent'], 'k--', lw=2, label='group')
axs[1,0].set_title('Latent Group'); axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)
axs[1,0].set_ylim(0, 1)

axs[1,1].plot(ep, histories['split'], label='split goal')
axs[1,1].plot(ep, histories['detail_ratio'], label='detail_ratio')
axs[1,1].axhline(0.1, color='r', ls='--', alpha=0.5, label='min target')
axs[1,1].axhline(0.5, color='r', ls='--', alpha=0.5, label='max target')
axs[1,1].set_title('Split Constraint'); axs[1,1].legend(); axs[1,1].grid(True, alpha=0.3)

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

# Core traversals
with torch.no_grad():
    s = samp[0:1]
    mu_b, _ = model.encode(s)
    fig, axs = plt.subplots(NUM_TRAVERSE_DIMS, 11, figsize=(22, NUM_TRAVERSE_DIMS*2))
    fig.suptitle("Core Traversals (Dims 0-14)", fontsize=16, y=1.01)
    scales = np.linspace(-3, 3, 11)
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

# Detail traversals
with torch.no_grad():
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

# Cross-recon - THE KEY TEST
with torch.no_grad():
    n = 8  # More examples
    x1, x2 = samp[:n], samp[n:2*n] if samp.shape[0] >= 2*n else samp[:n]
    _, _, _, z1 = model(x1)
    _, _, _, z2 = model(x2)
    z_sw = torch.cat([z1[:,:split_idx], z2[:,split_idx:]], dim=1)
    r_sw = model.decode(z_sw)
    r_sw = torch.clamp(r_sw, 0, 1)

    fig, axs = plt.subplots(3, n, figsize=(n*2.5, 8))
    fig.suptitle("Cross-Recon: x1 Core + x2 Detail (Does texture transfer?)", fontsize=14)
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

# Training history
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs[0,0].plot(histories['loss'], 'b-', lw=2); axs[0,0].set_title('Loss'); axs[0,0].grid(True, alpha=0.3)
axs[0,1].plot(histories['ssim'], 'g-', lw=2); axs[0,1].set_title('SSIM'); axs[0,1].grid(True, alpha=0.3)
axs[0,2].plot(histories['texture'], 'r-', lw=2); axs[0,2].set_title('Texture Goal'); axs[0,2].grid(True, alpha=0.3)
axs[1,0].plot(histories['detail_ratio'], 'm-', lw=2)
axs[1,0].axhline(0.1, c='k', ls='--', alpha=0.5); axs[1,0].axhline(0.5, c='k', ls='--', alpha=0.5)
axs[1,0].set_title('Detail Ratio'); axs[1,0].grid(True, alpha=0.3)
axs[1,1].plot(histories['kl'], 'c-', lw=2); axs[1,1].set_title('KL'); axs[1,1].grid(True, alpha=0.3)
axs[1,2].plot(histories['min_group'], 'k-', lw=2); axs[1,2].set_title('Min Group'); axs[1,2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_history.png', dpi=150, bbox_inches='tight')
plt.close()

# NEW: Dimension activity visualization
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Active dims over time
axs[0,0].plot(ep, histories['core_active'], 'b-', lw=2, label='Core')
axs[0,0].plot(ep, histories['detail_active'], 'r-', lw=2, label='Detail')
axs[0,0].axhline(64, color='k', ls='--', alpha=0.3, label='Max (64)')
axs[0,0].set_title('Active Dimensions (var > 0.1)'); axs[0,0].legend()
axs[0,0].set_xlabel('Epoch'); axs[0,0].set_ylabel('# Active Dims')
axs[0,0].grid(True, alpha=0.3); axs[0,0].set_ylim(0, 70)

# Effective dimensionality over time
axs[0,1].plot(ep, histories['core_effective'], 'b-', lw=2, label='Core')
axs[0,1].plot(ep, histories['detail_effective'], 'r-', lw=2, label='Detail')
axs[0,1].set_title('Effective Dimensionality (entropy-based)'); axs[0,1].legend()
axs[0,1].set_xlabel('Epoch'); axs[0,1].set_ylabel('Effective Dims')
axs[0,1].grid(True, alpha=0.3)

# Final dimension variance - Core
if dim_variance_history['core']:
    final_core_var = dim_variance_history['core'][-1]
    axs[1,0].bar(range(split_idx), final_core_var, color='blue', alpha=0.7)
    axs[1,0].axhline(0.1, color='r', ls='--', lw=2, label='Active threshold')
    active_core = (final_core_var > 0.1).sum()
    axs[1,0].set_title(f'Core Dims Variance (Final) - {active_core}/64 active')
    axs[1,0].set_xlabel('Dimension'); axs[1,0].set_ylabel('Variance')
    axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)

# Final dimension variance - Detail
if dim_variance_history['detail']:
    final_detail_var = dim_variance_history['detail'][-1]
    axs[1,1].bar(range(split_idx), final_detail_var, color='red', alpha=0.7)
    axs[1,1].axhline(0.1, color='r', ls='--', lw=2, label='Active threshold')
    active_detail = (final_detail_var > 0.1).sum()
    axs[1,1].set_title(f'Detail Dims Variance (Final) - {active_detail}/64 active')
    axs[1,1].set_xlabel('Dimension'); axs[1,1].set_ylabel('Variance')
    axs[1,1].legend(); axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/dimension_activity.png', dpi=150, bbox_inches='tight')
plt.close()

# Dimension variance evolution heatmap
if dim_variance_history['core'] and len(dim_variance_history['core']) > 1:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    core_var_matrix = np.array(dim_variance_history['core'])
    im0 = axs[0].imshow(core_var_matrix.T, aspect='auto', cmap='viridis', 
                        extent=[1, len(core_var_matrix), split_idx, 0])
    axs[0].set_title('Core Dimension Variance Over Training')
    axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Dimension')
    plt.colorbar(im0, ax=axs[0], label='Variance')
    
    detail_var_matrix = np.array(dim_variance_history['detail'])
    im1 = axs[1].imshow(detail_var_matrix.T, aspect='auto', cmap='viridis',
                        extent=[1, len(detail_var_matrix), split_idx, 0])
    axs[1].set_title('Detail Dimension Variance Over Training')
    axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Dimension')
    plt.colorbar(im1, ax=axs[1], label='Variance')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dimension_variance_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n✓ Saved to {OUTPUT_DIR}/")
print("="*60 + "\nCOMPLETE\n" + "="*60)
