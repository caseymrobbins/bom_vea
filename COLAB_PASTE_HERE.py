"""
BOM-VAE vs β-VAE CelebA Comparison - Colab-Ready Script

INSTRUCTIONS:
1. In Google Colab, select Runtime → Change runtime type → L4 GPU
2. Copy this ENTIRE file into a single Colab cell
3. Run the cell
4. Wait ~3-4 hours for results

IMPROVEMENTS IN THIS VERSION:
- Auto-recovery from KL explosions during early training
- Detailed violation reporting (shows which constraints failed)
- Automatic constraint widening if calibration was too tight
- More conservative initial bounds with safety margins
"""

# ==================== SETUP ====================
print("Installing dependencies...")
import subprocess
subprocess.run(["pip", "install", "gdown", "-q"], check=True)

import os, zipfile, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display, Image as IPImage
import gdown

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Device: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ==================== DOWNLOAD CELEBA ====================
celeba_path = '/content/celeba'
zip_path = '/content/celeba.zip'

if not os.path.exists(celeba_path) or len(os.listdir(celeba_path)) == 0:
    print("\n" + "="*70)
    print("CELEBA DATASET REQUIRED")
    print("="*70)
    print("\nOption 1: Download from Kaggle (RECOMMENDED)")
    print("-" * 70)
    print("1. Go to: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
    print("2. Click 'Download' (requires Kaggle account)")
    print("3. Upload 'archive.zip' to Colab")
    print("4. Run this cell again")
    print("\nOption 2: Use Kaggle API (if you have kaggle.json)")
    print("-" * 70)
    print("Run these commands in a separate cell:")
    print("  !pip install kaggle")
    print("  !mkdir -p ~/.kaggle")
    print("  # Upload your kaggle.json to /content/")
    print("  !cp /content/kaggle.json ~/.kaggle/")
    print("  !chmod 600 ~/.kaggle/kaggle.json")
    print("  !kaggle datasets download -d jessicali9530/celeba-dataset")
    print("  !unzip -q celeba-dataset.zip -d /content/celeba")
    print("\nOption 3: Google Drive (manual)")
    print("-" * 70)
    print("1. Download from: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8")
    print("2. Upload img_align_celeba.zip to your Google Drive")
    print("3. Mount Drive in Colab:")
    print("  from google.colab import drive")
    print("  drive.mount('/content/drive')")
    print("  !unzip /content/drive/MyDrive/img_align_celeba.zip -d /content/celeba")
    print("\nOption 4: Try automatic download (may fail due to rate limits)")
    print("-" * 70)

    try:
        if not os.path.exists(zip_path):
            print("Attempting automatic download...")
            # Try the direct file link format (more reliable)
            url = "https://drive.google.com/file/d/1xJs_8JB0HYXiaAmU8PTG9qbk0WJ2Wo1U/view?usp=sharing"
            gdown.download(url, zip_path, quiet=False, fuzzy=True)

        if os.path.exists(zip_path):
            print("Extracting dataset...")
            os.makedirs(celeba_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(celeba_path)
            print("✓ Extraction complete!")
    except Exception as e:
        print(f"\n⚠️  Automatic download failed: {e}")
        print("\nPlease use one of the manual methods above.")
        print("After downloading, the structure should be:")
        print("  /content/celeba/")
        print("    └── img_align_celeba/")
        print("        ├── 000001.jpg")
        print("        ├── 000002.jpg")
        print("        └── ...")
        raise RuntimeError("CelebA dataset not found. Please download manually (see instructions above).")

num_images = len(glob.glob(f"{celeba_path}/**/*.jpg", recursive=True))
if num_images == 0:
    print("\n❌ No images found!")
    print("Expected structure:")
    print("  /content/celeba/")
    print("    └── img_align_celeba/")
    print("        ├── 000001.jpg")
    print("        ├── 000002.jpg")
    print("        └── ...")
    raise RuntimeError("CelebA dataset not found. Please download manually.")

print(f"✓ Found {num_images:,} images\n")

# ==================== VAE MODEL ====================
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(self.fc_dec(z).view(-1, 256, 4, 4)), mu, logvar

# ==================== DATA LOADING ====================
def load_celeba(data_path, batch_size=128):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    if os.path.basename(data_path) == 'img_align_celeba':
        data_path = os.path.dirname(data_path)
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print(f"✓ Loaded CelebA: {len(dataset):,} images, {len(loader)} batches")
    return loader

BATCH_SIZE = 128  # Change to 64 for T4, 256 for A100
train_loader = load_celeba(celeba_path, batch_size=BATCH_SIZE)
test_loader = load_celeba(celeba_path, batch_size=BATCH_SIZE)

# ==================== METRICS ====================
def compute_metrics(x, x_recon, mu, logvar):
    B = x.size(0)
    mse = F.mse_loss(x_recon, x, reduction='none').view(B, -1).mean(1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    dx = torch.abs(x_recon[:,:,:,1:] - x_recon[:,:,:,:-1])
    dy = torch.abs(x_recon[:,:,1:,:] - x_recon[:,:,:-1,:])
    sharp = (dx.mean([1,2,3]) + dy.mean([1,2,3])) / 2
    return mse, kl, sharp

def evaluate(model, loader, device, max_batches=100):
    model.eval()
    all_mse, all_kl, all_sharp = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            x = batch[0].to(device)
            x_recon, mu, logvar = model(x)
            mse, kl, sharp = compute_metrics(x, x_recon, mu, logvar)
            all_mse.extend(mse.cpu().numpy())
            all_kl.extend(kl.cpu().numpy())
            all_sharp.extend(sharp.cpu().numpy())
    return {'mse': np.mean(all_mse), 'kl': np.mean(all_kl), 'sharp': np.mean(all_sharp)}

# ==================== β-VAE TRAINING ====================
def train_beta_vae(model, loader, device, beta, n_epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"β-VAE (β={beta}) Epoch {epoch}/{n_epochs}")
        for batch in pbar:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            mse, kl, sharp = compute_metrics(x, x_recon, mu, logvar)
            loss = mse.mean() + beta * kl.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append({'mse': mse.mean().item(), 'kl': kl.mean().item(), 'sharp': sharp.mean().item()})
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'kl': f"{kl.mean().item():.0f}"})
    return history

# ==================== BOM-VAE TRAINING (WITH AUTO-RECOVERY) ====================
def regular_constraint_lower_better(value, floor):
    return (floor - value) / floor

def regular_constraint_higher_better(value, ceiling):
    return value / ceiling

def box_constraint(value, floor_low, optimum, floor_high):
    left = (value - floor_low) / (optimum - floor_low)
    right = (floor_high - value) / (floor_high - optimum)
    return torch.minimum(left, right)

def compute_bom_loss(x, x_recon, mu, logvar, mse_floor, kl_floor_low, kl_optimum, kl_floor_high, sharp_ceiling):
    """Compute BOM loss with detailed violation reporting."""
    mse, kl, sharp = compute_metrics(x, x_recon, mu, logvar)
    mse_score = regular_constraint_lower_better(mse, mse_floor)
    kl_score = box_constraint(kl, kl_floor_low, kl_optimum, kl_floor_high)
    sharp_score = regular_constraint_higher_better(sharp, sharp_ceiling)
    scores = torch.stack([mse_score, kl_score, sharp_score], dim=1)
    s_min, min_idx = torch.min(scores, dim=1)

    metrics = {
        'mse': mse.mean().item(), 'kl': kl.mean().item(), 'sharp': sharp.mean().item(),
        'mse_max': mse.max().item(), 'kl_max': kl.max().item(), 'kl_min': kl.min().item(),
        'sharp_min': sharp.min().item(), 's_min': s_min.mean().item(),
        'violations': (s_min <= 0).sum().item(),
        'mse_violations': (mse_score <= 0).sum().item(),
        'kl_violations': (kl_score <= 0).sum().item(),
        'sharp_violations': (sharp_score <= 0).sum().item(),
    }

    if metrics['violations'] > 0:
        return None, metrics

    loss = -torch.log(s_min).mean()
    names = ['mse', 'kl', 'sharp']
    metrics['bottleneck'] = names[torch.bincount(min_idx, minlength=3).argmax().item()]
    metrics['loss'] = loss.item()
    return loss, metrics

def calibrate_bom(model, loader, device, n_batches=50):
    model.train()
    all_mse, all_kl, all_sharp = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches: break
            x = batch[0].to(device)
            x_recon, mu, logvar = model(x)
            mse, kl, sharp = compute_metrics(x, x_recon, mu, logvar)
            all_mse.extend(mse.cpu().numpy())
            all_kl.extend(kl.cpu().numpy())
            all_sharp.extend(sharp.cpu().numpy())

    mse_arr, kl_arr, sharp_arr = np.array(all_mse), np.array(all_kl), np.array(all_sharp)
    params = {
        'mse_floor': mse_arr.max() * 2.0,
        'kl_floor_low': max(kl_arr.min() * 0.1, 0.1),
        'kl_optimum': kl_arr.mean(),
        'kl_floor_high': kl_arr.max() * 100.0,  # Very loose initially
        'sharp_ceiling': sharp_arr.mean() * 0.5,
    }
    print(f"Calibration: MSE={mse_arr.mean():.4f} (max={mse_arr.max():.4f}), KL={kl_arr.mean():.1f} (range=[{kl_arr.min():.1f}, {kl_arr.max():.1f}])")
    print(f"Initial constraints: mse_floor={params['mse_floor']:.4f}, kl_box=[{params['kl_floor_low']:.1f}, {params['kl_optimum']:.1f}, {params['kl_floor_high']:.1f}]\n")
    return params

def train_bom_vae(model, loader, device, n_epochs=20):
    """BOM-VAE training with auto-recovery from early failures."""
    params = calibrate_bom(model, loader, device)
    mse_floor = params['mse_floor']
    kl_floor_low = params['kl_floor_low']
    kl_optimum = params['kl_optimum']
    kl_floor_high = params['kl_floor_high']
    sharp_ceiling = params['sharp_ceiling']

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, epoch_s_min = [], []
        epoch_violations = 0
        epoch_mse_violations, epoch_kl_violations, epoch_sharp_violations = 0, 0, 0
        epoch_mse_values, epoch_kl_values, epoch_sharp_values = [], [], []

        pbar = tqdm(loader, desc=f"BOM-VAE Epoch {epoch}/{n_epochs}")
        for batch in pbar:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss, metrics = compute_bom_loss(x, x_recon, mu, logvar, mse_floor, kl_floor_low, kl_optimum, kl_floor_high, sharp_ceiling)

            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss.append(metrics['loss'])
                epoch_s_min.append(metrics['s_min'])
            else:
                epoch_violations += metrics['violations']
                epoch_mse_violations += metrics['mse_violations']
                epoch_kl_violations += metrics['kl_violations']
                epoch_sharp_violations += metrics['sharp_violations']

            epoch_mse_values.append(metrics['mse'])
            epoch_kl_values.append(metrics['kl'])
            epoch_sharp_values.append(metrics['sharp'])
            history.append(metrics)
            pbar.set_postfix({'s_min': f"{metrics['s_min']:.3f}", 'kl': f"{metrics['kl']:.0f}", 'viol': epoch_violations})

        avg_s_min = np.mean(epoch_s_min) if epoch_s_min else 0
        violation_rate = epoch_violations / (len(loader) * loader.batch_size)

        print(f"  Epoch {epoch}: s_min={avg_s_min:.3f}, violations={epoch_violations} ({violation_rate*100:.1f}%)")
        print(f"    Values: MSE={np.mean(epoch_mse_values):.4f} (max={np.max(epoch_mse_values):.4f}), KL={np.mean(epoch_kl_values):.1f} (range=[{np.min(epoch_kl_values):.1f}, {np.max(epoch_kl_values):.1f}])")

        if epoch_violations > 0:
            print(f"    ⚠️  Violations: MSE={epoch_mse_violations}, KL={epoch_kl_violations}, Sharp={epoch_sharp_violations}")

        # Auto-recovery for early epochs with >20% violations
        if epoch <= 3 and violation_rate > 0.2:
            print(f"    🔧 AUTO-RECOVERY: {violation_rate*100:.1f}% violations in early epoch")
            if epoch_mse_violations > 0:
                mse_floor = max(np.max(epoch_mse_values) * 3.0, mse_floor * 1.5)
                print(f"       MSE: Widened floor to {mse_floor:.4f}")
            if epoch_kl_violations > 0:
                kl_floor_high = max(np.max(epoch_kl_values) * 2.0, kl_floor_high * 2.0)
                print(f"       KL: Widened high bound to {kl_floor_high:.1f}")
                if np.min(epoch_kl_values) < kl_floor_low:
                    kl_floor_low = max(np.min(epoch_kl_values) * 0.5, 0.1)
                    print(f"       KL: Widened low bound to {kl_floor_low:.1f}")
            if epoch_sharp_violations > 0:
                sharp_ceiling = np.min(epoch_sharp_values) * 0.8
                print(f"       Sharp: Lowered ceiling to {sharp_ceiling:.4f}")

        # Adaptive squeeze (only if not too many violations)
        if epoch >= 3 and avg_s_min > 0.5 and violation_rate < 0.1:
            squeeze_factor = max(0.5, 1.0 - (avg_s_min - 0.5) * 0.5)
            print(f"    🔧 Squeeze: factor={squeeze_factor:.2f}")
            mse_floor *= squeeze_factor
            kl_floor_low = min(kl_floor_low + (50 - kl_floor_low) * (1 - squeeze_factor), 50)
            kl_optimum = min(kl_optimum + (80 - kl_optimum) * (1 - squeeze_factor), 80)
            kl_floor_high = max(kl_floor_high - (kl_floor_high - 150) * (1 - squeeze_factor), 150)

    return history

# ==================== RUN COMPARISON ====================
N_EPOCHS = 20  # Change to 5 for quick test
results = {}

print(f"\n{'='*60}\nTRAINING COMPARISON: {N_EPOCHS} EPOCHS\n{'='*60}\n")

for beta in [0.0001, 0.001, 0.01, 0.1]:
    print(f"\n{'='*60}\nβ-VAE (β={beta})\n{'='*60}")
    model = VAE(latent_dim=128).to(device)
    history = train_beta_vae(model, train_loader, device, beta=beta, n_epochs=N_EPOCHS)
    test_metrics = evaluate(model, test_loader, device, max_batches=100)
    results[f'beta_{beta}'] = {'model': model, 'history': history, 'test': test_metrics}
    print(f"Test: MSE={test_metrics['mse']:.4f}, KL={test_metrics['kl']:.1f}, Sharp={test_metrics['sharp']:.4f}")

print(f"\n{'='*60}\nBOM-VAE (no β tuning)\n{'='*60}")
model_bom = VAE(latent_dim=128).to(device)
history_bom = train_bom_vae(model_bom, train_loader, device, n_epochs=N_EPOCHS)
test_metrics_bom = evaluate(model_bom, test_loader, device, max_batches=100)
results['bom'] = {'model': model_bom, 'history': history_bom, 'test': test_metrics_bom}
print(f"Test: MSE={test_metrics_bom['mse']:.4f}, KL={test_metrics_bom['kl']:.1f}, Sharp={test_metrics_bom['sharp']:.4f}")

# ==================== RESULTS ====================
print("\n" + "="*70 + "\nFINAL RESULTS\n" + "="*70)
print(f"{'Method':<20} {'MSE':>10} {'KL':>10} {'Sharpness':>12}")
print("-"*70)
for name, data in results.items():
    t = data['test']
    print(f"{name:<20} {t['mse']:>10.4f} {t['kl']:>10.1f} {t['sharp']:>12.4f}")
print("-"*70)

# ==================== VISUALIZATIONS ====================
# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name, data in results.items():
    h = data['history']
    axes[0].plot([x['mse'] for x in h], label=name.replace('_', '='), alpha=0.8)
    axes[1].plot([x['kl'] for x in h], label=name.replace('_', '='), alpha=0.8)
    axes[2].plot([x['sharp'] for x in h], label=name.replace('_', '='), alpha=0.8)
axes[0].set_title('MSE (↓)'); axes[0].set_yscale('log'); axes[0].legend()
axes[1].set_title('KL'); axes[1].legend()
axes[2].set_title('Sharpness (↑)'); axes[2].legend()
plt.tight_layout()
plt.savefig('/content/training_comparison.png', dpi=150)
plt.show()

# Pareto plot
plt.figure(figsize=(10, 6))
for name, data in results.items():
    t = data['test']
    marker = 's' if 'beta' in name else 'o'
    size = 150 if 'bom' in name else 80
    plt.scatter(t['mse'], t['kl'], s=size, marker=marker, label=name.replace('_', '='), edgecolors='black', linewidths=2)
plt.xlabel('MSE (↓)'); plt.ylabel('KL'); plt.title('Pareto Front'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig('/content/pareto.png', dpi=150)
plt.show()

# Reconstructions
test_batch = next(iter(test_loader))[0][:8].to(device)
fig, axes = plt.subplots(len(results) + 1, 8, figsize=(16, 2*(len(results)+1)))
for i in range(8):
    axes[0, i].imshow(test_batch[i].cpu().permute(1,2,0)); axes[0, i].axis('off')
axes[0, 0].set_ylabel('Original', fontsize=10)
for row, (name, data) in enumerate(results.items(), 1):
    data['model'].eval()
    with torch.no_grad():
        recon, _, _ = data['model'](test_batch)
    for i in range(8):
        axes[row, i].imshow(recon[i].cpu().permute(1,2,0)); axes[row, i].axis('off')
    axes[row, 0].set_ylabel(name.replace('_', '='), fontsize=10)
plt.tight_layout()
plt.savefig('/content/reconstructions.png', dpi=150)
plt.show()

print("\n✓ Done! Download plots from Files panel on the left")
