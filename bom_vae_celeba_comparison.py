"""
BOM-VAE vs β-VAE Comparison on CelebA

**Hypothesis**: BOM achieves comparable or better results than β-VAE without requiring hyperparameter tuning.

**Adaptive squeeze rule**:
```
squeeze_amount = (s_min - 0.5) * k
```
- When s_min = 0.9: squeeze aggressively
- When s_min = 0.55: squeeze gently
- When s_min ≤ 0.5: stop squeezing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==================== VAE Architecture ====================

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


# ==================== Data Loading ====================

def load_celeba(data_path, batch_size=64, image_size=64, num_workers=2):
    """
    Load CelebA dataset.

    Args:
        data_path: Path to CelebA root directory (should contain img_align_celeba folder)
        batch_size: Batch size
        image_size: Size to resize images to
        num_workers: Number of data loading workers
    """
    # CelebA images are 218x178, we'll crop to 178x178 then resize
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # Check if data_path points directly to img_align_celeba
    if os.path.basename(data_path) == 'img_align_celeba':
        # Use parent directory for ImageFolder
        data_path = os.path.dirname(data_path)

    # CelebA is typically structured as: root/img_align_celeba/image.jpg
    # ImageFolder expects: root/class/image.jpg, so img_align_celeba becomes the "class"
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )

    print(f"Loaded CelebA: {len(dataset)} images, {len(loader)} batches")
    return loader


# ==================== Shared Metrics ====================

def compute_metrics(x, x_recon, mu, logvar):
    """Compute MSE, KL, and sharpness."""
    B = x.size(0)
    mse = F.mse_loss(x_recon, x, reduction='none').view(B, -1).mean(1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    dx = torch.abs(x_recon[:,:,:,1:] - x_recon[:,:,:,:-1])
    dy = torch.abs(x_recon[:,:,1:,:] - x_recon[:,:,:-1,:])
    sharp = (dx.mean([1,2,3]) + dy.mean([1,2,3])) / 2
    return mse, kl, sharp


def evaluate(model, loader, device, max_batches=None):
    """Evaluate model on test set."""
    model.eval()
    all_mse, all_kl, all_sharp = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = batch[0].to(device)
            x_recon, mu, logvar = model(x)
            mse, kl, sharp = compute_metrics(x, x_recon, mu, logvar)
            all_mse.extend(mse.cpu().numpy())
            all_kl.extend(kl.cpu().numpy())
            all_sharp.extend(sharp.cpu().numpy())

    return {
        'mse': np.mean(all_mse),
        'kl': np.mean(all_kl),
        'sharp': np.mean(all_sharp),
    }


# ==================== β-VAE Training ====================

def train_beta_vae(model, loader, device, beta, n_epochs=20):
    """
    Standard β-VAE training.
    Loss = MSE + β * KL
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, epoch_mse, epoch_kl = [], [], []

        pbar = tqdm(loader, desc=f"β-VAE (β={beta}) Epoch {epoch}")
        for batch in pbar:
            x = batch[0].to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)

            mse, kl, sharp = compute_metrics(x, x_recon, mu, logvar)
            loss = mse.mean() + beta * kl.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_mse.append(mse.mean().item())
            epoch_kl.append(kl.mean().item())

            history.append({
                'mse': mse.mean().item(),
                'kl': kl.mean().item(),
                'sharp': sharp.mean().item(),
            })

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'mse': f"{mse.mean().item():.4f}", 'kl': f"{kl.mean().item():.0f}"})

        print(f"  Epoch {epoch}: loss={np.mean(epoch_loss):.4f}, mse={np.mean(epoch_mse):.4f}, kl={np.mean(epoch_kl):.0f}")

    return history


# ==================== BOM-VAE Training ====================

def regular_constraint_lower_better(value, floor):
    """Score for objectives where lower is better (MSE)."""
    return (floor - value) / floor


def regular_constraint_higher_better(value, ceiling):
    """Score for objectives where higher is better (sharpness)."""
    return value / ceiling


def box_constraint(value, floor_low, optimum, floor_high):
    """Score for objectives that need to stay in a range (KL)."""
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

    violations = (s_min <= 0).sum().item()

    # Track which constraints are failing
    mse_violations = (mse_score <= 0).sum().item()
    kl_violations = (kl_score <= 0).sum().item()
    sharp_violations = (sharp_score <= 0).sum().item()

    metrics = {
        'mse': mse.mean().item(),
        'kl': kl.mean().item(),
        'sharp': sharp.mean().item(),
        'mse_max': mse.max().item(),
        'kl_max': kl.max().item(),
        'kl_min': kl.min().item(),
        'sharp_min': sharp.min().item(),
        'mse_score': mse_score.mean().item(),
        'kl_score': kl_score.mean().item(),
        'sharp_score': sharp_score.mean().item(),
        's_min': s_min.mean().item(),
        'violations': violations,
        'mse_violations': mse_violations,
        'kl_violations': kl_violations,
        'sharp_violations': sharp_violations,
    }

    if violations > 0:
        return None, metrics

    loss = -torch.log(s_min).mean()
    names = ['mse', 'kl', 'sharp']
    metrics['bottleneck'] = names[torch.bincount(min_idx, minlength=3).argmax().item()]
    metrics['loss'] = loss.item()

    return loss, metrics


def calibrate_bom(model, loader, device, n_batches=50, extra_margin=1.0):
    """
    Calibrate BOM constraints based on model's current outputs.

    Args:
        extra_margin: Additional safety margin multiplier (1.0 = normal, 2.0 = 2x wider)
    """
    model.train()  # Important: use train mode for BatchNorm
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

    mse_arr = np.array(all_mse)
    kl_arr = np.array(all_kl)
    sharp_arr = np.array(all_sharp)

    # More conservative initial bounds with extra margin for robustness
    params = {
        'mse_floor': mse_arr.max() * 2.0 * extra_margin,
        'kl_floor_low': max(kl_arr.min() * 0.1, 0.1),  # Prevent zero
        'kl_optimum': kl_arr.mean(),
        'kl_floor_high': kl_arr.max() * 100.0 * extra_margin,  # Very loose initially, scaled by margin
        'sharp_ceiling': sharp_arr.mean() * 0.5,  # Start low for sharpness
    }

    print(f"Calibration (margin={extra_margin:.1f}x): MSE={mse_arr.mean():.4f} (max={mse_arr.max():.4f}), KL={kl_arr.mean():.1f} (range=[{kl_arr.min():.1f}, {kl_arr.max():.1f}]), Sharp={sharp_arr.mean():.4f}")
    print(f"Initial constraints: mse_floor={params['mse_floor']:.4f}, kl_box=[{params['kl_floor_low']:.1f}, {params['kl_optimum']:.1f}, {params['kl_floor_high']:.1f}], sharp_ceiling={params['sharp_ceiling']:.4f}")

    return params


def train_bom_vae(model, loader, device, n_epochs=20):
    """
    BOM-VAE training with adaptive squeeze and auto-recovery from early failures.

    Squeeze rule: squeeze_amount = (s_min - 0.5) * k
    - s_min > 0.5: squeeze proportionally
    - s_min <= 0.5: don't squeeze

    Auto-recovery: If early epochs have too many violations, automatically widen constraints.
    """
    # Calibrate with standard margin
    params = calibrate_bom(model, loader, device, extra_margin=1.0)

    mse_floor = params['mse_floor']
    kl_floor_low = params['kl_floor_low']
    kl_optimum = params['kl_optimum']
    kl_floor_high = params['kl_floor_high']
    sharp_ceiling = params['sharp_ceiling']

    # Targets
    target_kl_floor_low = 50
    target_kl_optimum = 80
    target_kl_floor_high = 150

    # Adaptive squeeze settings
    squeeze_k = 0.5  # Gain factor
    min_s_min_for_squeeze = 0.5
    squeeze_start_epoch = 3

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, epoch_s_min = [], []
        epoch_violations = 0
        epoch_mse_violations = 0
        epoch_kl_violations = 0
        epoch_sharp_violations = 0

        # Track actual values for reporting
        epoch_mse_values = []
        epoch_kl_values = []
        epoch_sharp_values = []

        pbar = tqdm(loader, desc=f"BOM-VAE Epoch {epoch}")
        for batch in pbar:
            x = batch[0].to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)

            loss, metrics = compute_bom_loss(
                x, x_recon, mu, logvar,
                mse_floor, kl_floor_low, kl_optimum, kl_floor_high, sharp_ceiling
            )

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

            # Track values
            epoch_mse_values.append(metrics['mse'])
            epoch_kl_values.append(metrics['kl'])
            epoch_sharp_values.append(metrics['sharp'])

            history.append(metrics)
            pbar.set_postfix({'s_min': f"{metrics['s_min']:.3f}", 'kl': f"{metrics['kl']:.0f}", 'viol': epoch_violations})

        avg_s_min = np.mean(epoch_s_min) if epoch_s_min else 0
        violation_rate = epoch_violations / (len(loader) * loader.batch_size) if len(loader) > 0 else 0

        print(f"  Epoch {epoch}: s_min={avg_s_min:.3f}, violations={epoch_violations} ({violation_rate*100:.1f}%)")
        print(f"    Values: MSE={np.mean(epoch_mse_values):.4f} (max={np.max(epoch_mse_values):.4f}), KL={np.mean(epoch_kl_values):.1f} (range=[{np.min(epoch_kl_values):.1f}, {np.max(epoch_kl_values):.1f}]), Sharp={np.mean(epoch_sharp_values):.4f}")
        print(f"    Constraints: mse_floor={mse_floor:.4f}, kl_box=[{kl_floor_low:.1f}, {kl_optimum:.1f}, {kl_floor_high:.1f}], sharp_ceil={sharp_ceiling:.4f}")

        # Report which constraints are failing
        if epoch_violations > 0:
            print(f"    ⚠️  Violations by constraint: MSE={epoch_mse_violations}, KL={epoch_kl_violations}, Sharp={epoch_sharp_violations}")

        # Auto-recovery: If early epochs have >20% violations, widen constraints
        if epoch <= 3 and violation_rate > 0.2:
            print(f"    🔧 AUTO-RECOVERY: {violation_rate*100:.1f}% violations detected in early epoch")

            if epoch_mse_violations > 0:
                old_mse_floor = mse_floor
                mse_floor = max(np.max(epoch_mse_values) * 3.0, mse_floor * 1.5)
                print(f"       MSE: Widening floor {old_mse_floor:.4f} -> {mse_floor:.4f}")

            if epoch_kl_violations > 0:
                old_kl_high = kl_floor_high
                kl_floor_high = max(np.max(epoch_kl_values) * 2.0, kl_floor_high * 2.0)
                print(f"       KL: Widening high bound {old_kl_high:.1f} -> {kl_floor_high:.1f}")

                # Also check if KL went below lower bound
                if np.min(epoch_kl_values) < kl_floor_low:
                    old_kl_low = kl_floor_low
                    kl_floor_low = max(np.min(epoch_kl_values) * 0.5, 0.1)
                    print(f"       KL: Widening low bound {old_kl_low:.1f} -> {kl_floor_low:.1f}")

            if epoch_sharp_violations > 0:
                old_sharp_ceil = sharp_ceiling
                sharp_ceiling = np.min(epoch_sharp_values) * 0.8
                print(f"       Sharp: Lowering ceiling {old_sharp_ceil:.4f} -> {sharp_ceiling:.4f}")

            print(f"    🔁 Continuing with widened constraints...")

        # Adaptive squeeze (only if not too many violations)
        if epoch >= squeeze_start_epoch and avg_s_min > min_s_min_for_squeeze and violation_rate < 0.1:
            squeeze_amount = (avg_s_min - min_s_min_for_squeeze) * squeeze_k
            squeeze_factor = 1.0 - squeeze_amount  # e.g., s_min=0.9 -> factor=0.8
            squeeze_factor = max(0.5, squeeze_factor)  # Don't squeeze more than 50%

            print(f"    🔧 Squeeze: s_min={avg_s_min:.3f} -> factor={squeeze_factor:.2f}")

            # Squeeze MSE floor
            mse_floor *= squeeze_factor

            # Squeeze KL box toward targets
            if kl_floor_low < target_kl_floor_low:
                kl_floor_low += (target_kl_floor_low - kl_floor_low) * (1 - squeeze_factor)
            if kl_optimum < target_kl_optimum:
                kl_optimum += (target_kl_optimum - kl_optimum) * (1 - squeeze_factor)
            if kl_floor_high > target_kl_floor_high:
                kl_floor_high -= (kl_floor_high - target_kl_floor_high) * (1 - squeeze_factor)

    return history


# ==================== Visualization ====================

def save_results_plots(results, output_dir='./outputs'):
    """Save all comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Training curves comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, data in results.items():
        h = data['history']
        label = name.replace('_', '=')

        axes[0].plot([x['mse'] for x in h], label=label, alpha=0.8)
        axes[1].plot([x['kl'] for x in h], label=label, alpha=0.8)
        axes[2].plot([x['sharp'] for x in h], label=label, alpha=0.8)

    axes[0].set_title('MSE (↓ better)')
    axes[0].set_xlabel('Step')
    axes[0].legend()
    axes[0].set_yscale('log')

    axes[1].set_title('KL Divergence')
    axes[1].set_xlabel('Step')
    axes[1].legend()

    axes[2].set_title('Sharpness (↑ better)')
    axes[2].set_xlabel('Step')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Pareto plot: MSE vs KL
    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        t = data['test']
        marker = 's' if 'beta' in name else 'o'
        size = 100 if 'bom' in name else 60
        plt.scatter(t['mse'], t['kl'], s=size, c=[color], marker=marker, label=name.replace('_', '='), edgecolors='black')

    plt.xlabel('MSE (↓ better)')
    plt.ylabel('KL (moderate is better)')
    plt.title('Pareto Front: MSE vs KL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/pareto_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_reconstructions(results, test_batch, output_dir='./outputs'):
    """Save reconstruction comparison."""
    n_models = len(results)
    fig, axes = plt.subplots(n_models + 1, 8, figsize=(16, 2*(n_models+1)))

    # Original
    for i in range(8):
        axes[0, i].imshow(test_batch[i].cpu().permute(1,2,0))
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12)

    # Each model's reconstruction
    for row, (name, data) in enumerate(results.items(), 1):
        model = data['model']
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(test_batch)

        for i in range(8):
            axes[row, i].imshow(recon[i].cpu().permute(1,2,0))
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(name.replace('_', '='), fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/reconstructions_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_samples(results, device, latent_dim=128, output_dir='./outputs'):
    """Save samples from prior comparison."""
    z = torch.randn(8, latent_dim, device=device)

    n_models = len(results)
    fig, axes = plt.subplots(n_models, 8, figsize=(16, 2*n_models))

    for row, (name, data) in enumerate(results.items()):
        model = data['model']
        model.eval()
        with torch.no_grad():
            samples = model.dec(model.fc_dec(z).view(-1, 256, 4, 4))

        for i in range(8):
            axes[row, i].imshow(samples[i].cpu().permute(1,2,0))
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(name.replace('_', '='), fontsize=12)

    plt.suptitle('Samples from Prior (same z for all models)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/samples_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==================== Main Comparison ====================

def run_comparison(celeba_path, n_epochs=20, batch_size=64, output_dir='./outputs'):
    """
    Run full BOM-VAE vs β-VAE comparison on CelebA.

    Args:
        celeba_path: Path to CelebA dataset (should contain img_align_celeba folder)
        n_epochs: Number of training epochs
        batch_size: Batch size
        output_dir: Directory to save outputs
    """
    # Load data
    train_loader = load_celeba(celeba_path, batch_size=batch_size)
    test_loader = load_celeba(celeba_path, batch_size=batch_size)  # In practice, use separate test set

    results = {}

    # β-VAE with different β values
    betas = [0.0001, 0.001, 0.01, 0.1]

    for beta in betas:
        print(f"\n{'='*60}")
        print(f"Training β-VAE with β={beta}")
        print('='*60)

        model = VAE(latent_dim=128).to(device)
        history = train_beta_vae(model, train_loader, device, beta=beta, n_epochs=n_epochs)
        test_metrics = evaluate(model, test_loader, device, max_batches=100)

        results[f'beta_{beta}'] = {
            'model': model,
            'history': history,
            'test': test_metrics,
        }

        print(f"\nTest results: MSE={test_metrics['mse']:.4f}, KL={test_metrics['kl']:.1f}, Sharp={test_metrics['sharp']:.4f}")

    # BOM-VAE
    print(f"\n{'='*60}")
    print(f"Training BOM-VAE (no β tuning required)")
    print('='*60)

    model_bom = VAE(latent_dim=128).to(device)
    history_bom = train_bom_vae(model_bom, train_loader, device, n_epochs=n_epochs)
    test_metrics_bom = evaluate(model_bom, test_loader, device, max_batches=100)

    results['bom'] = {
        'model': model_bom,
        'history': history_bom,
        'test': test_metrics_bom,
    }

    print(f"\nTest results: MSE={test_metrics_bom['mse']:.4f}, KL={test_metrics_bom['kl']:.1f}, Sharp={test_metrics_bom['sharp']:.4f}")

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Method':<20} {'MSE':>10} {'KL':>10} {'Sharpness':>12}")
    print("-"*70)

    for name, data in results.items():
        t = data['test']
        print(f"{name:<20} {t['mse']:>10.4f} {t['kl']:>10.1f} {t['sharp']:>12.4f}")

    print("-"*70)

    # Save visualizations
    print(f"\nSaving visualizations to {output_dir}...")
    save_results_plots(results, output_dir)

    # Get test batch for reconstructions and samples
    test_batch = next(iter(test_loader))[0][:8].to(device)
    save_reconstructions(results, test_batch, output_dir)
    save_samples(results, device, latent_dim=128, output_dir=output_dir)

    print(f"\nDone! Results saved to {output_dir}/")

    return results


# ==================== Entry Point ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BOM-VAE vs β-VAE comparison on CelebA')
    parser.add_argument('--celeba_path', type=str, required=True,
                        help='Path to CelebA dataset (should contain img_align_celeba folder)')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for results (default: ./outputs)')

    args = parser.parse_args()

    results = run_comparison(
        celeba_path=args.celeba_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
