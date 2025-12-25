# train.py
# Main training script - just run this!
# Edit configs/config.py to change settings

import os
import sys
import time
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    CALIBRATION_BATCHES, LATENT_DIM, IMAGE_CHANNELS, OUTPUT_DIR,
    EVAL_SAMPLES, NUM_TRAVERSE_DIMS, GOAL_SPECS, RECALIBRATION_EPOCHS,
    GROUP_NAMES
)
from utils.data import load_from_config
from models.vae import create_model
from models.vgg import VGGFeatures
from losses.goals import GoalSystem
from losses.bom_loss import compute_raw_losses, grouped_bom_loss, check_tensor
from utils.viz import (
    plot_group_balance, plot_goal_details, plot_reconstructions,
    plot_traversals, plot_cross_reconstruction, plot_dimension_activity,
    plot_training_history
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== SETUP ====================
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, data_info = load_from_config()

# Create model
model = create_model(LATENT_DIM, IMAGE_CHANNELS, DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# VGG for perceptual/texture
vgg = VGGFeatures(DEVICE)

# Goal system
goal_system = GoalSystem(GOAL_SPECS)

# Split index
split_idx = LATENT_DIM // 2

# ==================== HISTORIES ====================
histories = {
    'loss': [], 'min_group': [], 'bottleneck': [],
    'ssim': [], 'mse': [], 'edge': [],
    'kl_raw': [], 'detail_ratio_raw': [],
    'core_var_raw': [], 'detail_var_raw': [],
    'core_var_max_raw': [], 'detail_var_max_raw': [],
    'texture_loss': [], 'texture_dist_x2': [], 'texture_dist_x1': [],
    **{f'group_{n}': [] for n in GROUP_NAMES},
    'pixel': [], 'edge_goal': [], 'perceptual': [],
    'core_mse': [], 'core_edge': [], 'cross': [], 'texture_contrastive': [], 'texture_match': [],
    'kl_goal': [], 'cov_goal': [], 'weak': [],
    'detail_ratio_goal': [], 'core_var_goal': [], 'detail_var_goal': [],
    'core_var_max_goal': [], 'detail_var_max_goal': [],
    'core_active': [], 'detail_active': [],
    'core_effective': [], 'detail_effective': [],
}

dim_variance_history = {'core': [], 'detail': []}

# ==================== TRAINING ====================
print("\n" + "=" * 100)
print(f"BOM VAE v11 - {data_info['name'].upper()} - {EPOCHS} EPOCHS")
print("=" * 100 + "\n")

last_good_state = copy.deepcopy(model.state_dict())
last_good_optimizer = copy.deepcopy(optimizer.state_dict())
nan_recovery_count = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    epoch_data = {k: [] for k in histories.keys()}
    bn_counts = {n: 0 for n in GROUP_NAMES}
    skip_count = 0
    all_mu_core, all_mu_detail = [], []
    
    needs_recalibration = epoch in RECALIBRATION_EPOCHS or epoch == 1
    if needs_recalibration:
        goal_system.start_recalibration()
        print(f"\n📊 Epoch {epoch}: Collecting samples for calibration...")

    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch_idx, (x, _) in enumerate(pbar):
        x = x.to(DEVICE, non_blocking=True)
        if not check_tensor(x):
            skip_count += 1
            continue
        
        optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar, z = model(x)
        
        # Calibration phase
        if needs_recalibration and batch_idx < CALIBRATION_BATCHES:
            with torch.no_grad():
                raw_losses = compute_raw_losses(recon, x, mu, logvar, z, model, vgg, split_idx)
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
        
        # BOM training
        result = grouped_bom_loss(recon, x, mu, logvar, z, model, goal_system, vgg, split_idx, GROUP_NAMES)
        
        if result is None:
            skip_count += 1
            if skip_count > 10:
                print(f"\n⚠️  Too many NaN batches, recovering...")
                model.load_state_dict(last_good_state)
                optimizer.load_state_dict(last_good_optimizer)
                nan_recovery_count += 1
                skip_count = 0
                if nan_recovery_count > 5:
                    break
            continue
        
        loss = result['loss']
        loss.backward()
        
        # Check for NaN gradients
        has_nan_grad = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in model.parameters()
        )
        if has_nan_grad:
            skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Collect statistics
        if batch_idx % 10 == 0:
            with torch.no_grad():
                all_mu_core.append(mu[:, :split_idx].cpu())
                all_mu_detail.append(mu[:, split_idx:].cpu())
        
        if batch_idx % 200 == 0 and batch_idx > 0:
            last_good_state = copy.deepcopy(model.state_dict())
            last_good_optimizer = copy.deepcopy(optimizer.state_dict())
            skip_count = 0

        # Record metrics
        with torch.no_grad():
            groups = result['groups']
            group_vals = result['group_values']
            ind_goals = result['individual_goals']
            raw_vals = result['raw_values']
            
            epoch_data['loss'].append(loss.item())
            epoch_data['min_group'].append(groups.min().item())
            epoch_data['bottleneck'].append(result['min_idx'].item())
            epoch_data['ssim'].append(result['ssim'])
            epoch_data['mse'].append(result['mse'])
            epoch_data['edge'].append(result['edge_loss'])
            
            for key in ['kl_raw', 'detail_ratio_raw', 'core_var_raw', 'detail_var_raw',
                       'core_var_max_raw', 'detail_var_max_raw']:
                epoch_data[key].append(raw_vals[key])
            epoch_data['texture_loss'].append(raw_vals['texture_loss'])
            epoch_data['texture_dist_x2'].append(raw_vals['dist_x2'])
            epoch_data['texture_dist_x1'].append(raw_vals['dist_x1'])
            
            for n in GROUP_NAMES:
                epoch_data[f'group_{n}'].append(group_vals[n])
            bn_counts[GROUP_NAMES[result['min_idx'].item()]] += 1
            
            epoch_data['pixel'].append(ind_goals['pixel'])
            epoch_data['edge_goal'].append(ind_goals['edge'])
            epoch_data['perceptual'].append(ind_goals['perceptual'])
            epoch_data['core_mse'].append(ind_goals['core_mse'])
            epoch_data['core_edge'].append(ind_goals['core_edge'])
            epoch_data['cross'].append(ind_goals['cross'])
            epoch_data['texture_contrastive'].append(ind_goals['texture_contrastive'])
            epoch_data['texture_match'].append(ind_goals['texture_match'])
            epoch_data['kl_goal'].append(ind_goals['kl'])
            epoch_data['cov_goal'].append(ind_goals['cov'])
            epoch_data['weak'].append(ind_goals['weak'])
            epoch_data['detail_ratio_goal'].append(ind_goals['detail_ratio'])
            epoch_data['core_var_goal'].append(ind_goals['core_var'])
            epoch_data['detail_var_goal'].append(ind_goals['detail_var'])
            epoch_data['core_var_max_goal'].append(ind_goals['core_var_max'])
            epoch_data['detail_var_max_goal'].append(ind_goals['detail_var_max'])

        pbar.set_postfix({
            'loss': f"{loss.item():.2f}",
            'min': f"{groups.min().item():.3f}",
            'bn': GROUP_NAMES[result['min_idx'].item()],
            'ssim': f"{result['ssim']:.3f}",
        })

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

    # Update histories
    for k in histories.keys():
        if k == 'bottleneck':
            histories[k].append(max(bn_counts, key=bn_counts.get) if bn_counts else 'none')
        elif epoch_data[k]:
            histories[k].append(np.mean(epoch_data[k]))
        else:
            histories[k].append(histories[k][-1] if histories[k] else 0)

    # Print epoch summary
    d_x2 = histories['texture_dist_x2'][-1]
    d_x1 = histories['texture_dist_x1'][-1]
    print(f"\nEpoch {epoch:2d} | Loss: {histories['loss'][-1]:.3f} | Min: {histories['min_group'][-1]:.3f} | SSIM: {histories['ssim'][-1]:.3f}")
    print(f"         Texture: d(x2)={d_x2:.1f} d(x1)={d_x1:.1f} gap={d_x1-d_x2:.1f}")
    g_str = " | ".join([f"{n}:{histories[f'group_{n}'][-1]:.2f}" for n in GROUP_NAMES])
    print(f"         Groups: {g_str}\n")

# ==================== SAVE ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'histories': histories,
    'goal_specs': GOAL_SPECS,
    'goal_scales': goal_system.scales,
    'dim_variance_history': dim_variance_history,
    'data_info': data_info,
}, f'{OUTPUT_DIR}/bom_vae_v11.pt')
print(f"✓ Model saved to {OUTPUT_DIR}/bom_vae_v11.pt")

# ==================== EVAL ====================
print("\n" + "=" * 60 + "\nEVALUATION\n" + "=" * 60)

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except:
    os.system("pip install torchmetrics[image] -q")
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image import StructuralSimilarityIndexMeasure

fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)
ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

model.eval()
ssim_s, lpips_s, mse_t, cnt = [], [], 0, 0

with torch.no_grad():
    for x, _ in tqdm(train_loader, desc="Eval"):
        if cnt >= EVAL_SAMPLES:
            break
        x = x.to(DEVICE)
        r, _, _, _ = model(x)
        r = torch.clamp(r, 0, 1)
        fid.update(x, real=True)
        fid.update(r, real=False)
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

# Get samples for viz
samples, _ = next(iter(train_loader))

plot_group_balance(histories, GROUP_NAMES, f'{OUTPUT_DIR}/group_balance.png',
                   f"BOM VAE v11 - {data_info['name']}")
plot_goal_details(histories, GROUP_NAMES, f'{OUTPUT_DIR}/goal_details.png')
plot_reconstructions(model, samples, split_idx, f'{OUTPUT_DIR}/reconstructions.png', DEVICE)
plot_traversals(model, samples, split_idx,
                f'{OUTPUT_DIR}/traversals_core.png',
                f'{OUTPUT_DIR}/traversals_detail.png',
                NUM_TRAVERSE_DIMS, DEVICE)
plot_cross_reconstruction(model, samples, split_idx, f'{OUTPUT_DIR}/cross_reconstruction.png', DEVICE)
plot_dimension_activity(histories, dim_variance_history, split_idx, f'{OUTPUT_DIR}/dimension_activity.png')
plot_training_history(histories, f'{OUTPUT_DIR}/training_history.png')

print(f"\n✓ Visualizations saved to {OUTPUT_DIR}/")
print("=" * 60 + "\nCOMPLETE\n" + "=" * 60)
