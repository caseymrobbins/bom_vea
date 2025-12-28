# train.py - v15: Progressive group-by-group tightening
# Based on v14: Discriminator + Detail contracts
# Core = STRUCTURE (edges, geometry)
# Detail = APPEARANCE (colors, lighting)
# v15: Tighten one group per epoch (epochs 15,17,19,21,23,25,27) to force focus
import os, sys, time, copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import *
from utils.data import load_from_config
from models.vae import create_model
from models.discriminator import create_discriminator
from models.vgg import VGGFeatures
from losses.goals import GoalSystem
from losses.bom_loss import compute_raw_losses, grouped_bom_loss, check_tensor
from utils.viz import plot_group_balance, plot_reconstructions, plot_traversals, plot_cross_reconstruction, plot_training_history

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")

train_loader, data_info = load_from_config()
model = create_model(LATENT_DIM, IMAGE_CHANNELS, DEVICE)
discriminator = create_discriminator(IMAGE_CHANNELS, DEVICE)

# A100: Compile models for significant speedup (PyTorch 2.0+)
if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
    print("Compiling models with torch.compile (PyTorch 2.0+)...")
    model = torch.compile(model, mode='reduce-overhead')  # 'reduce-overhead' best for training
    discriminator = torch.compile(discriminator, mode='reduce-overhead')
    print("Models compiled!")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer_d = optim.AdamW(discriminator.parameters(), lr=LEARNING_RATE_D, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=EPOCHS, eta_min=1e-5)

vgg = VGGFeatures(DEVICE)
goal_system = GoalSystem(GOAL_SPECS)
split_idx = LATENT_DIM // 2

# Augmentation for consistency loss - batched on GPU
import torchvision.transforms as T

aug_transform = torch.nn.Sequential(
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
).to(DEVICE)

def apply_augmentation(x):
    return aug_transform(x)

histories = {
    'loss': [], 'min_group': [], 'bottleneck': [], 'ssim': [], 'mse': [], 'edge': [],
    'kl_core_raw': [], 'kl_detail_raw': [],
    'logvar_core_raw': [], 'logvar_detail_raw': [],
    'detail_ratio_raw': [], 'core_var_raw': [], 'detail_var_raw': [],
    'core_var_max_raw': [], 'detail_var_max_raw': [], 'consistency_raw': [],
    'structure_loss': [], 'appearance_loss': [], 'color_hist_loss': [],
    'realism_recon_raw': [], 'realism_swap_raw': [],
    'core_color_leak_raw': [], 'detail_edge_leak_raw': [],
    'detail_mean_raw': [], 'detail_var_mean_raw': [], 'detail_cov_raw': [],
    **{f'group_{n}': [] for n in GROUP_NAMES},
    'pixel': [], 'edge_goal': [], 'perceptual': [],
    'core_mse': [], 'core_edge': [],
    'swap_structure': [], 'swap_appearance': [], 'swap_color_hist': [],
    'realism_recon': [], 'realism_swap': [],
    'core_color_leak': [], 'detail_edge_leak': [],
    'kl_core_goal': [], 'kl_detail_goal': [],
    'cov_goal': [], 'weak': [], 'consistency_goal': [],
    'detail_mean_goal': [], 'detail_var_mean_goal': [], 'detail_cov_goal': [],
    'detail_ratio_goal': [], 'core_var_goal': [], 'detail_var_goal': [],
    'core_var_max_goal': [], 'detail_var_max_goal': [],
    'core_active': [], 'detail_active': [], 'core_effective': [], 'detail_effective': [],
}
dim_variance_history = {'core': [], 'detail': []}

print("\n" + "=" * 100)
print(f"BOM VAE v15 - {data_info['name'].upper()} - {EPOCHS} EPOCHS")
print("v15: Behavioral disentanglement walls (intervention testing)")
print("     - NEW: Direct leak detection (core→color, detail→edge)")
print("     - PatchGAN discriminator with spectral norm")
print("     - KL divergence for BOTH core and detail channels")
print("     - No clamps, fail-fast on barrier violations")
print(f"\nBOM: {'SOFTMIN' if USE_SOFTMIN else 'HARD MIN'} barrier (softmin disabled - unstable)")
print("=" * 100 + "\n")

# BOM: No "last good state" safety net - let barrier violations crash loudly
# last_good_state = copy.deepcopy(model.state_dict())
# last_good_state_d = copy.deepcopy(discriminator.state_dict())
# last_good_optimizer = copy.deepcopy(optimizer.state_dict())
# last_good_optimizer_d = copy.deepcopy(optimizer_d.state_dict())

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    epoch_data = {k: [] for k in histories.keys()}
    bn_counts = {n: 0 for n in GROUP_NAMES}
    skip_count = 0
    all_mu_core, all_mu_detail = [], []
    
    needs_recal = epoch in RECALIBRATION_EPOCHS or epoch == 1
    if needs_recal:
        # Progressive tightening: one group at a time
        if epoch in TIGHTENING_SCHEDULE:
            target_group = TIGHTENING_SCHEDULE[epoch]
            print(f"\n🔧 Epoch {epoch}: TIGHTENING '{target_group.upper()}' GROUP...")

            if target_group == 'latent':
                # Add upper bounds to KL: LOWER(1.0) → BOX_ASYMMETRIC[50, 5000]
                GOAL_SPECS['kl_core'] = {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 50, 'upper': 5000, 'healthy': 1000}
                GOAL_SPECS['kl_detail'] = {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 50, 'upper': 5000, 'healthy': 1000}
                # Tighten detail contracts: [-15, 15] → [-3, 3], [0.01, 350] → [0.1, 50]
                GOAL_SPECS['detail_mean']['lower'] = -3.0
                GOAL_SPECS['detail_mean']['upper'] = 3.0
                GOAL_SPECS['detail_var_mean']['lower'] = 0.1
                GOAL_SPECS['detail_var_mean']['upper'] = 50.0
                print(f"    KL: LOWER(1.0) → BOX_ASYMMETRIC[50, 5000] healthy=1000")
                print(f"    detail_mean: [-3, 3]")
                print(f"    detail_var_mean: [0.1, 50]")

            elif target_group == 'health':
                # Tighten health constraints: [0.0, 0.70] → [0.05, 0.50], [0.0, 300] → [0.1, 50]
                GOAL_SPECS['detail_ratio']['lower'] = 0.05
                GOAL_SPECS['detail_ratio']['upper'] = 0.50
                GOAL_SPECS['detail_var_mean']['lower'] = 0.1
                GOAL_SPECS['detail_var_mean']['upper'] = 50.0
                GOAL_SPECS['core_var_health']['lower'] = 0.1
                GOAL_SPECS['core_var_health']['upper'] = 50.0
                GOAL_SPECS['detail_var_health']['lower'] = 0.1
                GOAL_SPECS['detail_var_health']['upper'] = 50.0
                print(f"    detail_ratio: [0.05, 0.50]")
                print(f"    detail_var_mean: [0.1, 50]")
                print(f"    variance health: [0.1, 50]")

            elif target_group in ['recon', 'core', 'swap', 'realism', 'disentangle']:
                # These groups use MINIMIZE_SOFT with auto-scale
                # Tightening means recalibrating to current performance
                # This forces BOM to improve beyond current baseline
                print(f"    Recalibrating MINIMIZE_SOFT goals in {target_group} group")
                print(f"    (BOM will focus gradient here for 1-2 epochs)")

            # Reinitialize goal system with tightened specs
            goal_system = GoalSystem(GOAL_SPECS)

        goal_system.start_recalibration()
        print(f"\n📊 Epoch {epoch}: Calibrating {TIGHTENING_SCHEDULE.get(epoch, 'all groups')}...")

    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch_idx, (x, _) in enumerate(pbar):
        x = x.to(DEVICE, non_blocking=True)
        if not check_tensor(x): skip_count += 1; continue
        
        with torch.no_grad():
            x_aug = apply_augmentation(x)
        
        optimizer.zero_grad(set_to_none=True)
        recon, mu, logvar, z = model(x)
        
        # v14: Train discriminator first
        if goal_system.calibrated and batch_idx % 2 == 0:  # Train D every other step
            discriminator.train()
            optimizer_d.zero_grad(set_to_none=True)

            # Real images should get score 1
            d_real = discriminator(x)
            loss_d_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))

            # Fake images (recon) should get score 0
            with torch.no_grad():
                recon_for_d = model(x)[0].detach()
            d_fake = discriminator(recon_for_d)
            loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_d.step()

        if needs_recal and batch_idx < CALIBRATION_BATCHES:
            with torch.no_grad():
                raw = compute_raw_losses(recon, x, mu, logvar, z, model, vgg, split_idx, discriminator, x_aug)
                goal_system.collect(raw)
            if not goal_system.calibrated:
                F.mse_loss(recon, x).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pbar.set_postfix({'phase': 'CALIBRATING'})
                continue

        if needs_recal and batch_idx == CALIBRATION_BATCHES:
            goal_system.calibrate(epoch=epoch)
            needs_recal = False

        result = grouped_bom_loss(recon, x, mu, logvar, z, model, goal_system, vgg, split_idx, GROUP_NAMES, discriminator, x_aug, USE_SOFTMIN, SOFTMIN_TEMPERATURE)

        # BOM philosophy: Let it crash loudly if constraints violated, don't mask with reloads
        if result is None:
            print(f"ERROR: Loss computation failed (returned None) at epoch {epoch}, batch {batch_idx}")
            print(f"This means a barrier was violated or NaN/Inf detected.")
            print(f"Fix: Adjust initialization or widen BOX constraints, don't mask the crash.")
            raise RuntimeError("BOM barrier violation - loss returned None")

        loss = result['loss']
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"ERROR: Loss is NaN/Inf at epoch {epoch}, batch {batch_idx}")
            print(f"loss value: {loss.item()}")
            raise RuntimeError("BOM barrier violation - loss is NaN/Inf")

        loss.backward()

        # BOM philosophy: No grad clipping - if gradients explode, the barrier is wrong
        # if any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in model.parameters()):
        #     skip_count += 1; optimizer.zero_grad(set_to_none=True); continue
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        
        if batch_idx % 10 == 0:
            with torch.no_grad():
                all_mu_core.append(mu[:, :split_idx].cpu())
                all_mu_detail.append(mu[:, split_idx:].cpu())
        
        # BOM: No checkpointing "last good state" - let failures be visible
        # if batch_idx % 200 == 0 and batch_idx > 0:
        #     last_good_state = copy.deepcopy(model.state_dict())
        #     last_good_state_d = copy.deepcopy(discriminator.state_dict())
        #     last_good_optimizer = copy.deepcopy(optimizer.state_dict())
        #     last_good_optimizer_d = copy.deepcopy(optimizer_d.state_dict())

        with torch.no_grad():
            groups = result['groups']
            gv, ig, rv = result['group_values'], result['individual_goals'], result['raw_values']
            
            epoch_data['loss'].append(loss.item())
            epoch_data['min_group'].append(groups.min().item())
            epoch_data['bottleneck'].append(result['min_idx'].item())
            epoch_data['ssim'].append(result['ssim'])
            epoch_data['mse'].append(result['mse'])
            epoch_data['edge'].append(result['edge_loss'])

            # v15: Updated raw values including leak detection and logvar tracking
            for k in ['kl_core_raw', 'kl_detail_raw', 'logvar_core_raw', 'logvar_detail_raw',
                     'detail_ratio_raw', 'core_var_raw', 'detail_var_raw',
                     'core_var_max_raw', 'detail_var_max_raw', 'consistency_raw',
                     'detail_mean_raw', 'detail_var_mean_raw', 'detail_cov_raw',
                     'realism_recon_raw', 'realism_swap_raw',
                     'core_color_leak_raw', 'detail_edge_leak_raw']:
                epoch_data[k].append(rv.get(k, 0))
            epoch_data['structure_loss'].append(rv['structure_loss'])
            epoch_data['appearance_loss'].append(rv['appearance_loss'])
            epoch_data['color_hist_loss'].append(rv['color_hist_loss'])
            
            for n in GROUP_NAMES: epoch_data[f'group_{n}'].append(gv[n])
            bn_counts[GROUP_NAMES[result['min_idx'].item()]] += 1
            
            epoch_data['pixel'].append(ig['pixel'])
            epoch_data['edge_goal'].append(ig['edge'])
            epoch_data['perceptual'].append(ig['perceptual'])
            epoch_data['core_mse'].append(ig['core_mse'])
            epoch_data['core_edge'].append(ig['core_edge'])
            epoch_data['swap_structure'].append(ig['swap_structure'])
            epoch_data['swap_appearance'].append(ig['swap_appearance'])
            epoch_data['swap_color_hist'].append(ig['swap_color_hist'])
            epoch_data['realism_recon'].append(ig['realism_recon'])
            epoch_data['realism_swap'].append(ig['realism_swap'])
            epoch_data['core_color_leak'].append(ig['core_color_leak'])
            epoch_data['detail_edge_leak'].append(ig['detail_edge_leak'])
            epoch_data['kl_core_goal'].append(ig['kl_core'])
            epoch_data['kl_detail_goal'].append(ig['kl_detail'])
            epoch_data['cov_goal'].append(ig['cov'])
            epoch_data['weak'].append(ig['weak'])
            epoch_data['consistency_goal'].append(ig.get('consistency', 0.5))
            epoch_data['detail_mean_goal'].append(ig['detail_mean'])
            epoch_data['detail_var_mean_goal'].append(ig['detail_var_mean'])
            epoch_data['detail_cov_goal'].append(ig['detail_cov'])
            epoch_data['detail_ratio_goal'].append(ig['detail_ratio'])
            epoch_data['core_var_goal'].append(ig['core_var'])
            epoch_data['detail_var_goal'].append(ig['detail_var'])
            epoch_data['core_var_max_goal'].append(ig['core_var_max'])
            epoch_data['detail_var_max_goal'].append(ig['detail_var_max'])

        pbar.set_postfix({
            'loss': f"{loss.item():.2f}", 'min': f"{groups.min().item():.3f}",
            'bn': GROUP_NAMES[result['min_idx'].item()], 'ssim': f"{result['ssim']:.3f}",
        })

    if nan_count > 5: 
        print("Too many instability issues. Stopping.")
        break


    scheduler.step()
    scheduler_d.step()
    last_good_state = copy.deepcopy(model.state_dict())
    last_good_state_d = copy.deepcopy(discriminator.state_dict())
    last_good_optimizer = copy.deepcopy(optimizer.state_dict())
    last_good_optimizer_d = copy.deepcopy(optimizer_d.state_dict())
    
    if all_mu_core:
        mc, md = torch.cat(all_mu_core), torch.cat(all_mu_detail)
        cv, dv = mc.var(0), md.var(0)
        dim_variance_history['core'].append(cv.numpy())
        dim_variance_history['detail'].append(dv.numpy())
        epoch_data['core_active'] = [(cv > 0.1).sum().item()]
        epoch_data['detail_active'] = [(dv > 0.1).sum().item()]
        cvn = cv / (cv.sum() + 1e-8) + 1e-8
        dvn = dv / (dv.sum() + 1e-8) + 1e-8
        epoch_data['core_effective'] = [torch.exp(-torch.sum(cvn * torch.log(cvn))).item()]
        epoch_data['detail_effective'] = [torch.exp(-torch.sum(dvn * torch.log(dvn))).item()]

    for k in histories:
        if k == 'bottleneck': histories[k].append(max(bn_counts, key=bn_counts.get) if bn_counts else 'none')
        elif epoch_data[k]: histories[k].append(np.mean(epoch_data[k]))
        else: histories[k].append(histories[k][-1] if histories[k] else 0)

    struct = histories['structure_loss'][-1]
    appear = histories['appearance_loss'][-1]
    kl_c = histories['kl_core_raw'][-1]
    kl_d = histories['kl_detail_raw'][-1]

    # Calculate bottleneck percentages
    total_batches = sum(bn_counts.values())
    bn_pcts = {n: (bn_counts.get(n, 0) / total_batches * 100) if total_batches > 0 else 0 for n in GROUP_NAMES}

    print(f"\nEpoch {epoch:2d} | Loss: {histories['loss'][-1]:.3f} | Min: {histories['min_group'][-1]:.3f} | SSIM: {histories['ssim'][-1]:.3f}")
    print(f"         Structure: {struct:.4f} | Appearance: {appear:.4f}")
    print(f"         KL_core: {kl_c:.1f} | KL_detail: {kl_d:.1f}")
    print(f"         Groups: " + " | ".join(f"{n}:{histories[f'group_{n}'][-1]:.2f}" for n in GROUP_NAMES))
    print(f"         Bottlenecks: " + " | ".join(f"{n}:{bn_pcts[n]:.1f}%" for n in GROUP_NAMES))

# SAVE
torch.save({
    'model': model.state_dict(),
    'discriminator': discriminator.state_dict(),
    'histories': histories,
    'scales': goal_system.scales,
    'dim_var': dim_variance_history
}, f'{OUTPUT_DIR}/bom_vae_v15.pt')
print(f"\n✓ Saved to {OUTPUT_DIR}/bom_vae_v15.pt")

# EVAL
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
lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)
ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

model.eval()
ss, lp, mse_t, cnt = [], [], 0, 0
with torch.no_grad():
    for x, _ in tqdm(train_loader, desc="Eval"):
        if cnt >= EVAL_SAMPLES: break
        x = x.to(DEVICE); r = torch.clamp(model(x)[0], 0, 1)
        fid.update(x, real=True); fid.update(r, real=False)
        ss.append(ssim_m(r, x).item()); lp.append(lpips_m(x, r).item())
        mse_t += F.mse_loss(r, x, reduction='sum').item(); cnt += x.shape[0]

print(f"\n  MSE:   {mse_t/(cnt*3*64*64):.6f}\n  SSIM:  {np.mean(ss):.4f}\n  LPIPS: {np.mean(lp):.4f}\n  FID:   {fid.compute().item():.2f}")

# VIZ
print("\nGenerating visualizations...")
samples, _ = next(iter(train_loader))
plot_group_balance(histories, GROUP_NAMES, f'{OUTPUT_DIR}/group_balance.png', f"BOM VAE v15 - {data_info['name']}")
plot_reconstructions(model, samples, split_idx, f'{OUTPUT_DIR}/reconstructions.png', DEVICE)
plot_traversals(model, samples, split_idx, f'{OUTPUT_DIR}/traversals_core.png', f'{OUTPUT_DIR}/traversals_detail.png', NUM_TRAVERSE_DIMS, DEVICE)
plot_cross_reconstruction(model, samples, split_idx, f'{OUTPUT_DIR}/cross_reconstruction.png', DEVICE)
plot_training_history(histories, f'{OUTPUT_DIR}/training_history.png')

print(f"\n✓ All saved to {OUTPUT_DIR}/")
print("=" * 60 + "\nCOMPLETE\n" + "=" * 60)
