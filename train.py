# train.py - v17: "Lazy Optimizer" design with asymmetric KL squeeze
# Core = STRUCTURE (edges, geometry)
# Detail = APPEARANCE (colors, lighting)
# Latent group has 4 sub-groups: KL, Structure, Capacity, Detail Stats
# v17 changes: KL squeeze 15k→3k (epochs 2-15), appearance upper bound, relaxed capacity (0.4)
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

def diagnose_rollback(check_result, group_names):
    """Provide detailed diagnostic information about why a rollback occurred."""
    if check_result is None:
        return "S_min ≤ 0 | Complete constraint violation"

    # FRONT-LOAD critical info: show min group score FIRST
    gv = check_result.get('group_values', {})
    if gv:
        min_group_name = group_names[check_result['min_idx'].item()]
        min_group_score = gv[min_group_name]
        prefix = f"min_group={min_group_score:.4f} ({min_group_name})"
    else:
        prefix = "Unknown group"

    diag = []

    # Check for NaN/Inf loss
    if torch.isnan(check_result['loss']):
        diag.append("Loss=NaN")
    if torch.isinf(check_result['loss']):
        diag.append("Loss=Inf")

    # Show top 3 failing individual goals (≤ 0.01 = danger zone)
    ig = check_result.get('individual_goals', {})
    failing_goals = {name: val for name, val in ig.items() if val <= 0.01}
    if failing_goals:
        top_failing = sorted(failing_goals.items(), key=lambda x: x[1])[:3]
        diag.append(f"Failing: {', '.join(f'{k}={v:.4f}' for k, v in top_failing)}")

    # Show all groups in danger zone (≤ 0.05)
    if gv:
        danger_groups = {name: val for name, val in gv.items() if val <= 0.05}
        if danger_groups:
            diag.append(f"Danger: {', '.join(f'{k}={v:.4f}' for k, v in sorted(danger_groups.items(), key=lambda x: x[1]))}")

    # Combine: "[min_group info] | [details]"
    if diag:
        return f"{prefix} | {' | '.join(diag)}"
    else:
        return prefix

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
    'core_active_raw': [], 'detail_active_raw': [],
    'core_effective_raw': [], 'detail_effective_raw': [],
    'detail_ratio_raw': [], 'core_var_raw': [], 'detail_var_raw': [],
    'core_var_max_raw': [], 'detail_var_max_raw': [], 'consistency_raw': [],
    'structure_loss': [], 'appearance_loss': [], 'color_hist_loss': [],
    'realism_recon_raw': [], 'realism_swap_raw': [],
    'core_color_leak_raw': [], 'detail_edge_leak_raw': [],
    'traversal_raw': [], 'traversal_core_effect_raw': [], 'traversal_detail_effect_raw': [],
    'detail_mean_raw': [], 'detail_var_mean_raw': [], 'detail_cov_raw': [],
    **{f'group_{n}': [] for n in GROUP_NAMES},
    'pixel': [], 'edge_goal': [], 'perceptual': [],
    'core_mse': [], 'core_edge': [],
    'swap_structure': [], 'swap_appearance': [], 'swap_color_hist': [],
    'realism_recon': [], 'realism_swap': [],
    'core_color_leak': [], 'detail_edge_leak': [], 'traversal_goal': [],
    'kl_core_goal': [], 'kl_detail_goal': [],
    'cov_goal': [], 'weak': [], 'consistency_goal': [],
    'core_active_goal': [], 'detail_active_goal': [],
    'core_effective_goal': [], 'detail_effective_goal': [],
    'detail_mean_goal': [], 'detail_var_mean_goal': [], 'detail_cov_goal': [],
    'detail_ratio_goal': [], 'core_var_goal': [], 'detail_var_goal': [],
    'core_var_max_goal': [], 'detail_var_max_goal': [],
    'core_active': [], 'detail_active': [], 'core_effective': [], 'detail_effective': [],
}
dim_variance_history = {'core': [], 'detail': []}

print("\n" + "=" * 100)
print(f"BOM VAE v17 - {data_info['name'].upper()} - {EPOCHS} EPOCHS")
print("v16: LBO Constitution compliance FIXES (epoch 13-14 collapse prevented)")
print("     - Directive #1: Pure -log(min(S_i)) - NO softmin, NO epsilon")
print("     - Directive #3: No clamping on goals")
print("     - Directive #4: Discrete rejection/rollback on S_min ≤ 0")
print(f"     - Directive #6 FIX: Adaptive squeeze 10%/8%/6% (was 5%/4%/3%) + S_min > 0.5 stability check")
print(f"     - Backoff at {ROLLBACK_THRESHOLD_MAX*100:.0f}% (was 50%), target {ROLLBACK_THRESHOLD_TARGET*100:.0f}% rollback rate")
print(f"     - Start tightening epoch {ADAPTIVE_TIGHTENING_START} (was 5), health bounds 2x wider")
print("     - Behavioral disentanglement (core→structure, detail→appearance)")
print("=" * 100 + "\n")

# LBO: Rollback mechanism for discrete rejection
# last_good_state = copy.deepcopy(model.state_dict())
# last_good_state_d = copy.deepcopy(discriminator.state_dict())
# last_good_optimizer = copy.deepcopy(optimizer.state_dict())
# last_good_optimizer_d = copy.deepcopy(optimizer_d.state_dict())

# Adaptive tightening termination: track when threshold is hit
threshold_hit_epoch = None
previous_goal_specs = None  # For rollback if tightening is too aggressive

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    epoch_data = {k: [] for k in histories.keys()}
    bn_counts = {n: 0 for n in GROUP_NAMES}
    skip_count = 0
    all_mu_core, all_mu_detail = [], []

    # Track consecutive rollbacks to avoid spam
    consecutive_rollbacks = 0
    first_rollback_info = None

    # LBO Directive #6: Natural adaptive squeeze - no manual recalibration
    # Only calibrate scales at epoch 1, then let LBO's infinite gradient do the work
    needs_recal = False
    if epoch == 1:
        goal_system.start_recalibration()
        needs_recal = True
        print(f"\n📊 Epoch 1: Initial calibration of all goal scales...")

    # Remove epoch 1 safety margin at start of epoch 2
    if epoch == 2 and goal_system.epoch1_margin_applied:
        goal_system.remove_epoch1_margin()

    # v17d: Apply KL squeeze schedule (starts epoch 3, after ceiling discovery in epoch 1)
    if epoch in KL_SQUEEZE_SCHEDULE and epoch >= 3:  # Start squeeze from epoch 3
        new_upper = KL_SQUEEZE_SCHEDULE[epoch]
        if new_upper is not None:
            # Update both kl_core and kl_detail upper bounds
            GOAL_SPECS['kl_core']['upper'] = new_upper
            GOAL_SPECS['kl_detail']['upper'] = new_upper
            # Re-initialize goal system normalizers with new bounds
            goal_system.goal_specs = GOAL_SPECS
            goal_system.rebuild_normalizers()
            print(f"🔽 KL ceiling squeezed to {new_upper:,} nats (epoch {epoch})")

    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch_idx, (x, _) in enumerate(pbar):
        x = x.to(DEVICE, non_blocking=True)
        if not check_tensor(x): skip_count += 1; continue
        
        with torch.no_grad():
            x_aug = apply_augmentation(x)

        # DIAGNOSTIC: Check batch health before forward pass
        if batch_idx in [199, 200] or (batch_idx > 200 and batch_idx % 50 == 0):
            with torch.no_grad():
                print(f"\n🔍 BATCH {batch_idx} DIAGNOSTICS:")
                print(f"   Input x: min={x.min():.4f}, max={x.max():.4f}, has_nan={torch.isnan(x).any()}, has_inf={torch.isinf(x).any()}")

                # Check model weights
                sample_param = next(model.parameters())
                print(f"   Model weights: has_nan={torch.isnan(sample_param).any()}, has_inf={torch.isinf(sample_param).any()}")

                # Check if any gradients are present (should be zero after zero_grad, but let's check)
                has_grads = any(p.grad is not None for p in model.parameters())
                if has_grads:
                    grad_sample = next(p.grad for p in model.parameters() if p.grad is not None)
                    print(f"   ⚠️  Gradients still present! has_nan={torch.isnan(grad_sample).any()}")

                # Check if encoder produces NaN
                test_mu, test_logvar = model.encode(x)
                print(f"   Encoder mu: min={test_mu.min():.4f}, max={test_mu.max():.4f}, has_nan={torch.isnan(test_mu).any()}")
                print(f"   Encoder logvar: min={test_logvar.min():.4f}, max={test_logvar.max():.4f}, has_nan={torch.isnan(test_logvar).any()}")

                if not torch.isnan(test_mu).any() and not torch.isnan(test_logvar).any():
                    # Encoder is fine, check reparameterize
                    test_z = model.reparameterize(test_mu, test_logvar)
                    print(f"   Reparameterize z: min={test_z.min():.4f}, max={test_z.max():.4f}, has_nan={torch.isnan(test_z).any()}")

                    if not torch.isnan(test_z).any():
                        # z is fine, check decoder
                        test_recon = model.decode(test_z)
                        print(f"   Decoder recon: min={test_recon.min():.4f}, max={test_recon.max():.4f}, has_nan={torch.isnan(test_recon).any()}")

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
            # Note: Removed BN reset - calibration stats should be fine for LBO training

        result = grouped_bom_loss(recon, x, mu, logvar, z, model, goal_system, vgg, split_idx, GROUP_NAMES, discriminator, x_aug)

        # v16: Debug output for raw and normalized values (once per epoch)
        if DEBUG_RAW_NORMALIZED and batch_idx == 0 and epoch >= 2 and goal_system.calibrated:
            print(f"\n🔍 DEBUG: Raw vs Normalized Values (Epoch {epoch}, Batch {batch_idx})")
            print("-" * 100)
            raw_vals = result['raw_values']
            norm_vals = result['individual_goals']
            # Only iterate over keys that exist in both dictionaries (actual goals, not diagnostics)
            for goal_name in sorted(norm_vals.keys()):
                if goal_name in raw_vals:
                    raw = raw_vals[goal_name]
                    norm = norm_vals[goal_name]
                    print(f"  {goal_name:20s} | Raw: {raw:10.6f} → Normalized: {norm:6.4f}")
            print("-" * 100 + "\n")

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

        # LBO Directive #4: Discrete Rejection / Rollback Mechanism
        # Save state before stepping - allows rollback if update violates constraints
        state_before_step = {
            'model': {k: v.clone() for k, v in model.state_dict().items()},
            'optimizer': copy.deepcopy(optimizer.state_dict())
        }

        # DIAGNOSTIC: Check model health BEFORE step
        with torch.no_grad():
            sample_param = next(model.parameters())
            if torch.isnan(sample_param).any() or torch.isinf(sample_param).any():
                print(f"    [PRE-STEP] Model weights already corrupted BEFORE optimizer.step()!")

        optimizer.step()

        # Look-ahead: Check if the update caused any group to violate constraints
        with torch.no_grad():
            recon_check, mu_check, logvar_check, z_check = model(x)
            check_result = grouped_bom_loss(recon_check, x, mu_check, logvar_check, z_check, model,
                                          goal_system, vgg, split_idx, GROUP_NAMES, discriminator,
                                          x_aug)

            # If update caused violation (S_min ≤ 0 or NaN), ROLLBACK
            if check_result is None or torch.isnan(check_result['loss']) or torch.isinf(check_result['loss']):
                # Restore state
                model.load_state_dict(state_before_step['model'])
                optimizer.load_state_dict(state_before_step['optimizer'])
                skip_count += 1
                optimizer.zero_grad(set_to_none=True)

                # DIAGNOSTIC: Verify restoration worked
                with torch.no_grad():
                    sample_param_after = next(model.parameters())
                    if torch.isnan(sample_param_after).any() or torch.isinf(sample_param_after).any():
                        print(f"    [POST-ROLLBACK] Model weights STILL corrupted after restore!")
                    # Also test forward pass after restore
                    try:
                        recon_test, mu_test, logvar_test, z_test = model(x)
                        if torch.isnan(recon_test).any() or torch.isnan(mu_test).any():
                            print(f"    [POST-ROLLBACK] Forward pass still produces NaN after restore!")
                    except:
                        print(f"    [POST-ROLLBACK] Forward pass crashed after restore!")

                # Track consecutive rollbacks to batch output
                consecutive_rollbacks += 1

                if consecutive_rollbacks == 1:
                    # First rollback - capture full diagnostic
                    if check_result is not None:
                        groups = check_result.get('groups', [])
                        raw_vals = check_result.get('raw_values', {})
                        min_group_val = groups.min().item() if len(groups) > 0 else 0.0

                        # Find which constraint failed
                        failed_goals = []
                        for name, val in check_result.get('individual_goals', {}).items():
                            if val <= 0:
                                raw_val = raw_vals.get(name + '_raw', raw_vals.get(name, 'N/A'))
                                failed_goals.append(f"{name}={val:.4f} (raw={raw_val})")

                        first_rollback_info = {
                            'batch': batch_idx,
                            'min_group': min_group_val,
                            'failed': ', '.join(failed_goals[:3]) if failed_goals else 'Unknown'  # Show first 3
                        }
                    else:
                        first_rollback_info = {
                            'batch': batch_idx,
                            'min_group': 0.0,
                            'failed': 'Loss returned None'
                        }

                    print(f"\n⚠️  [ROLLBACK] Epoch {epoch}, Batch {batch_idx}")
                    print(f"    S_min = {first_rollback_info['min_group']:.6f}")
                    print(f"    Failed: {first_rollback_info['failed']}")

                elif consecutive_rollbacks % 10 == 0:
                    # Every 10th consecutive rollback, show count
                    print(f"    ... {consecutive_rollbacks} consecutive rollbacks (since batch {first_rollback_info['batch']})")

                continue
            else:
                # Successful step - reset consecutive counter
                if consecutive_rollbacks > 0:
                    print(f"    ✓ Recovered after {consecutive_rollbacks} rollbacks (batch {batch_idx})\n")
                    consecutive_rollbacks = 0
                    first_rollback_info = None
        
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

            # Updated raw values including leak detection and logvar tracking
            for k in ['kl_core_raw', 'kl_detail_raw', 'logvar_core_raw', 'logvar_detail_raw',
                     'core_active_raw', 'detail_active_raw', 'core_effective_raw', 'detail_effective_raw',
                     'detail_ratio_raw', 'core_var_raw', 'detail_var_raw',
                     'core_var_max_raw', 'detail_var_max_raw', 'consistency_raw',
                     'detail_mean_raw', 'detail_var_mean_raw', 'detail_cov_raw',
                     'realism_recon_raw', 'realism_swap_raw',
                     'core_color_leak_raw', 'detail_edge_leak_raw',
                     'traversal_raw', 'traversal_core_effect_raw', 'traversal_detail_effect_raw']:
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
            epoch_data['traversal_goal'].append(ig['traversal'])
            epoch_data['kl_core_goal'].append(ig['kl_core'])
            epoch_data['kl_detail_goal'].append(ig['kl_detail'])
            epoch_data['cov_goal'].append(ig['cov'])
            epoch_data['weak'].append(ig['weak'])
            epoch_data['consistency_goal'].append(ig.get('consistency', 0.5))
            epoch_data['core_active_goal'].append(ig['core_active'])
            epoch_data['detail_active_goal'].append(ig['detail_active'])
            epoch_data['core_effective_goal'].append(ig['core_effective'])
            epoch_data['detail_effective_goal'].append(ig['detail_effective'])
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

    scheduler.step()
    scheduler_d.step()
    last_good_state = copy.deepcopy(model.state_dict())
    last_good_state_d = copy.deepcopy(discriminator.state_dict())
    last_good_optimizer = copy.deepcopy(optimizer.state_dict())
    last_good_optimizer_d = copy.deepcopy(optimizer_d.state_dict())

    # v17d: At end of epoch 1, discover max KL and set as ceiling for epoch 2+
    if epoch == 1 and goal_system.calibrated:
        max_kl_core = max(epoch_data['kl_core_raw']) if epoch_data['kl_core_raw'] else 0
        max_kl_detail = max(epoch_data['kl_detail_raw']) if epoch_data['kl_detail_raw'] else 0
        discovered_ceiling = max(max_kl_core, max_kl_detail)

        print(f"\n🔍 EPOCH 1 KL DISCOVERY:")
        print(f"   Max KL_core:   {max_kl_core:,.1f}")
        print(f"   Max KL_detail: {max_kl_detail:,.1f}")
        print(f"   Setting ceiling: {discovered_ceiling:,.1f}")

        # Update KL bounds for epoch 2+
        if discovered_ceiling > 0:  # Only set if we have valid data
            GOAL_SPECS['kl_core']['upper'] = discovered_ceiling
            GOAL_SPECS['kl_detail']['upper'] = discovered_ceiling
            goal_system.specs = GOAL_SPECS
            goal_system.rebuild_normalizers()  # Fixed: was initialize_normalizers()
            print(f"   ✓ KL ceiling will activate at start of epoch 2\n")
        else:
            print(f"   ⚠️  WARNING: No valid KL data (all rollbacks). Keeping unbounded for epoch 2.\n")

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

    # Adaptive tightening with progressive backoff
    rollback_rate = skip_count / total_batches if total_batches > 0 else 0

    # Check if last tightening was too aggressive (>15% rollbacks) - RESTORE previous constraints
    if epoch >= ADAPTIVE_TIGHTENING_START + 1 and rollback_rate > ROLLBACK_THRESHOLD_MAX and previous_goal_specs is not None:
        print(f"\n⚠️  Rollback rate too high ({rollback_rate*100:.0f}%), RESTORING previous constraints")
        # Restore the constraints from before tightening
        for name in GOAL_SPECS:
            GOAL_SPECS[name] = copy.deepcopy(previous_goal_specs[name])
        goal_system.specs = GOAL_SPECS
        goal_system.rebuild_normalizers()
        previous_goal_specs = None  # Clear backup

    # Decide if we should tighten this epoch
    # LBO Directive #6: Only tighten when "VAE stabilizes (S_min > 0.5)"
    # v17: Constant 5% squeeze every epoch (simplified from progressive rates)
    current_rate = ADAPTIVE_TIGHTENING_RATE

    # Check stability: average min_group over last STABILITY_WINDOW epochs
    from configs.config import MIN_GROUP_STABILITY_THRESHOLD, STABILITY_WINDOW
    recent_min_groups = histories['min_group'][-STABILITY_WINDOW:] if len(histories['min_group']) >= STABILITY_WINDOW else histories['min_group']
    avg_min_group = sum(recent_min_groups) / len(recent_min_groups) if recent_min_groups else 0.0
    is_stable = avg_min_group >= MIN_GROUP_STABILITY_THRESHOLD

    should_tighten = epoch >= ADAPTIVE_TIGHTENING_START and rollback_rate < ROLLBACK_THRESHOLD_TARGET and is_stable

    # Track when we first hit the target threshold
    if epoch >= ADAPTIVE_TIGHTENING_START and rollback_rate >= ROLLBACK_THRESHOLD_TARGET and threshold_hit_epoch is None:
        threshold_hit_epoch = epoch

    print(f"\nEpoch {epoch:2d} | Loss: {histories['loss'][-1]:.3f} | Min: {histories['min_group'][-1]:.3f} | SSIM: {histories['ssim'][-1]:.3f}")
    print(f"         Structure: {struct:.4f} | Appearance: {appear:.4f}")
    print(f"         KL_core: {kl_c:.1f} | KL_detail: {kl_d:.1f}")
    print(f"         Groups: " + " | ".join(f"{n}:{histories[f'group_{n}'][-1]:.2f}" for n in GROUP_NAMES))
    print(f"         Bottlenecks: " + " | ".join(f"{n}:{bn_pcts[n]:.1f}%" for n in GROUP_NAMES))
    print(f"         Rollbacks: {skip_count}/{total_batches} ({rollback_rate*100:.1f}%)", end="")

    if should_tighten:
        tightening_pct = int((1 - current_rate) * 100)
        print(f" → 🔧 TIGHTENING {tightening_pct}% (rollback={rollback_rate*100:.1f}% < {ROLLBACK_THRESHOLD_TARGET*100:.0f}%, stable={avg_min_group:.3f} > {MIN_GROUP_STABILITY_THRESHOLD})")

        # Save current constraints before tightening (for potential rollback)
        previous_goal_specs = {name: copy.deepcopy(spec) for name, spec in GOAL_SPECS.items()}

        # Tighten constraints progressively
        # MINIMIZE_SOFT: Full tightening (harder to satisfy)
        for name, spec in GOAL_SPECS.items():
            if spec['type'] == ConstraintType.MINIMIZE_SOFT and isinstance(spec.get('scale'), (int, float)):
                spec['scale'] *= current_rate

        # BOX: Gentler tightening (50% of MINIMIZE_SOFT rate) to avoid boundary violations
        box_rate = 1.0 - (1.0 - current_rate) * 0.5  # Half the tightening
        for name, spec in GOAL_SPECS.items():
            if spec['type'] in [ConstraintType.BOX, ConstraintType.BOX_ASYMMETRIC]:
                if 'lower' in spec and 'upper' in spec:
                    center = (spec['lower'] + spec['upper']) / 2
                    range_half = (spec['upper'] - spec['lower']) / 2
                    new_range_half = range_half * box_rate
                    spec['lower'] = center - new_range_half
                    spec['upper'] = center + new_range_half

        # Update goal_system with tightened specs (rebuild normalizers, keep scales)
        goal_system.specs = GOAL_SPECS
        goal_system.rebuild_normalizers()
    else:
        if epoch >= ADAPTIVE_TIGHTENING_START:
            if not is_stable:
                print(f" → ⏸️  Skipping tightening (unstable: avg_min={avg_min_group:.3f} < {MIN_GROUP_STABILITY_THRESHOLD})")
            else:
                print(f" → ⚠️  At limit ({rollback_rate*100:.1f}% >= {ROLLBACK_THRESHOLD_TARGET*100:.0f}%)")
        else:
            print()

    # Adaptive tightening termination: after hitting threshold, run 1 more epoch then stop
    if threshold_hit_epoch is not None and epoch > threshold_hit_epoch:
        print(f"\n🏁 STOPPING: Threshold hit at epoch {threshold_hit_epoch}, completed 1 additional epoch")
        break

# SAVE
torch.save({
    'model': model.state_dict(),
    'discriminator': discriminator.state_dict(),
    'histories': histories,
    'scales': goal_system.scales,
    'dim_var': dim_variance_history
}, f'{OUTPUT_DIR}/bom_vae_v17.pt')
print(f"\n✓ Saved to {OUTPUT_DIR}/bom_vae_v17.pt")

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
plot_group_balance(histories, GROUP_NAMES, f'{OUTPUT_DIR}/group_balance.png', f"BOM VAE v17 - {data_info['name']}")
plot_reconstructions(model, samples, split_idx, f'{OUTPUT_DIR}/reconstructions.png', DEVICE)
plot_traversals(model, samples, split_idx, f'{OUTPUT_DIR}/traversals_core.png', f'{OUTPUT_DIR}/traversals_detail.png', NUM_TRAVERSE_DIMS, DEVICE)
plot_cross_reconstruction(model, samples, split_idx, f'{OUTPUT_DIR}/cross_reconstruction.png', DEVICE)
plot_training_history(histories, f'{OUTPUT_DIR}/training_history.png')

print(f"\n✓ All saved to {OUTPUT_DIR}/")
print("=" * 60 + "\nCOMPLETE\n" + "=" * 60)
