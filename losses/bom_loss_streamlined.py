# losses/bom_loss_streamlined.py
# STREAMLINED VERSION - 9 goals instead of 35
# Priority: Useful latents > Perfect reconstruction

import torch
import torch.nn.functional as F
from configs import config_streamlined as config
from losses.goals import geometric_mean

# Import all helper functions from original bom_loss
from losses.bom_loss import (
    edges, mean_color, color_histogram, compute_ssim,
    check_tensor, check_latent_dict,
    mse_per_sample, mse_per_sample_1d, mse_per_sample_spatial,
    soft_active_count, block_diag_prior_kl
)

_LATENT_STRUCTURE_KEYS = ("core", "mid")
_LATENT_APPEARANCE_KEYS = ("detail", "resid")

def concat_latents(latents, keys):
    return torch.cat([latents[key] for key in keys], dim=1)

def structure_latents(latents):
    return concat_latents(latents, _LATENT_STRUCTURE_KEYS)

def appearance_latents(latents):
    return concat_latents(latents, _LATENT_APPEARANCE_KEYS)


def compute_raw_losses(recon, x, mu, logvar, z, model, vgg, discriminator=None, x_aug=None, tc_logits=None):
    """Compute raw losses for calibration - streamlined version."""
    B = x.shape[0]
    z_core = structure_latents(z)
    z_detail = appearance_latents(z)
    mu_core = structure_latents(mu)
    mu_detail = appearance_latents(mu)
    logvar_core = structure_latents(logvar)
    logvar_detail = appearance_latents(logvar)
    mu_all = torch.cat([mu_core, mu_detail], dim=1)
    logvar_all = torch.cat([logvar_core, logvar_detail], dim=1)

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)

    with torch.no_grad():
        x_feat = vgg(x)
    recon_feat = vgg(recon)
    edges_x = edges(x)

    losses = {}

    # 1. RECONSTRUCTION - Weighted merge: 0.2*pixel + 0.3*edge + 0.5*perceptual
    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val): ssim_val = torch.tensor(0.0, device=x.device)
    pixel_loss = (pixel_mse + 0.1 * (1.0 - ssim_val)).item()
    edge_loss = F.mse_loss(edges(recon), edges_x).item()
    perceptual_loss = F.mse_loss(recon_feat, x_feat).item()
    losses['reconstruction'] = (0.2 * pixel_loss + 0.3 * edge_loss + 0.5 * perceptual_loss)

    # 2. CROSS_RECON - Merge swap_structure + swap_appearance + swap_color_hist
    if B >= 4:
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]
        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = model.decode(z_sw)
        swap_struct = F.mse_loss(edges(r_sw), edges(x1)).item()
        swap_appear = F.mse_loss(mean_color(r_sw), mean_color(x2)).item()
        swap_hist = F.mse_loss(color_histogram(r_sw), color_histogram(x2)).item()
        losses['cross_recon'] = (swap_struct + swap_appear + swap_hist) / 3.0
    else:
        losses['cross_recon'] = 0.1
        r_sw = recon

    # 3. REALISM - Merge realism_recon + realism_swap
    if discriminator is not None:
        with torch.no_grad():
            d_recon = torch.sigmoid(discriminator(recon)).mean()
            d_swap = torch.sigmoid(discriminator(r_sw)).mean()
        losses['realism'] = ((1.0 - d_recon) + (1.0 - d_swap)).item() / 2.0
    else:
        losses['realism'] = 0.5

    # 4. KL_DIVERGENCE - Merge kl_core + kl_detail + prior_kl
    logvar_core_safe = torch.clamp(logvar_core, min=-30.0, max=20.0)
    kl_per_dim_core = -0.5 * (1 + logvar_core_safe - mu_core.pow(2) - logvar_core_safe.exp())
    kl_core_val = kl_per_dim_core.sum(dim=1).mean().item()

    logvar_detail_safe = torch.clamp(logvar_detail, min=-30.0, max=20.0)
    kl_per_dim_detail = -0.5 * (1 + logvar_detail_safe - mu_detail.pow(2) - logvar_detail_safe.exp())
    kl_detail_val = kl_per_dim_detail.sum(dim=1).mean().item()

    prior_kl_val = block_diag_prior_kl(
        mu_all, logvar_all, config.PRIOR_BLOCK_SIZE, config.PRIOR_INTRA_CORR
    ).mean().item()

    # Total KL (weighted average)
    losses['kl_divergence'] = (kl_core_val + kl_detail_val + prior_kl_val) / 3.0

    # 5. DISENTANGLEMENT - Merge sep_core + sep_mid + sep_detail (TC discriminators)
    if tc_logits is not None:
        sep_core = tc_logits.get('core', torch.zeros(B, device=x.device)).abs().mean().item()
        sep_mid = tc_logits.get('mid', torch.zeros(B, device=x.device)).abs().mean().item()
        sep_detail = tc_logits.get('detail', torch.zeros(B, device=x.device)).abs().mean().item()
        losses['disentanglement'] = (sep_core + sep_mid + sep_detail) / 3.0
    else:
        losses['disentanglement'] = 0.0

    # 6. BEHAVIORAL_SEPARATION - Merge core_color_leak + detail_edge_leak
    with torch.no_grad():
        noise_scale = 0.5
        z_core_pert = z_core + torch.randn_like(z_core) * noise_scale
        z_pert_core = torch.cat([z_core_pert, z_detail], dim=1)
        recon_pert_core = model.decode(z_pert_core)

        z_detail_pert = z_detail + torch.randn_like(z_detail) * noise_scale
        z_pert_detail = torch.cat([z_core, z_detail_pert], dim=1)
        recon_pert_detail = model.decode(z_pert_detail)

    core_color_leak = F.mse_loss(mean_color(recon_pert_core), mean_color(recon)).item()
    detail_edge_leak = F.mse_loss(edges(recon_pert_detail), edges(recon)).item()
    losses['behavioral_separation'] = (core_color_leak + detail_edge_leak) / 2.0

    # 7. CAPACITY - Merge core_active + detail_active + core_effective + detail_effective
    core_var_per_dim = mu_core.var(0)
    detail_var_per_dim = mu_detail.var(0)
    core_active_count = (core_var_per_dim > 0.1).float().sum()
    detail_active_count = (detail_var_per_dim > 0.1).float().sum()
    total_dims_core = float(mu_core.shape[1])
    total_dims_detail = float(mu_detail.shape[1])

    core_inactive_ratio = (total_dims_core - core_active_count) / total_dims_core
    detail_inactive_ratio = (total_dims_detail - detail_active_count) / total_dims_detail

    core_var_norm = core_var_per_dim / (core_var_per_dim.sum() + 1e-2) + 1e-2
    detail_var_norm = detail_var_per_dim / (detail_var_per_dim.sum() + 1e-2) + 1e-2
    core_var_norm_safe = torch.clamp(core_var_norm, min=1e-2, max=1.0)
    detail_var_norm_safe = torch.clamp(detail_var_norm, min=1e-2, max=1.0)
    core_effective = torch.exp(-torch.sum(core_var_norm_safe * torch.log(core_var_norm_safe)))
    detail_effective = torch.exp(-torch.sum(detail_var_norm_safe * torch.log(detail_var_norm_safe)))

    core_ineffective_ratio = (total_dims_core - core_effective) / total_dims_core
    detail_ineffective_ratio = (total_dims_detail - detail_effective) / total_dims_detail

    # Average of all 4 ratios
    losses['capacity'] = (
        core_inactive_ratio.item() + detail_inactive_ratio.item() +
        core_ineffective_ratio.item() + detail_ineffective_ratio.item()
    ) / 4.0

    # 8. LATENT_STATS - Merge logvar + cov + weak
    z_c = z_core - z_core.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-2)
    diag = torch.diag(cov) + 1e-2
    cov_penalty = (cov.pow(2).sum() - diag.pow(2).sum()) / torch.clamp(diag.pow(2).sum(), min=1e-1)
    weak_penalty = (mu_core.var(0) < 0.1).float().mean()

    logvar_penalty = (torch.abs(logvar_core.mean()) + torch.abs(logvar_detail.mean())) / 2.0

    losses['latent_stats'] = (cov_penalty.item() + weak_penalty.item() + logvar_penalty.item()) / 3.0

    # 9. CONSISTENCY - Keep as-is
    if x_aug is not None:
        with torch.no_grad():
            mu_aug, _ = model.encode(x_aug)
            mu_aug_core = structure_latents(mu_aug)
        losses['consistency'] = F.mse_loss(mu_core, mu_aug_core).item()
    else:
        losses['consistency'] = 0.01

    losses['_ssim'] = ssim_val.item()
    losses['_kl_core'] = kl_core_val
    losses['_kl_detail'] = kl_detail_val
    losses['_prior_kl'] = prior_kl_val

    return losses


def grouped_bom_loss_streamlined(recon, x, mu, logvar, z, model, goals, vgg, group_names, discriminator=None, x_aug=None, tc_logits=None):
    """
    Streamlined LBO-VAE Loss - 9 flat goals instead of 35.

    Pure LBO: loss = -log(min(all 9 goals))

    Goals (9 total - no grouping):
    1. kl_divergence
    2. disentanglement
    3. capacity
    4. behavioral_separation
    5. latent_stats
    6. reconstruction
    7. cross_recon
    8. realism
    9. consistency
    """

    # ========== INPUT VALIDATION ==========
    if not all([check_tensor(t) for t in [recon, x]]) or not all([check_latent_dict(t) for t in [mu, logvar, z]]):
        return None

    B = x.shape[0]
    z_core = structure_latents(z)
    z_detail = appearance_latents(z)
    mu_core = structure_latents(mu)
    mu_detail = appearance_latents(mu)
    logvar_core = structure_latents(logvar)
    logvar_detail = appearance_latents(logvar)
    mu_all = torch.cat([mu_core, mu_detail], dim=1)
    logvar_all = torch.cat([logvar_core, logvar_detail], dim=1)

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)
    if not check_tensor(recon_core):
        print(f"    [DECODER FAILURE] recon_core contains NaN/Inf")
        return None

    # ========== EXTRACT FEATURES ==========
    with torch.no_grad():
        x_feat = vgg(x)
    recon_feat = vgg(recon)
    if not all([check_tensor(t) for t in [x_feat, recon_feat]]):
        return None

    edges_x = edges(x)

    # ========== GOAL 1: RECONSTRUCTION (per-sample [B]) ==========
    # Weighted: 0.2*pixel + 0.3*edge + 0.5*perceptual
    pixel_mse_per_sample = mse_per_sample(recon, x)
    ssim_per_sample = compute_ssim(recon, x, per_sample=True)
    ssim_per_sample = torch.where(torch.isnan(ssim_per_sample), torch.zeros_like(ssim_per_sample), ssim_per_sample)

    pixel_loss = pixel_mse_per_sample + 0.1 * (1.0 - ssim_per_sample)
    edge_loss = mse_per_sample(edges(recon), edges_x)
    perceptual_loss = mse_per_sample_spatial(recon_feat, x_feat)

    reconstruction_loss = 0.2 * pixel_loss + 0.3 * edge_loss + 0.5 * perceptual_loss
    g_reconstruction = goals.goal(reconstruction_loss, 'reconstruction')  # [B]

    # ========== GOAL 2: CROSS_RECON (per-sample [B]) ==========
    if B >= 4:
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]
        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = model.decode(z_sw)

        swap_struct_loss = mse_per_sample(edges(r_sw), edges(x1))
        swap_appear_loss = mse_per_sample_1d(mean_color(r_sw), mean_color(x2))
        swap_hist_loss = mse_per_sample_1d(color_histogram(r_sw), color_histogram(x2))

        cross_recon_loss = (swap_struct_loss + swap_appear_loss + swap_hist_loss) / 3.0
        g_cross_recon = goals.goal(cross_recon_loss, 'cross_recon')  # [B]
    else:
        g_cross_recon = torch.full((B,), 0.5, device=x.device)
        cross_recon_loss = torch.zeros(B, device=x.device)
        swap_struct_loss = torch.zeros(B, device=x.device)
        swap_appear_loss = torch.zeros(B, device=x.device)
        swap_hist_loss = torch.zeros(B, device=x.device)
        r_sw = recon

    # ========== GOAL 3: REALISM (per-sample [B]) ==========
    if discriminator is not None:
        d_recon_logits = discriminator(recon)
        d_swap_logits = discriminator(r_sw)

        realism_loss_recon = 1.0 - torch.sigmoid(d_recon_logits).mean(dim=[1, 2, 3])
        realism_loss_swap = 1.0 - torch.sigmoid(d_swap_logits).mean(dim=[1, 2, 3])

        realism_loss = (realism_loss_recon + realism_loss_swap) / 2.0
        g_realism = goals.goal(realism_loss, 'realism')  # [B]
    else:
        g_realism = torch.full((B,), 0.5, device=x.device)
        realism_loss = torch.zeros(B, device=x.device)

    # ========== GOAL 4: KL_DIVERGENCE (per-sample [B]) ==========
    logvar_core_safe = torch.clamp(logvar_core, min=-30.0, max=20.0)
    kl_per_dim_core = -0.5 * (1 + logvar_core_safe - mu_core.pow(2) - logvar_core_safe.exp())
    kl_core_val = kl_per_dim_core.sum(dim=1)  # [B]

    logvar_detail_safe = torch.clamp(logvar_detail, min=-30.0, max=20.0)
    kl_per_dim_detail = -0.5 * (1 + logvar_detail_safe - mu_detail.pow(2) - logvar_detail_safe.exp())
    kl_detail_val = kl_per_dim_detail.sum(dim=1)  # [B]

    prior_kl_val = block_diag_prior_kl(
        mu_all, logvar_all, config.PRIOR_BLOCK_SIZE, config.PRIOR_INTRA_CORR
    )  # [B]

    # Merge: average of 3 KL components
    kl_divergence_val = (kl_core_val + kl_detail_val + prior_kl_val) / 3.0  # [B]
    g_kl_divergence = goals.goal(kl_divergence_val, 'kl_divergence')  # [B]

    # ========== GOAL 5: DISENTANGLEMENT (per-sample [B]) ==========
    if tc_logits is not None:
        sep_core = tc_logits.get('core', torch.zeros(B, device=x.device)).abs()
        sep_mid = tc_logits.get('mid', torch.zeros(B, device=x.device)).abs()
        sep_detail = tc_logits.get('detail', torch.zeros(B, device=x.device)).abs()
        disentangle_val = (sep_core + sep_mid + sep_detail) / 3.0  # [B]
    else:
        disentangle_val = torch.zeros(B, device=x.device)

    g_disentanglement = goals.goal(disentangle_val, 'disentanglement')  # [B]

    # ========== GOAL 6: BEHAVIORAL_SEPARATION (per-sample [B]) ==========
    noise_scale = 0.5
    z_core_pert = z_core + torch.randn_like(z_core) * noise_scale
    z_pert_core = torch.cat([z_core_pert, z_detail], dim=1)
    recon_pert_core = model.decode(z_pert_core)

    z_detail_pert = z_detail + torch.randn_like(z_detail) * noise_scale
    z_pert_detail = torch.cat([z_core, z_detail_pert], dim=1)
    recon_pert_detail = model.decode(z_pert_detail)

    core_color_leak = mse_per_sample_1d(mean_color(recon_pert_core), mean_color(recon))  # [B]
    detail_edge_leak = mse_per_sample(edges(recon_pert_detail), edges(recon))  # [B]

    behavioral_sep_val = (core_color_leak + detail_edge_leak) / 2.0  # [B]
    g_behavioral_separation = goals.goal(behavioral_sep_val, 'behavioral_separation')  # [B]

    # ========== GOAL 7: CAPACITY (batch-level, expand to [B]) ==========
    core_var_per_dim = mu_core.var(0)
    detail_var_per_dim = mu_detail.var(0)

    core_active_count = soft_active_count(core_var_per_dim, threshold=0.1, temperature=0.05)
    detail_active_count = soft_active_count(detail_var_per_dim, threshold=0.1, temperature=0.05)
    total_dims_core = float(mu_core.shape[1])
    total_dims_detail = float(mu_detail.shape[1])

    core_inactive_ratio = (total_dims_core - core_active_count) / total_dims_core
    detail_inactive_ratio = (total_dims_detail - detail_active_count) / total_dims_detail

    core_var_norm = core_var_per_dim / (core_var_per_dim.sum() + 1e-2) + 1e-2
    detail_var_norm = detail_var_per_dim / (detail_var_per_dim.sum() + 1e-2) + 1e-2
    core_var_norm_safe = torch.clamp(core_var_norm, min=1e-2, max=1.0)
    detail_var_norm_safe = torch.clamp(detail_var_norm, min=1e-2, max=1.0)
    core_effective = torch.exp(-torch.sum(core_var_norm_safe * torch.log(core_var_norm_safe)))
    detail_effective = torch.exp(-torch.sum(detail_var_norm_safe * torch.log(detail_var_norm_safe)))

    core_ineffective_ratio = (total_dims_core - core_effective) / total_dims_core
    detail_ineffective_ratio = (total_dims_detail - detail_effective) / total_dims_detail

    capacity_val = (core_inactive_ratio + detail_inactive_ratio + core_ineffective_ratio + detail_ineffective_ratio) / 4.0
    g_capacity = goals.goal(capacity_val, 'capacity')  # scalar
    if not isinstance(g_capacity, torch.Tensor) or g_capacity.dim() == 0:
        g_capacity = g_capacity * torch.ones(B, device=x.device)  # [B]

    # ========== GOAL 8: LATENT_STATS (batch-level, expand to [B]) ==========
    z_c = z_core - z_core.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-2)
    diag = torch.diag(cov) + 1e-2
    cov_penalty = (cov.pow(2).sum() - diag.pow(2).sum()) / torch.clamp(diag.pow(2).sum(), min=1e-1)

    weak_penalty = 1.0 - torch.sigmoid((mu_core.var(0) - 0.1) / 0.05).mean()

    logvar_penalty = (torch.abs(logvar_core.mean()) + torch.abs(logvar_detail.mean())) / 2.0

    latent_stats_val = (cov_penalty + weak_penalty + logvar_penalty) / 3.0
    g_latent_stats = goals.goal(latent_stats_val, 'latent_stats')  # scalar
    if not isinstance(g_latent_stats, torch.Tensor) or g_latent_stats.dim() == 0:
        g_latent_stats = g_latent_stats * torch.ones(B, device=x.device)  # [B]

    # ========== GOAL 9: CONSISTENCY (per-sample [B]) ==========
    if x_aug is not None:
        with torch.no_grad():
            mu_aug, _ = model.encode(x_aug)
        mu_aug_core = structure_latents(mu_aug)
        consistency_loss = mse_per_sample_1d(mu_core, mu_aug_core)  # [B]
        g_consistency = goals.goal(consistency_loss, 'consistency')  # [B]
    else:
        g_consistency = torch.full((B,), 0.5, device=x.device)
        consistency_loss = torch.zeros(B, device=x.device)

    # ========== FLAT STRUCTURE: STACK ALL 9 GOALS ==========
    # No grouping - pure LBO with 9 constraints
    # Stack as [B, 9]
    goals_tensor = torch.stack([
        g_kl_divergence,           # 0
        g_disentanglement,         # 1
        g_capacity,                # 2
        g_behavioral_separation,   # 3
        g_latent_stats,            # 4
        g_reconstruction,          # 5
        g_cross_recon,             # 6
        g_realism,                 # 7
        g_consistency,             # 8
    ], dim=1)  # [B, 9]

    # ========== CHECK FOR NaN/Inf ==========
    if torch.isnan(goals_tensor).any() or torch.isinf(goals_tensor).any():
        print(f"    [NaN/Inf DETECTED] in goals tensor")
        return None

    # ========== LBO LOSS (GLOBAL BOTTLENECK) ==========
    # Pure LBO: -log(min(all 9 goals across all samples))
    global_min = goals_tensor.min()
    if torch.isnan(global_min) or torch.isinf(global_min):
        print(f"    [LBO BARRIER] global_min is NaN/Inf")
        return None

    if global_min <= 0:
        min_per_sample, idx_per_sample = goals_tensor.min(dim=1)
        failed_mask = min_per_sample <= 0
        n_failed = failed_mask.sum().item()
        failed_indices = idx_per_sample[failed_mask]
        if len(failed_indices) > 0:
            most_common_failure = failed_indices.mode().values.item()
            goal_name = group_names[most_common_failure] if most_common_failure < len(group_names) else f"goal_{most_common_failure}"
        else:
            goal_name = "unknown"

        print(f"    [LBO BARRIER] {n_failed}/{B} samples failed, most common: '{goal_name}'")
        return None

    loss = -torch.log(global_min)
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"    [LOSS NaN/Inf] Found NaN/Inf in -log(global_min={global_min:.6f})")
        return None

    # ========== RETURN RESULTS ==========
    min_per_sample, idx_per_sample = goals_tensor.min(dim=1)
    min_goal_idx = idx_per_sample.mode().values.item()

    individual_goals = {
        # Streamlined 9 goals
        'kl_divergence': g_kl_divergence.mean().item(),
        'disentanglement': g_disentanglement.mean().item(),
        'capacity': g_capacity.mean().item(),
        'behavioral_separation': g_behavioral_separation.mean().item(),
        'latent_stats': g_latent_stats.mean().item(),
        'reconstruction': g_reconstruction.mean().item(),
        'cross_recon': g_cross_recon.mean().item(),
        'realism': g_realism.mean().item(),
        'consistency': g_consistency.mean().item(),
        # Backwards compatibility - stub values for old individual goals
        'pixel': 0.5, 'edge': 0.5, 'perceptual': 0.5,
        'core_mse': 0.5, 'core_edge': 0.5,
        'swap_structure': 0.5, 'swap_appearance': 0.5, 'swap_color_hist': 0.5,
        'realism_recon': 0.5, 'realism_swap': 0.5,
        'core_color_leak': 0.5, 'detail_edge_leak': 0.5,
        'traversal': 0.5,
        'sep_core': 0.5, 'sep_mid': 0.5, 'sep_detail': 0.5,
    }

    # Goal values (same as individual_goals, kept for backwards compatibility)
    group_values = {n: goals_tensor[:, i].mean().item() for i, n in enumerate(group_names)}

    raw_values = {
        'kl_core_raw': kl_core_val.mean().item(),
        'kl_detail_raw': kl_detail_val.mean().item(),
        'prior_kl_raw': prior_kl_val.mean().item(),
        'kl_total_raw': kl_divergence_val.mean().item(),
        'disentangle_raw': disentangle_val.mean().item(),
        'behavioral_sep_raw': behavioral_sep_val.mean().item(),
        'capacity_raw': capacity_val.item() if isinstance(capacity_val, torch.Tensor) else capacity_val,
        'latent_stats_raw': latent_stats_val.item() if isinstance(latent_stats_val, torch.Tensor) else latent_stats_val,
        'reconstruction_raw': reconstruction_loss.mean().item(),
        'cross_recon_raw': cross_recon_loss.mean().item(),
        'realism_raw': realism_loss.mean().item(),
        'consistency_raw': consistency_loss.mean().item(),
        # Swap components (for backwards compatibility with train.py)
        'structure_loss': swap_struct_loss.mean().item(),
        'appearance_loss': swap_appear_loss.mean().item(),
        'color_hist_loss': swap_hist_loss.mean().item(),
    }

    return {
        'loss': loss,
        'groups': goals_tensor,  # [B, 9] - all 9 goal scores per sample
        'min_idx': min_goal_idx,  # Index of most common bottleneck goal
        'group_values': group_values,  # Backwards compatible naming
        'individual_goals': individual_goals,
        'raw_values': raw_values,
        'ssim': ssim_per_sample.mean().item(),
        'mse': pixel_mse_per_sample.mean().item(),
        'edge_loss': edge_loss.mean().item(),
    }
