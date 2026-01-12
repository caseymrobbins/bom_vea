# losses/bom_loss.py
# CLEAN REWRITE - LBO VAE Loss Function
# All goals organized by group with clear structure

import torch
import torch.nn.functional as F
from losses.goals import geometric_mean

_sobel_x, _sobel_y = None, None

def _get_sobel(device):
    global _sobel_x, _sobel_y
    if _sobel_x is None or _sobel_x.device != device:
        _sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3).to(device)
        _sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3).to(device)
    return _sobel_x, _sobel_y

def edges(img):
    """Extract edge map - captures STRUCTURE."""
    sobel_x, sobel_y = _get_sobel(img.device)
    g = img.mean(1, keepdim=True)
    return (F.conv2d(g, sobel_x, padding=1).pow(2) + F.conv2d(g, sobel_y, padding=1).pow(2)).sqrt()

def mean_color(img):
    """Global mean color per channel - captures overall APPEARANCE."""
    return img.mean(dim=[2, 3])  # (B, 3)

def color_histogram(img, bins=16):
    """Color histogram - captures color distribution / APPEARANCE."""
    B, C, H, W = img.shape
    img_flat = img.view(B, C, -1)  # (B, 3, H*W)

    histograms = []
    for c in range(C):
        pixels = img_flat[:, c, :]  # (B, H*W)
        bin_centers = torch.linspace(0, 1, bins, device=img.device)
        diff = pixels.unsqueeze(-1) - bin_centers.view(1, 1, -1)
        weights = torch.exp(-diff.pow(2) / 0.01)
        hist = weights.sum(dim=1)
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        histograms.append(hist)

    return torch.cat(histograms, dim=1)

def compute_ssim(x, y, window_size=11):
    C1, C2 = 0.01**2, 0.03**2
    gauss = torch.exp(-torch.arange(window_size, device=x.device, dtype=torch.float32).sub(window_size//2).pow(2) / (2*1.5**2))
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

def check_tensor(t):
    return not (torch.isnan(t).any() or torch.isinf(t).any())

def compute_raw_losses(recon, x, mu, logvar, z, model, vgg, split_idx, discriminator=None, x_aug=None):
    """Compute all raw losses for calibration. Used to collect statistics before setting scales."""
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core, logvar_detail = logvar[:, :split_idx], logvar[:, split_idx:]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)

    with torch.no_grad():
        x_feat = vgg(x)
    recon_feat = vgg(recon)
    edges_x = edges(x)

    losses = {}

    # Reconstruction
    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val): ssim_val = torch.tensor(0.0, device=x.device)
    losses['pixel'] = (pixel_mse + 0.1 * (1.0 - ssim_val)).item()
    losses['edge'] = F.mse_loss(edges(recon), edges_x).item()
    losses['perceptual'] = F.mse_loss(recon_feat, x_feat).item()

    # Core
    losses['core_mse'] = F.mse_loss(recon_core, x).item()
    losses['core_edge'] = F.mse_loss(edges(recon_core), edges_x).item()

    # Swap
    if B >= 4:
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]
        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = model.decode(z_sw)
        losses['swap_structure'] = F.mse_loss(edges(r_sw), edges(x1)).item()
        losses['swap_appearance'] = F.mse_loss(mean_color(r_sw), mean_color(x2)).item()
        losses['swap_color_hist'] = F.mse_loss(color_histogram(r_sw), color_histogram(x2)).item()
    else:
        losses['swap_structure'] = 0.1
        losses['swap_appearance'] = 0.01
        losses['swap_color_hist'] = 0.01
        r_sw = recon

    # Realism
    if discriminator is not None:
        with torch.no_grad():
            d_recon = torch.sigmoid(discriminator(recon)).mean()
            d_swap = torch.sigmoid(discriminator(r_sw)).mean()
        losses['realism_recon'] = (1.0 - d_recon).item()
        losses['realism_swap'] = (1.0 - d_swap).item()
    else:
        losses['realism_recon'] = 0.5
        losses['realism_swap'] = 0.5

    # KL
    logvar_core_safe = torch.clamp(logvar_core, min=-30.0, max=20.0)
    kl_per_dim_core = -0.5 * (1 + logvar_core_safe - mu_core.pow(2) - logvar_core_safe.exp())
    losses['kl_core'] = kl_per_dim_core.sum(dim=1).mean().item()

    logvar_detail_safe = torch.clamp(logvar_detail, min=-30.0, max=20.0)
    kl_per_dim_detail = -0.5 * (1 + logvar_detail_safe - mu_detail.pow(2) - logvar_detail_safe.exp())
    losses['kl_detail'] = kl_per_dim_detail.sum(dim=1).mean().item()

    losses['logvar_core'] = logvar_core.mean().item()
    losses['logvar_detail'] = logvar_detail.mean().item()

    # Structure
    z_c = z_core - z_core.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    losses['cov'] = ((cov.pow(2).sum() - diag.pow(2).sum()) / diag.pow(2).sum()).item()
    losses['weak'] = (mu_core.var(0) < 0.1).float().mean().item()

    if x_aug is not None:
        with torch.no_grad():
            mu_aug, _ = model.encode(x_aug)
            mu_aug_core = mu_aug[:, :split_idx]
        losses['core_consistency'] = F.mse_loss(mu_core, mu_aug_core).item()
    else:
        losses['core_consistency'] = 0.01

    # Capacity
    core_var_per_dim = mu_core.var(0)
    detail_var_per_dim = mu_detail.var(0)
    core_active_count = (core_var_per_dim > 0.1).float().sum()
    detail_active_count = (detail_var_per_dim > 0.1).float().sum()
    total_dims = float(mu_core.shape[1])

    core_inactive_ratio = (total_dims - core_active_count) / total_dims
    detail_inactive_ratio = (total_dims - detail_active_count) / total_dims
    losses['core_active'] = core_inactive_ratio.item()
    losses['detail_active'] = detail_inactive_ratio.item()

    core_var_norm = core_var_per_dim / (core_var_per_dim.sum() + 1e-8) + 1e-8
    detail_var_norm = detail_var_per_dim / (detail_var_per_dim.sum() + 1e-8) + 1e-8
    core_effective = torch.exp(-torch.sum(core_var_norm * torch.log(core_var_norm)))
    detail_effective = torch.exp(-torch.sum(detail_var_norm * torch.log(detail_var_norm)))

    core_ineffective_ratio = (total_dims - core_effective) / total_dims
    detail_ineffective_ratio = (total_dims - detail_effective) / total_dims
    losses['core_effective'] = core_ineffective_ratio.item()
    losses['detail_effective'] = detail_ineffective_ratio.item()

    # Detail stats
    losses['detail_mean'] = mu_detail.mean(0).abs().mean().item()
    losses['detail_var_mean'] = mu_detail.var(0).mean().item()

    z_d = z_detail - z_detail.mean(0, keepdim=True)
    cov_d = (z_d.T @ z_d) / (B - 1 + 1e-8)
    diag_d = torch.diag(cov_d) + 1e-8
    losses['detail_cov'] = ((cov_d.pow(2).sum() - diag_d.pow(2).sum()) / diag_d.pow(2).sum()).item()

    # Disentangle
    with torch.no_grad():
        noise_scale = 0.5
        z_core_pert = z_core + torch.randn_like(z_core) * noise_scale
        z_pert_core = torch.cat([z_core_pert, z_detail], dim=1)
        recon_pert_core = model.decode(z_pert_core)

        z_detail_pert = z_detail + torch.randn_like(z_detail) * noise_scale
        z_pert_detail = torch.cat([z_core, z_detail_pert], dim=1)
        recon_pert_detail = model.decode(z_pert_detail)

    core_color_leak = F.mse_loss(mean_color(recon_pert_core), mean_color(recon))
    detail_edge_leak = F.mse_loss(edges(recon_pert_detail), edges(recon))
    losses['core_color_leak'] = core_color_leak.item()
    losses['detail_edge_leak'] = detail_edge_leak.item()

    core_edge_shift = F.mse_loss(edges(recon_pert_core), edges(recon))
    detail_color_shift = F.mse_loss(mean_color(recon_pert_detail), mean_color(recon))
    traversal_loss = 0.5 * (
        1.0 / (core_edge_shift + 1e-4) +
        1.0 / (detail_color_shift + 1e-4)
    )
    losses['traversal'] = traversal_loss.item()
    losses['traversal_core_effect'] = core_edge_shift.item()
    losses['traversal_detail_effect'] = detail_color_shift.item()

    # Health
    detail_contrib = (recon - recon_core).abs().mean()
    losses['detail_ratio'] = (detail_contrib / (recon_core.abs().mean() + 1e-8)).item()
    losses['core_var_health'] = mu_core.var(0).median().item()
    losses['detail_var_health'] = mu_detail.var(0).median().item()

    losses['_ssim'] = ssim_val.item()

    return losses


def grouped_bom_loss(recon, x, mu, logvar, z, model, goals, vgg, split_idx, group_names, discriminator=None, x_aug=None):
    """
    LBO VAE Loss Function - Clean Implementation

    Computes loss as: -log(min(groups)) where each group is geometric mean of goals

    LBO Directives:
    1. Pure min() barrier - NO softmin, NO epsilon on loss
    2. Pure geometric mean - NO epsilon on aggregation
    3. NO clamping on goals or groups
    4. Discrete rollback if any group ≤ 0
    """

    # ========== INPUT VALIDATION ==========
    if not all([check_tensor(t) for t in [recon, x, mu, logvar, z]]):
        bad_tensors = []
        for name, t in [('recon', recon), ('x', x), ('mu', mu), ('logvar', logvar), ('z', z)]:
            if not check_tensor(t):
                bad_tensors.append(name)
        print(f"    [INPUT TENSOR FAILURE] Bad tensors: {', '.join(bad_tensors)}")
        return None

    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core, logvar_detail = logvar[:, :split_idx], logvar[:, split_idx:]

    # ========== DECODE CORE-ONLY ==========
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
        bad_feats = []
        if not check_tensor(x_feat): bad_feats.append('x_feat')
        if not check_tensor(recon_feat): bad_feats.append('recon_feat')
        print(f"    [VGG FEATURE FAILURE] Bad features: {', '.join(bad_feats)}")
        return None

    edges_x = edges(x)

    # ========== GROUP A: RECONSTRUCTION ==========
    # Full image quality metrics
    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val): ssim_val = torch.tensor(0.0, device=x.device)

    g_pixel = goals.goal(pixel_mse + 0.1 * (1.0 - ssim_val), 'pixel')
    g_edge = goals.goal(F.mse_loss(edges(recon), edges_x), 'edge')
    g_perceptual = goals.goal(F.mse_loss(recon_feat, x_feat), 'perceptual')

    # ========== GROUP B: CORE ==========
    # Core (structure channel) should preserve structure
    g_core_mse = goals.goal(F.mse_loss(recon_core, x), 'core_mse')
    g_core_edge = goals.goal(F.mse_loss(edges(recon_core), edges_x), 'core_edge')

    # ========== GROUP C: SWAP ==========
    # Structure from x1, appearance from x2
    if B >= 4:
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]
        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = model.decode(z_sw)

        structure_loss = F.mse_loss(edges(r_sw), edges(x1))
        appearance_loss = F.mse_loss(mean_color(r_sw), mean_color(x2))
        color_hist_loss = F.mse_loss(color_histogram(r_sw), color_histogram(x2))

        g_swap_structure = goals.goal(structure_loss, 'swap_structure')
        g_swap_appearance = goals.goal(appearance_loss, 'swap_appearance')
        g_swap_color_hist = goals.goal(color_hist_loss, 'swap_color_hist')
    else:
        # Batch too small for swapping
        g_swap_structure = torch.tensor(0.5, device=x.device)
        g_swap_appearance = torch.tensor(0.5, device=x.device)
        g_swap_color_hist = torch.tensor(0.5, device=x.device)
        structure_loss = appearance_loss = color_hist_loss = torch.tensor(0.0, device=x.device)
        r_sw = recon

    # ========== GROUP D: REALISM ==========
    # Discriminator should classify reconstructions as real
    if discriminator is not None:
        d_recon_logits = discriminator(recon)
        d_swap_logits = discriminator(r_sw)

        # Want D scores HIGH (realistic), so minimize (1 - sigmoid(D))
        realism_loss_recon = 1.0 - torch.sigmoid(d_recon_logits).mean()
        realism_loss_swap = 1.0 - torch.sigmoid(d_swap_logits).mean()

        g_realism_recon = goals.goal(realism_loss_recon, 'realism_recon')
        g_realism_swap = goals.goal(realism_loss_swap, 'realism_swap')
    else:
        g_realism_recon = torch.tensor(0.5, device=x.device)
        g_realism_swap = torch.tensor(0.5, device=x.device)
        realism_loss_recon = realism_loss_swap = torch.tensor(0.0, device=x.device)

    # ========== GROUP E: DISENTANGLE ==========
    # Behavioral walls: core affects structure, detail affects appearance
    noise_scale = 0.5

    # Core perturbation: should change structure, NOT colors
    z_core_pert = z_core + torch.randn_like(z_core) * noise_scale
    z_pert_core = torch.cat([z_core_pert, z_detail], dim=1)
    recon_pert_core = model.decode(z_pert_core)

    # Detail perturbation: should change colors, NOT structure
    z_detail_pert = z_detail + torch.randn_like(z_detail) * noise_scale
    z_pert_detail = torch.cat([z_core, z_detail_pert], dim=1)
    recon_pert_detail = model.decode(z_pert_detail)

    # Measure leaks (should be SMALL)
    core_color_leak = F.mse_loss(mean_color(recon_pert_core), mean_color(recon))
    detail_edge_leak = F.mse_loss(edges(recon_pert_detail), edges(recon))

    g_core_color_leak = goals.goal(core_color_leak, 'core_color_leak')
    g_detail_edge_leak = goals.goal(detail_edge_leak, 'detail_edge_leak')

    # Measure intended effects (should be LARGE) - used for traversal metric
    core_edge_shift = F.mse_loss(edges(recon_pert_core), edges(recon))
    detail_color_shift = F.mse_loss(mean_color(recon_pert_detail), mean_color(recon))

    # Traversal: reward large intended shifts (1/shift → minimize shift^-1)
    traversal_loss = 0.5 * (
        1.0 / (core_edge_shift + 1e-2) +
        1.0 / (detail_color_shift + 1e-2)
    )
    g_traversal = goals.goal(traversal_loss, 'traversal')

    # ========== GROUP F: LATENT (Hierarchical) ==========

    # SUB-GROUP F1: KL (Distribution Matching)
    logvar_core_safe = torch.clamp(logvar_core, min=-30.0, max=20.0)
    kl_per_dim_core = -0.5 * (1 + logvar_core_safe - mu_core.pow(2) - logvar_core_safe.exp())
    kl_core_val = kl_per_dim_core.sum(dim=1).mean()
    g_kl_core = goals.goal(kl_core_val, 'kl_core')

    logvar_detail_safe = torch.clamp(logvar_detail, min=-30.0, max=20.0)
    kl_per_dim_detail = -0.5 * (1 + logvar_detail_safe - mu_detail.pow(2) - logvar_detail_safe.exp())
    kl_detail_val = kl_per_dim_detail.sum(dim=1).mean()
    g_kl_detail = goals.goal(kl_detail_val, 'kl_detail')

    # Direct logvar constraints
    logvar_core_mean = logvar_core.mean()
    g_logvar_core = goals.goal(logvar_core_mean, 'logvar_core')
    logvar_detail_mean = logvar_detail.mean()
    g_logvar_detail = goals.goal(logvar_detail_mean, 'logvar_detail')

    group_kl = geometric_mean([g_kl_core, g_kl_detail, g_logvar_core, g_logvar_detail])

    # SUB-GROUP F2: STRUCTURE (Independence & Consistency)
    z_c = z_core - z_core.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-2)
    diag = torch.diag(cov) + 1e-2
    cov_penalty = (cov.pow(2).sum() - diag.pow(2).sum()) / torch.clamp(diag.pow(2).sum(), min=1e-1)
    g_cov = goals.goal(cov_penalty, 'cov')

    g_weak = goals.goal((mu_core.var(0) < 0.1).float().mean(), 'weak')

    if x_aug is not None:
        mu_aug, _ = model.encode(x_aug)
        mu_aug_core = mu_aug[:, :split_idx]
        consistency_loss = F.mse_loss(mu_core, mu_aug_core)
        g_consistency = goals.goal(consistency_loss, 'core_consistency')
    else:
        g_consistency = torch.tensor(0.5, device=x.device)
        consistency_loss = torch.tensor(0.0, device=x.device)

    group_structure = geometric_mean([g_cov, g_weak, g_consistency])

    # SUB-GROUP F3: CAPACITY (Dimension Utilization)
    core_var_per_dim = mu_core.var(0)
    detail_var_per_dim = mu_detail.var(0)

    core_active_count = (core_var_per_dim > 0.1).float().sum()
    detail_active_count = (detail_var_per_dim > 0.1).float().sum()
    total_dims = float(mu_core.shape[1])

    core_inactive_ratio = (total_dims - core_active_count) / total_dims
    detail_inactive_ratio = (total_dims - detail_active_count) / total_dims
    g_core_active = goals.goal(core_inactive_ratio, 'core_active')
    g_detail_active = goals.goal(detail_inactive_ratio, 'detail_active')

    # Effective dimensions (exponential of entropy)
    core_var_norm = core_var_per_dim / (core_var_per_dim.sum() + 1e-2) + 1e-2
    detail_var_norm = detail_var_per_dim / (detail_var_per_dim.sum() + 1e-2) + 1e-2
    core_var_norm_safe = torch.clamp(core_var_norm, min=1e-2, max=1.0)
    detail_var_norm_safe = torch.clamp(detail_var_norm, min=1e-2, max=1.0)
    core_effective = torch.exp(-torch.sum(core_var_norm_safe * torch.log(core_var_norm_safe)))
    detail_effective = torch.exp(-torch.sum(detail_var_norm_safe * torch.log(detail_var_norm_safe)))

    core_ineffective_ratio = (total_dims - core_effective) / total_dims
    detail_ineffective_ratio = (total_dims - detail_effective) / total_dims
    g_core_effective = goals.goal(core_ineffective_ratio, 'core_effective')
    g_detail_effective = goals.goal(detail_ineffective_ratio, 'detail_effective')

    group_capacity = geometric_mean([g_core_active, g_detail_active, g_core_effective, g_detail_effective])

    # SUB-GROUP F4: DETAIL STATS
    detail_mean_val = mu_detail.mean(0).abs().mean()
    g_detail_mean = goals.goal(detail_mean_val, 'detail_mean')

    detail_var_mean_val = mu_detail.var(0).mean()
    g_detail_var_mean = goals.goal(detail_var_mean_val, 'detail_var_mean')

    z_d = z_detail - z_detail.mean(0, keepdim=True)
    cov_d = (z_d.T @ z_d) / (B - 1 + 1e-2)
    diag_d = torch.diag(cov_d) + 1e-2
    detail_cov_penalty = (cov_d.pow(2).sum() - diag_d.pow(2).sum()) / torch.clamp(diag_d.pow(2).sum(), min=1e-1)
    g_detail_cov = goals.goal(detail_cov_penalty, 'detail_cov')

    group_detail_stats = geometric_mean([g_detail_mean, g_detail_var_mean, g_detail_cov, g_traversal])

    # ========== GROUP G: HEALTH ==========
    # Variance statistics
    detail_contrib = (recon - recon_core).abs().mean()
    detail_ratio = detail_contrib / torch.clamp(recon_core.abs().mean(), min=1e-2)
    g_detail_ratio = goals.goal(detail_ratio, 'detail_ratio')

    core_var_median = mu_core.var(0).median()
    detail_var_median = mu_detail.var(0).median()
    g_core_var = goals.goal(core_var_median, 'core_var_health')
    g_detail_var = goals.goal(detail_var_median, 'detail_var_health')

    # ========== AGGREGATE GROUPS ==========
    group_recon = geometric_mean([g_pixel, g_edge, g_perceptual])
    group_core = geometric_mean([g_core_mse, g_core_edge])
    group_swap = geometric_mean([g_swap_structure, g_swap_appearance, g_swap_color_hist])
    group_realism = geometric_mean([g_realism_recon, g_realism_swap])
    group_disentangle = geometric_mean([g_core_color_leak, g_detail_edge_leak])
    group_latent = geometric_mean([group_kl, group_structure, group_capacity, group_detail_stats])
    group_health = geometric_mean([g_detail_ratio, g_core_var, g_detail_var])

    groups = torch.stack([group_recon, group_core, group_swap, group_realism, group_disentangle, group_latent, group_health])

    # ========== CHECK FOR NaN/Inf ==========
    if torch.isnan(groups).any() or torch.isinf(groups).any():
        group_status = []
        for i, (name, g) in enumerate(zip(group_names, groups)):
            if torch.isnan(g):
                group_status.append(f"{name}=NaN")
            elif torch.isinf(g):
                group_status.append(f"{name}=Inf")
        print(f"    [NaN/Inf DETECTED] {', '.join(group_status)}")
        return None

    # ========== LBO LOSS ==========
    # Directive #1: Pure min() barrier (NO softmin)
    min_group = groups.min()
    min_group_idx = groups.argmin()

    # Directive #4: Discrete rollback if S_min ≤ 0
    if min_group <= 0:
        group_name = group_names[min_group_idx] if min_group_idx < len(group_names) else f"group_{min_group_idx}"

        # DETAILED DIAGNOSTICS: Show sub-group/goal breakdown for hierarchical groups
        if group_name == 'latent':
            latent_subgroups = {
                'kl': group_kl.item(),
                'structure': group_structure.item(),
                'capacity': group_capacity.item(),
                'detail_stats': group_detail_stats.item()
            }
            failed_subgroups = [name for name, val in latent_subgroups.items() if val <= 0]

            print(f"    [LBO BARRIER] Group 'latent' failed: S_min = {min_group:.6f}")
            print(f"    └─ Sub-groups: kl={group_kl.item():.6f}, structure={group_structure.item():.6f}, capacity={group_capacity.item():.6f}, detail_stats={group_detail_stats.item():.6f}")

            if failed_subgroups:
                print(f"    └─ Failed sub-groups: {', '.join(failed_subgroups)}")

                # Show individual goals in failed sub-groups
                if 'kl' in failed_subgroups:
                    print(f"       KL goals: kl_core={g_kl_core.item():.6f}, kl_detail={g_kl_detail.item():.6f}, logvar_core={g_logvar_core.item():.6f}, logvar_detail={g_logvar_detail.item():.6f}")
                if 'structure' in failed_subgroups:
                    print(f"       Structure goals: cov={g_cov.item():.6f}, weak={g_weak.item():.6f}, consistency={g_consistency.item():.6f}")
                if 'capacity' in failed_subgroups:
                    print(f"       Capacity goals: core_active={g_core_active.item():.6f}, detail_active={g_detail_active.item():.6f}, core_effective={g_core_effective.item():.6f}, detail_effective={g_detail_effective.item():.6f}")
                if 'detail_stats' in failed_subgroups:
                    print(f"       Detail stats goals: detail_mean={g_detail_mean.item():.6f}, detail_var_mean={g_detail_var_mean.item():.6f}, detail_cov={g_detail_cov.item():.6f}, traversal={g_traversal.item():.6f}")

        elif group_name == 'health':
            health_goals = {
                'detail_ratio': g_detail_ratio.item(),
                'core_var': g_core_var.item(),
                'detail_var': g_detail_var.item(),
            }
            failed_goals = [name for name, val in health_goals.items() if val <= 0]

            print(f"    [LBO BARRIER] Group 'health' failed: S_min = {min_group:.6f}")
            print(f"    └─ Goals: detail_ratio={g_detail_ratio.item():.6f}, core_var={g_core_var.item():.6f}, detail_var={g_detail_var.item():.6f}")

            if failed_goals:
                print(f"    └─ Failed goals: {', '.join(failed_goals)}")
        else:
            print(f"    [LBO BARRIER] Group '{group_name}' failed: S_min = {min_group:.6f}")

        return None

    # Directive #1: Pure log barrier (NO epsilon)
    loss = -torch.log(min_group)
    if torch.isnan(loss):
        print(f"    [LOSS NaN] -log({min_group:.6f}) = NaN")
        return None

    # ========== RETURN RESULTS ==========
    individual_goals = {
        'pixel': g_pixel.item(), 'edge': g_edge.item(), 'perceptual': g_perceptual.item(),
        'core_mse': g_core_mse.item(), 'core_edge': g_core_edge.item(),
        'swap_structure': g_swap_structure.item() if isinstance(g_swap_structure, torch.Tensor) else g_swap_structure,
        'swap_appearance': g_swap_appearance.item() if isinstance(g_swap_appearance, torch.Tensor) else g_swap_appearance,
        'swap_color_hist': g_swap_color_hist.item() if isinstance(g_swap_color_hist, torch.Tensor) else g_swap_color_hist,
        'realism_recon': g_realism_recon.item() if isinstance(g_realism_recon, torch.Tensor) else g_realism_recon,
        'realism_swap': g_realism_swap.item() if isinstance(g_realism_swap, torch.Tensor) else g_realism_swap,
        'core_color_leak': g_core_color_leak.item(), 'detail_edge_leak': g_detail_edge_leak.item(),
        'traversal': g_traversal.item(),
        'kl_core': g_kl_core.item(), 'kl_detail': g_kl_detail.item(),
        'logvar_core': g_logvar_core.item(), 'logvar_detail': g_logvar_detail.item(),
        'cov': g_cov.item(), 'weak': g_weak.item(),
        'consistency': g_consistency.item() if isinstance(g_consistency, torch.Tensor) else g_consistency,
        'core_active': g_core_active.item(), 'detail_active': g_detail_active.item(),
        'core_effective': g_core_effective.item(), 'detail_effective': g_detail_effective.item(),
        'detail_mean': g_detail_mean.item(), 'detail_var_mean': g_detail_var_mean.item(), 'detail_cov': g_detail_cov.item(),
        'detail_ratio': g_detail_ratio.item(),
        'core_var': g_core_var.item(), 'detail_var': g_detail_var.item(),
    }

    group_values = {n: g.item() for n, g in zip(group_names, groups)}

    raw_values = {
        'kl_core_raw': kl_core_val.item(), 'kl_detail_raw': kl_detail_val.item(),
        'logvar_core_raw': logvar_core_mean.item(), 'logvar_detail_raw': logvar_detail_mean.item(),
        'core_active_raw': core_active_count.item(), 'detail_active_raw': detail_active_count.item(),
        'core_effective_raw': core_effective.item(), 'detail_effective_raw': detail_effective.item(),
        'detail_ratio_raw': detail_ratio.item(),
        'core_var_raw': core_var_median.item(), 'detail_var_raw': detail_var_median.item(),
        'structure_loss': structure_loss.item() if isinstance(structure_loss, torch.Tensor) else structure_loss,
        'appearance_loss': appearance_loss.item() if isinstance(appearance_loss, torch.Tensor) else appearance_loss,
        'color_hist_loss': color_hist_loss.item() if isinstance(color_hist_loss, torch.Tensor) else color_hist_loss,
        'consistency_raw': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
        'realism_recon_raw': realism_loss_recon.item() if isinstance(realism_loss_recon, torch.Tensor) else realism_loss_recon,
        'realism_swap_raw': realism_loss_swap.item() if isinstance(realism_loss_swap, torch.Tensor) else realism_loss_swap,
        'core_color_leak_raw': core_color_leak.item(), 'detail_edge_leak_raw': detail_edge_leak.item(),
        'traversal_raw': traversal_loss.item(),
        'traversal_core_effect_raw': core_edge_shift.item(),
        'traversal_detail_effect_raw': detail_color_shift.item(),
        'detail_mean_raw': detail_mean_val.item(), 'detail_var_mean_raw': detail_var_mean_val.item(),
        'detail_cov_raw': detail_cov_penalty.item(),
    }

    return {
        'loss': loss, 'groups': groups, 'min_idx': min_group_idx,
        'group_values': group_values, 'individual_goals': individual_goals,
        'raw_values': raw_values, 'ssim': ssim_val.item(),
        'mse': pixel_mse.item(), 'edge_loss': F.mse_loss(edges(recon), edges_x).item(),
    }
