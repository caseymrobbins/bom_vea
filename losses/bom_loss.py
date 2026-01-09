# losses/bom_loss.py
# v15: Tightened constraints + Softmin A/B test
# Based on v14: Discriminator + Detail contracts
# r_sw should have: x1's STRUCTURE + x2's APPEARANCE

import torch
import torch.nn.functional as F
from losses.goals import geometric_mean

def softmin(x, temperature=0.1):
    """Smooth approximation of min using LogSumExp trick.

    softmin(x, T) = -T * log(sum(exp(-x / T)))

    As T → 0, this approaches hard min(x).
    As T → ∞, this approaches mean(x).
    """
    return -temperature * torch.logsumexp(-x / temperature, dim=0)

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
    
    # Compute histogram per channel
    histograms = []
    for c in range(C):
        # Soft histogram using gaussian kernels
        pixels = img_flat[:, c, :]  # (B, H*W)
        bin_centers = torch.linspace(0, 1, bins, device=img.device)  # (bins,)
        
        # Distance from each pixel to each bin center
        diff = pixels.unsqueeze(-1) - bin_centers.view(1, 1, -1)  # (B, H*W, bins)
        weights = torch.exp(-diff.pow(2) / 0.01)  # soft assignment
        hist = weights.sum(dim=1)  # (B, bins)
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)  # normalize
        histograms.append(hist)
    
    return torch.cat(histograms, dim=1)  # (B, 3*bins)

def spatial_color_map(img, grid_size=4):
    """Spatial color - captures local appearance / lighting."""
    # Downsample to grid_size x grid_size and flatten
    downsampled = F.adaptive_avg_pool2d(img, grid_size)  # (B, 3, grid, grid)
    return downsampled.view(img.shape[0], -1)  # (B, 3*grid*grid)

def luminance_map(img):
    """Luminance/brightness map - captures lighting pattern."""
    # Convert to grayscale luminance
    lum = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    # Downsample to capture broad lighting
    return F.adaptive_avg_pool2d(lum, 8)  # (B, 1, 8, 8)

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
    """Compute all raw losses for calibration. v15: Includes discriminator and detail contracts from v14."""
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core, logvar_detail = logvar[:, :split_idx], logvar[:, split_idx:]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)  # No clamp - decoder sigmoid handles mapping

    with torch.no_grad():
        x_feat = vgg(x)
    recon_feat = vgg(recon)
    edges_x = edges(x)

    losses = {}

    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val): ssim_val = torch.tensor(0.0, device=x.device)
    losses['pixel'] = (pixel_mse + 0.1 * (1.0 - ssim_val)).item()
    losses['edge'] = F.mse_loss(edges(recon), edges_x).item()
    losses['perceptual'] = F.mse_loss(recon_feat, x_feat).item()
    losses['core_mse'] = F.mse_loss(recon_core, x).item()
    losses['core_edge'] = F.mse_loss(edges(recon_core), edges_x).item()

    # Structure/appearance on swapped reconstruction
    if B >= 4:
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]

        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = model.decode(z_sw)  # No clamp - decoder sigmoid handles mapping

        losses['swap_structure'] = F.mse_loss(edges(r_sw), edges(x1)).item()
        losses['swap_appearance'] = F.mse_loss(mean_color(r_sw), mean_color(x2)).item()
        losses['swap_color_hist'] = F.mse_loss(color_histogram(r_sw), color_histogram(x2)).item()
    else:
        losses['swap_structure'] = 0.1
        losses['swap_appearance'] = 0.01
        losses['swap_color_hist'] = 0.01
        r_sw = recon  # Fallback for discriminator

    # v14: Discriminator realism scores
    if discriminator is not None:
        with torch.no_grad():
            d_recon = torch.sigmoid(discriminator(recon)).mean()
            d_swap = torch.sigmoid(discriminator(r_sw)).mean()
        # We want D scores HIGH (close to 1 = realistic), so loss = (1 - D_score)
        losses['realism_recon'] = (1.0 - d_recon).item()
        losses['realism_swap'] = (1.0 - d_swap).item()
    else:
        losses['realism_recon'] = 0.5
        losses['realism_swap'] = 0.5

    # v14: KL for BOTH core and detail - no clamps, let BOX constraints enforce bounds
    logvar_core_safe = torch.clamp(logvar_core, min=-30.0, max=20.0)
    kl_per_dim_core = -0.5 * (1 + logvar_core_safe - mu_core.pow(2) - logvar_core_safe.exp())
    losses['kl_core'] = kl_per_dim_core.sum(dim=1).mean().item()

    logvar_detail_safe = torch.clamp(logvar_detail, min=-30.0, max=20.0)
    kl_per_dim_detail = -0.5 * (1 + logvar_detail_safe - mu_detail.pow(2) - logvar_detail_safe.exp())
    losses['kl_detail'] = kl_per_dim_detail.sum(dim=1).mean().item()

    # Direct logvar values - prevent explosion
    losses['logvar_core'] = logvar_core.mean().item()
    losses['logvar_detail'] = logvar_detail.mean().item()

    z_c = z_core - z_core.mean(0, keepdim=True)  # No clamp on z
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    losses['cov'] = ((cov.pow(2).sum() - diag.pow(2).sum()) / diag.pow(2).sum()).item()  # No clamp on cov
    losses['weak'] = (mu_core.var(0) < 0.1).float().mean().item()

    if x_aug is not None:
        with torch.no_grad():
            mu_aug, _ = model.encode(x_aug)
            mu_aug_core = mu_aug[:, :split_idx]
        losses['core_consistency'] = F.mse_loss(mu_core, mu_aug_core).item()
    else:
        losses['core_consistency'] = 0.01

    # v14: Detail contracts - ensure detail channel has proper statistics
    losses['detail_mean'] = mu_detail.mean(0).abs().mean().item()  # Mean across batch, then dims
    losses['detail_var_mean'] = mu_detail.var(0).mean().item()     # Variance across batch, mean over dims

    # Detail covariance (same as core cov calculation)
    z_d = z_detail - z_detail.mean(0, keepdim=True)  # No clamp on z
    cov_d = (z_d.T @ z_d) / (B - 1 + 1e-8)
    diag_d = torch.diag(cov_d) + 1e-8
    losses['detail_cov'] = ((cov_d.pow(2).sum() - diag_d.pow(2).sum()) / diag_d.pow(2).sum()).item()  # No clamp on cov

    # v15: Disentanglement - behavioral leak detection via intervention testing
    with torch.no_grad():
        noise_scale = 0.5
        # Core-only perturbation
        z_core_pert = z_core + torch.randn_like(z_core) * noise_scale
        z_pert_core = torch.cat([z_core_pert, z_detail], dim=1)
        recon_pert_core = model.decode(z_pert_core)
        # Detail-only perturbation
        z_detail_pert = z_detail + torch.randn_like(z_detail) * noise_scale
        z_pert_detail = torch.cat([z_core, z_detail_pert], dim=1)
        recon_pert_detail = model.decode(z_pert_detail)

    losses['core_color_leak'] = F.mse_loss(mean_color(recon_pert_core), mean_color(recon)).item()
    losses['detail_edge_leak'] = F.mse_loss(edges(recon_pert_detail), edges(recon)).item()

    detail_contrib = (recon - recon_core).abs().mean()
    losses['detail_ratio'] = (detail_contrib / (recon_core.abs().mean() + 1e-8)).item()
    losses['core_var_health'] = mu_core.var(0).median().item()
    losses['detail_var_health'] = mu_detail.var(0).median().item()
    losses['core_var_max'] = mu_core.var(0).max().item()
    losses['detail_var_max'] = mu_detail.var(0).max().item()
    losses['_ssim'] = ssim_val.item()

    return losses

def grouped_bom_loss(recon, x, mu, logvar, z, model, goals, vgg, split_idx, group_names, discriminator=None, x_aug=None, use_softmin=False, softmin_temperature=0.1):
    """Compute BOM loss with grouped goals. v15: Adds softmin option, includes v14 discriminator + detail contracts."""
    if not all([check_tensor(t) for t in [recon, x, mu, logvar, z]]):
        return None

    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core, logvar_detail = logvar[:, :split_idx], logvar[:, split_idx:]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)  # No clamp - decoder sigmoid handles mapping
    if not check_tensor(recon_core): return None

    with torch.no_grad():
        x_feat = vgg(x)
    recon_feat = vgg(recon)
    if not all([check_tensor(t) for t in [x_feat, recon_feat]]): return None
    edges_x = edges(x)

    # GROUP A: RECONSTRUCTION
    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val): ssim_val = torch.tensor(0.0, device=x.device)
    g_pixel = goals.goal(pixel_mse + 0.1 * (1.0 - ssim_val), 'pixel')
    edge_loss = F.mse_loss(edges(recon), edges_x)
    g_edge = goals.goal(edge_loss, 'edge')
    g_perceptual = goals.goal(F.mse_loss(recon_feat, x_feat), 'perceptual')

    # GROUP B: CORE STRUCTURE
    g_core_mse = goals.goal(F.mse_loss(recon_core, x), 'core_mse')
    g_core_edge = goals.goal(F.mse_loss(edges(recon_core), edges_x), 'core_edge')
    
    # GROUP C: SWAP - structure/appearance separation
    if B >= 4:
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]

        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = model.decode(z_sw)  # No clamp - decoder sigmoid handles mapping

        structure_loss = F.mse_loss(edges(r_sw), edges(x1))
        g_swap_structure = goals.goal(structure_loss, 'swap_structure')

        appearance_loss = F.mse_loss(mean_color(r_sw), mean_color(x2))
        g_swap_appearance = goals.goal(appearance_loss, 'swap_appearance')

        color_hist_loss = F.mse_loss(color_histogram(r_sw), color_histogram(x2))
        g_swap_color_hist = goals.goal(color_hist_loss, 'swap_color_hist')
    else:
        g_swap_structure = torch.tensor(0.5, device=x.device)
        g_swap_appearance = torch.tensor(0.5, device=x.device)
        g_swap_color_hist = torch.tensor(0.5, device=x.device)
        structure_loss = appearance_loss = color_hist_loss = torch.tensor(0.0, device=x.device)
        r_sw = recon  # Fallback for discriminator

    # GROUP D: REALISM (v14 - Discriminator)
    if discriminator is not None:
        d_recon_logits = discriminator(recon)
        d_swap_logits = discriminator(r_sw)

        # LBO Directive #3: No clamping on raw outputs
        # If discriminator produces extreme logits causing NaN, Directive #4 rollback will handle it
        # Want D scores HIGH (realistic), so minimize (1 - sigmoid(D))
        realism_loss_recon = 1.0 - torch.sigmoid(d_recon_logits).mean()
        realism_loss_swap = 1.0 - torch.sigmoid(d_swap_logits).mean()

        g_realism_recon = goals.goal(realism_loss_recon, 'realism_recon')
        g_realism_swap = goals.goal(realism_loss_swap, 'realism_swap')
    else:
        g_realism_recon = torch.tensor(0.5, device=x.device)
        g_realism_swap = torch.tensor(0.5, device=x.device)
        realism_loss_recon = realism_loss_swap = torch.tensor(0.0, device=x.device)

    # GROUP E: DISENTANGLEMENT - behavioral walls via intervention testing
    # Test: What happens when we perturb ONLY core or ONLY detail?
    # Wall 1: Perturbing core should NOT change colors (core doesn't leak into appearance)
    # Wall 2: Perturbing detail should NOT change edges (detail doesn't leak into structure)
    with torch.no_grad():
        # Controlled perturbation magnitude (0.5 std units)
        noise_scale = 0.5

        # Core-only perturbation: change structure channel, keep appearance channel same
        z_core_pert = z_core + torch.randn_like(z_core) * noise_scale
        z_pert_core = torch.cat([z_core_pert, z_detail], dim=1)
        recon_pert_core = model.decode(z_pert_core)

        # Detail-only perturbation: change appearance channel, keep structure channel same
        z_detail_pert = z_detail + torch.randn_like(z_detail) * noise_scale
        z_pert_detail = torch.cat([z_core, z_detail_pert], dim=1)
        recon_pert_detail = model.decode(z_pert_detail)

    # Measure leaks (should be SMALL)
    core_color_leak = F.mse_loss(mean_color(recon_pert_core), mean_color(recon))
    detail_edge_leak = F.mse_loss(edges(recon_pert_detail), edges(recon))

    g_core_color_leak = goals.goal(core_color_leak, 'core_color_leak')
    g_detail_edge_leak = goals.goal(detail_edge_leak, 'detail_edge_leak')

    # GROUP F: LATENT QUALITY (v14 - KL for BOTH core and detail) - no clamps
    logvar_core_safe = torch.clamp(logvar_core, min=-30.0, max=20.0)
    kl_per_dim_core = -0.5 * (1 + logvar_core_safe - mu_core.pow(2) - logvar_core_safe.exp())
    kl_core_val = kl_per_dim_core.sum(dim=1).mean()
    g_kl_core = goals.goal(kl_core_val, 'kl_core')

    logvar_detail_safe = torch.clamp(logvar_detail, min=-30.0, max=20.0)
    kl_per_dim_detail = -0.5 * (1 + logvar_detail_safe - mu_detail.pow(2) - logvar_detail_safe.exp())
    kl_detail_val = kl_per_dim_detail.sum(dim=1).mean()
    g_kl_detail = goals.goal(kl_detail_val, 'kl_detail')

    # Direct logvar constraints - prevent exp(logvar) from exploding to Inf
    logvar_core_mean = logvar_core.mean()
    g_logvar_core = goals.goal(logvar_core_mean, 'logvar_core')
    logvar_detail_mean = logvar_detail.mean()
    g_logvar_detail = goals.goal(logvar_detail_mean, 'logvar_detail')

    z_c = z_core - z_core.mean(0, keepdim=True)  # No clamp on z
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    cov_penalty = (cov.pow(2).sum() - diag.pow(2).sum()) / diag.pow(2).sum()  # No clamp on cov
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

    # v14: Detail contracts - ensure detail channel has proper statistics
    detail_mean_val = mu_detail.mean(0).abs().mean()
    g_detail_mean = goals.goal(detail_mean_val, 'detail_mean')

    detail_var_mean_val = mu_detail.var(0).mean()
    g_detail_var_mean = goals.goal(detail_var_mean_val, 'detail_var_mean')

    z_d = z_detail - z_detail.mean(0, keepdim=True)  # No clamp on z
    cov_d = (z_d.T @ z_d) / (B - 1 + 1e-8)
    diag_d = torch.diag(cov_d) + 1e-8
    detail_cov_penalty = (cov_d.pow(2).sum() - diag_d.pow(2).sum()) / diag_d.pow(2).sum()  # No clamp on cov
    g_detail_cov = goals.goal(detail_cov_penalty, 'detail_cov')

    # GROUP F: HEALTH
    detail_contrib = (recon - recon_core).abs().mean()
    detail_ratio = detail_contrib / (recon_core.abs().mean() + 1e-8)
    g_detail_ratio = goals.goal(detail_ratio, 'detail_ratio')

    core_var_median, detail_var_median = mu_core.var(0).median(), mu_detail.var(0).median()
    g_core_var = goals.goal(core_var_median, 'core_var_health')
    g_detail_var = goals.goal(detail_var_median, 'detail_var_health')

    core_var_max, detail_var_max = mu_core.var(0).max(), mu_detail.var(0).max()
    g_core_var_max = goals.goal(core_var_max, 'core_var_max')
    g_detail_var_max = goals.goal(detail_var_max, 'detail_var_max')

    # GROUPED BOM - v15: Added disentanglement group (behavioral walls)
    # LBO Directive #3: No clamping - all groups must be allowed to reach their natural values
    # If any group → 0, the system WILL crash, triggering Directive #4 rollback
    group_recon = geometric_mean([g_pixel, g_edge, g_perceptual])
    group_core = geometric_mean([g_core_mse, g_core_edge])
    group_swap = geometric_mean([g_swap_structure, g_swap_appearance, g_swap_color_hist])
    group_realism = geometric_mean([g_realism_recon, g_realism_swap])
    group_disentangle = geometric_mean([g_core_color_leak, g_detail_edge_leak])
    group_latent = geometric_mean([g_kl_core, g_kl_detail, g_logvar_core, g_logvar_detail, g_cov, g_weak, g_consistency, g_detail_mean, g_detail_var_mean, g_detail_cov])
    group_health = geometric_mean([g_detail_ratio, g_core_var, g_detail_var, g_core_var_max, g_detail_var_max])

    groups = torch.stack([group_recon, group_core, group_swap, group_realism, group_disentangle, group_latent, group_health])
    if torch.isnan(groups).any() or torch.isinf(groups).any(): return None

    if use_softmin:
        # Softmin: smooth approximation of min (UNSTABLE - disabled in config)
        min_group = softmin(groups, softmin_temperature)
        min_group_idx = groups.argmin()  # Still track which group is weakest
    else:
        # Hard min: original BOM barrier (ACTIVE)
        min_group = groups.min()
        min_group_idx = groups.argmin()

    # LBO Directive #4: Reject S_min ≤ 0 BEFORE log calculation to prevent crash
    if min_group <= 0:
        return None  # Trigger rollback - constraint violated

    loss = -torch.log(min_group)  # Pure log barrier, NO EPSILON!
    if torch.isnan(loss): return None

    individual_goals = {
        'pixel': g_pixel.item(), 'edge': g_edge.item(), 'perceptual': g_perceptual.item(),
        'core_mse': g_core_mse.item(), 'core_edge': g_core_edge.item(),
        'swap_structure': g_swap_structure.item() if isinstance(g_swap_structure, torch.Tensor) else g_swap_structure,
        'swap_appearance': g_swap_appearance.item() if isinstance(g_swap_appearance, torch.Tensor) else g_swap_appearance,
        'swap_color_hist': g_swap_color_hist.item() if isinstance(g_swap_color_hist, torch.Tensor) else g_swap_color_hist,
        'realism_recon': g_realism_recon.item() if isinstance(g_realism_recon, torch.Tensor) else g_realism_recon,
        'realism_swap': g_realism_swap.item() if isinstance(g_realism_swap, torch.Tensor) else g_realism_swap,
        'core_color_leak': g_core_color_leak.item(), 'detail_edge_leak': g_detail_edge_leak.item(),
        'kl_core': g_kl_core.item(), 'kl_detail': g_kl_detail.item(),
        'logvar_core': g_logvar_core.item(), 'logvar_detail': g_logvar_detail.item(),
        'cov': g_cov.item(), 'weak': g_weak.item(),
        'consistency': g_consistency.item() if isinstance(g_consistency, torch.Tensor) else g_consistency,
        'detail_mean': g_detail_mean.item(), 'detail_var_mean': g_detail_var_mean.item(), 'detail_cov': g_detail_cov.item(),
        'detail_ratio': g_detail_ratio.item(),
        'core_var': g_core_var.item(), 'detail_var': g_detail_var.item(),
        'core_var_max': g_core_var_max.item(), 'detail_var_max': g_detail_var_max.item(),
    }

    group_values = {n: g.item() for n, g in zip(group_names, groups)}

    raw_values = {
        'kl_core_raw': kl_core_val.item(), 'kl_detail_raw': kl_detail_val.item(),
        'logvar_core_raw': logvar_core_mean.item(), 'logvar_detail_raw': logvar_detail_mean.item(),
        'detail_ratio_raw': detail_ratio.item(),
        'core_var_raw': core_var_median.item(), 'detail_var_raw': detail_var_median.item(),
        'core_var_max_raw': core_var_max.item(), 'detail_var_max_raw': detail_var_max.item(),
        'structure_loss': structure_loss.item() if isinstance(structure_loss, torch.Tensor) else structure_loss,
        'appearance_loss': appearance_loss.item() if isinstance(appearance_loss, torch.Tensor) else appearance_loss,
        'color_hist_loss': color_hist_loss.item() if isinstance(color_hist_loss, torch.Tensor) else color_hist_loss,
        'consistency_raw': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
        'realism_recon_raw': realism_loss_recon.item() if isinstance(realism_loss_recon, torch.Tensor) else realism_loss_recon,
        'realism_swap_raw': realism_loss_swap.item() if isinstance(realism_loss_swap, torch.Tensor) else realism_loss_swap,
        'core_color_leak_raw': core_color_leak.item(), 'detail_edge_leak_raw': detail_edge_leak.item(),
        'detail_mean_raw': detail_mean_val.item(), 'detail_var_mean_raw': detail_var_mean_val.item(),
        'detail_cov_raw': detail_cov_penalty.item(),
    }
    
    return {
        'loss': loss, 'groups': groups, 'min_idx': min_group_idx,
        'group_values': group_values, 'individual_goals': individual_goals,
        'raw_values': raw_values, 'ssim': ssim_val.item(),
        'mse': pixel_mse.item(), 'edge_loss': edge_loss.item(),
    }
