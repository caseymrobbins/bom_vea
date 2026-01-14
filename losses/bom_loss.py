# losses/bom_loss.py
# CLEAN REWRITE - LBO VAE Loss Function
# All goals organized by group with clear structure

import torch
import torch.nn.functional as F
from configs import config
from losses.goals import geometric_mean

_sobel_x, _sobel_y = None, None
# Hierarchical mapping: structure ("core") uses early stages, appearance ("detail") uses later stages.
_LATENT_STRUCTURE_KEYS = ("core", "mid")
_LATENT_APPEARANCE_KEYS = ("detail", "resid")

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
    edge_energy = F.conv2d(g, sobel_x, padding=1).pow(2) + F.conv2d(g, sobel_y, padding=1).pow(2)
    return torch.sqrt(edge_energy + 1e-12)

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

def compute_ssim(x, y, window_size=11, per_sample=False):
    """
    Compute SSIM similarity metric.

    Args:
        x, y: Images to compare [B, C, H, W]
        window_size: Gaussian window size
        per_sample: If True, return [B] else scalar
    """
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

    if per_sample:
        # Return [B]: per-sample SSIM
        return torch.clamp(ssim_map.mean(dim=[1,2,3]), -1, 1)
    else:
        # Return scalar: batch mean
        return torch.clamp(ssim_map.mean(), -1, 1)

def check_tensor(t):
    return not (torch.isnan(t).any() or torch.isinf(t).any())

def check_latent_dict(latents):
    return all(check_tensor(t) for t in latents.values())

def concat_latents(latents, keys):
    return torch.cat([latents[key] for key in keys], dim=1)

def structure_latents(latents):
    return concat_latents(latents, _LATENT_STRUCTURE_KEYS)

def appearance_latents(latents):
    return concat_latents(latents, _LATENT_APPEARANCE_KEYS)

# ========== PER-SAMPLE METRIC HELPERS ==========
# LBO-VAE: Compute energies per sample [B], not batch-mean scalars

def mse_per_sample(pred, target):
    """MSE per sample: [B, C, H, W] → [B]"""
    return F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])

def mse_per_sample_1d(pred, target):
    """MSE per sample for 1D features: [B, D] → [B]"""
    return F.mse_loss(pred, target, reduction='none').mean(dim=1)

def mse_per_sample_spatial(pred, target):
    """MSE per sample for spatial features: [B, C, H, W] → [B]"""
    return F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])

def soft_active_count(var_per_dim, threshold=0.1, temperature=0.05):
    """
    Differentiable version of (var > threshold).sum()

    Uses sigmoid to create soft step function:
    - var << threshold → sigmoid ≈ 0 (inactive)
    - var >> threshold → sigmoid ≈ 1 (active)
    - Gradients flow smoothly through threshold region
    """
    return torch.sigmoid((var_per_dim - threshold) / temperature).sum()

def block_diag_prior_kl(mu, logvar, block_size, rho):
    """
    KL(q(z|x) || p(z)) for a block-diagonal prior with intra-block correlation.

    q is diagonal Gaussian with mean mu and log-variance logvar.
    p is block-diagonal Gaussian with zero mean and covariance:
        Sigma_b = (1 - rho) I + rho * 11^T for each block.
    """
    B, D = mu.shape
    logvar_safe = torch.clamp(logvar, min=-30.0, max=20.0)
    var = torch.exp(logvar_safe)
    kl_total = torch.zeros(B, device=mu.device, dtype=mu.dtype)

    start = 0
    while start < D:
        end = min(start + block_size, D)
        b = end - start
        mu_b = mu[:, start:end]
        var_b = var[:, start:end]

        denom = (1.0 - rho) + 1e-6
        denom_corr = (1.0 - rho + rho * b) + 1e-6
        a = 1.0 / denom
        c = -rho / (denom * denom_corr)

        sum_var = var_b.sum(dim=1)
        sum_mu = mu_b.sum(dim=1)
        sum_mu_sq = (mu_b.pow(2)).sum(dim=1)

        trace_term = (a + c) * sum_var
        quad_term = a * sum_mu_sq + c * sum_mu.pow(2)
        base_log = torch.log(torch.as_tensor(1.0 - rho + 1e-6, device=mu.device, dtype=mu.dtype))
        corr_log = torch.log(torch.as_tensor(1.0 - rho + rho * b + 1e-6, device=mu.device, dtype=mu.dtype))
        logdet = (b - 1) * base_log + corr_log

        kl_block = 0.5 * (trace_term + quad_term - b + logdet - logvar_safe[:, start:end].sum(dim=1))
        kl_total = kl_total + kl_block
        start = end

    return kl_total

def compute_raw_losses(recon, x, mu, logvar, z, model, vgg, discriminator=None, x_aug=None):
    """Compute all raw losses for calibration. Used to collect statistics before setting scales."""
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
    losses['prior_kl'] = block_diag_prior_kl(
        mu_all,
        logvar_all,
        config.PRIOR_BLOCK_SIZE,
        config.PRIOR_INTRA_CORR
    ).mean().item()

    losses['logvar_core'] = logvar_core.mean().item()
    losses['logvar_detail'] = logvar_detail.mean().item()

    logvar_safe = torch.clamp(logvar, min=-30.0, max=20.0)
    kl_per_dim_prior = -0.5 * (1 + logvar_safe - mu.pow(2) - logvar_safe.exp())
    losses['prior_kl'] = kl_per_dim_prior.sum(dim=1).mean().item()

    # Structure
    z_c = z_core - z_core.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-2)
    diag = torch.diag(cov) + 1e-2
    cov_penalty = (cov.pow(2).sum() - diag.pow(2).sum()) / torch.clamp(diag.pow(2).sum(), min=1e-1)
    losses['cov'] = cov_penalty.item()
    losses['weak'] = (mu_core.var(0) < 0.1).float().mean().item()

    if x_aug is not None:
        with torch.no_grad():
            mu_aug, _ = model.encode(x_aug)
            mu_aug_core = structure_latents(mu_aug)
        losses['core_consistency'] = F.mse_loss(mu_core, mu_aug_core).item()
    else:
        losses['core_consistency'] = 0.01

    # Capacity
    core_var_per_dim = mu_core.var(0)
    detail_var_per_dim = mu_detail.var(0)
    core_active_count = (core_var_per_dim > 0.1).float().sum()
    detail_active_count = (detail_var_per_dim > 0.1).float().sum()
    total_dims_core = float(mu_core.shape[1])
    total_dims_detail = float(mu_detail.shape[1])

    core_inactive_ratio = (total_dims_core - core_active_count) / total_dims_core
    detail_inactive_ratio = (total_dims_detail - detail_active_count) / total_dims_detail
    losses['core_active'] = core_inactive_ratio.item()
    losses['detail_active'] = detail_inactive_ratio.item()

    core_var_norm = core_var_per_dim / (core_var_per_dim.sum() + 1e-2) + 1e-2
    detail_var_norm = detail_var_per_dim / (detail_var_per_dim.sum() + 1e-2) + 1e-2
    core_var_norm_safe = torch.clamp(core_var_norm, min=1e-2, max=1.0)
    detail_var_norm_safe = torch.clamp(detail_var_norm, min=1e-2, max=1.0)
    core_effective = torch.exp(-torch.sum(core_var_norm_safe * torch.log(core_var_norm_safe)))
    detail_effective = torch.exp(-torch.sum(detail_var_norm_safe * torch.log(detail_var_norm_safe)))

    core_ineffective_ratio = (total_dims_core - core_effective) / total_dims_core
    detail_ineffective_ratio = (total_dims_detail - detail_effective) / total_dims_detail
    losses['core_effective'] = core_ineffective_ratio.item()
    losses['detail_effective'] = detail_ineffective_ratio.item()

    # Detail stats
    losses['detail_mean'] = mu_detail.mean(0).abs().mean().item()
    losses['detail_var_mean'] = mu_detail.var(0).mean().item()

    z_d = z_detail - z_detail.mean(0, keepdim=True)
    cov_d = (z_d.T @ z_d) / (B - 1 + 1e-2)
    diag_d = torch.diag(cov_d) + 1e-2
    detail_cov_penalty = (cov_d.pow(2).sum() - diag_d.pow(2).sum()) / torch.clamp(diag_d.pow(2).sum(), min=1e-1)
    losses['detail_cov'] = detail_cov_penalty.item()

    if tc_logits is not None:
        losses['sep_core'] = tc_logits.get('core', torch.zeros(B, device=x.device)).mean().item()
        losses['sep_mid'] = tc_logits.get('mid', torch.zeros(B, device=x.device)).mean().item()
        losses['sep_detail'] = tc_logits.get('detail', torch.zeros(B, device=x.device)).mean().item()
    else:
        losses['sep_core'] = 0.0
        losses['sep_mid'] = 0.0
        losses['sep_detail'] = 0.0

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
        1.0 / (core_edge_shift + 1e-2) +
        1.0 / (detail_color_shift + 1e-2)
    )
    losses['traversal'] = traversal_loss.item()
    losses['traversal_core_effect'] = core_edge_shift.item()
    losses['traversal_detail_effect'] = detail_color_shift.item()

    # Health
    detail_contrib = (recon - recon_core).abs().mean()
    losses['detail_ratio'] = (detail_contrib / torch.clamp(recon_core.abs().mean(), min=1e-2)).item()
    losses['core_var_health'] = mu_core.var(0).median().item()
    losses['detail_var_health'] = mu_detail.var(0).median().item()

    losses['_ssim'] = ssim_val.item()

    return losses


def grouped_bom_loss(recon, x, mu, logvar, z, model, goals, vgg, group_names, discriminator=None, x_aug=None):
    """
    LBO-VAE Loss Function - Logarithmic Bottleneck Optimization

    Pure LBO formulation:
    - Loss: -log(min(all_scores)) where min is taken over all samples and all groups
    - Single global bottleneck: the worst score across entire batch determines gradient routing
    - Discrete rollback if ANY score ≤ 0

    LBO Directives:
    1. Pure min() barrier - NO softmin, NO epsilon, NO clamping
    2. Pure geometric mean - NO epsilon on aggregation
    3. NO clamping on goals or groups
    4. Discrete rollback if any group ≤ 0
    5. Global bottleneck: -log(min(groups))
    """

    # ========== INPUT VALIDATION ==========
    if not all([check_tensor(t) for t in [recon, x]]) or not all([check_latent_dict(t) for t in [mu, logvar, z]]):
        bad_tensors = []
        for name, t in [('recon', recon), ('x', x)]:
            if not check_tensor(t):
                bad_tensors.append(name)
        for name, t in [('mu', mu), ('logvar', logvar), ('z', z)]:
            if not check_latent_dict(t):
                bad_tensors.append(name)
        print(f"    [INPUT TENSOR FAILURE] Bad tensors: {', '.join(bad_tensors)}")
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

    # ========== GROUP A: RECONSTRUCTION (PER-SAMPLE) ==========
    # Compute energies per sample: [B] not scalars
    pixel_mse_per_sample = mse_per_sample(recon, x)  # [B]
    ssim_per_sample = compute_ssim(recon, x, per_sample=True)  # [B]
    # Handle NaN in SSIM per sample
    ssim_per_sample = torch.where(torch.isnan(ssim_per_sample), torch.zeros_like(ssim_per_sample), ssim_per_sample)

    # Goals now vectorized: [B] → [B]
    g_pixel = goals.goal(pixel_mse_per_sample + 0.1 * (1.0 - ssim_per_sample), 'pixel')  # [B]
    g_edge = goals.goal(mse_per_sample(edges(recon), edges_x), 'edge')  # [B]
    g_perceptual = goals.goal(mse_per_sample_spatial(recon_feat, x_feat), 'perceptual')  # [B] - VGG features are spatial

    # ========== GROUP B: CORE (PER-SAMPLE) ==========
    # Core (structure channel) should preserve structure
    g_core_mse = goals.goal(mse_per_sample(recon_core, x), 'core_mse')  # [B]
    g_core_edge = goals.goal(mse_per_sample(edges(recon_core), edges_x), 'core_edge')  # [B]

    # ========== GROUP C: SWAP (PER-SAMPLE) ==========
    # Structure from x1, appearance from x2
    if B >= 4:
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]
        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = model.decode(z_sw)

        # Per-sample losses: [B]
        structure_loss = mse_per_sample(edges(r_sw), edges(x1))  # [B]
        appearance_loss = mse_per_sample_1d(mean_color(r_sw), mean_color(x2))  # [B]
        color_hist_loss = mse_per_sample_1d(color_histogram(r_sw), color_histogram(x2))  # [B]

        g_swap_structure = goals.goal(structure_loss, 'swap_structure')  # [B]
        g_swap_appearance = goals.goal(appearance_loss, 'swap_appearance')  # [B]
        g_swap_color_hist = goals.goal(color_hist_loss, 'swap_color_hist')  # [B]
    else:
        # Batch too small for swapping - return [B] of 0.5
        g_swap_structure = torch.full((B,), 0.5, device=x.device)
        g_swap_appearance = torch.full((B,), 0.5, device=x.device)
        g_swap_color_hist = torch.full((B,), 0.5, device=x.device)
        structure_loss = torch.zeros(B, device=x.device)
        appearance_loss = torch.zeros(B, device=x.device)
        color_hist_loss = torch.zeros(B, device=x.device)
        r_sw = recon

    # ========== GROUP D: REALISM (PER-SAMPLE) ==========
    # Discriminator should classify reconstructions as real
    if discriminator is not None:
        d_recon_logits = discriminator(recon)  # [B, 1, H, W] - spatial discriminator
        d_swap_logits = discriminator(r_sw)    # [B, 1, H, W]

        # Want D scores HIGH (realistic), so minimize (1 - sigmoid(D))
        # Per-sample: average over spatial dimensions [B]
        realism_loss_recon = 1.0 - torch.sigmoid(d_recon_logits).mean(dim=[1, 2, 3])  # [B]
        realism_loss_swap = 1.0 - torch.sigmoid(d_swap_logits).mean(dim=[1, 2, 3])    # [B]

        g_realism_recon = goals.goal(realism_loss_recon, 'realism_recon')  # [B]
        g_realism_swap = goals.goal(realism_loss_swap, 'realism_swap')      # [B]
    else:
        g_realism_recon = torch.full((B,), 0.5, device=x.device)
        g_realism_swap = torch.full((B,), 0.5, device=x.device)
        realism_loss_recon = torch.zeros(B, device=x.device)
        realism_loss_swap = torch.zeros(B, device=x.device)

    # ========== GROUP E: DISENTANGLE (PER-SAMPLE) ==========
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

    # Measure leaks (should be SMALL) - per sample: [B]
    core_color_leak = mse_per_sample_1d(mean_color(recon_pert_core), mean_color(recon))  # [B]
    detail_edge_leak = mse_per_sample(edges(recon_pert_detail), edges(recon))  # [B]

    g_core_color_leak = goals.goal(core_color_leak, 'core_color_leak')  # [B]
    g_detail_edge_leak = goals.goal(detail_edge_leak, 'detail_edge_leak')  # [B]

    # Measure intended effects (should be LARGE) - used for traversal metric
    core_edge_shift = mse_per_sample(edges(recon_pert_core), edges(recon))  # [B]
    detail_color_shift = mse_per_sample_1d(mean_color(recon_pert_detail), mean_color(recon))  # [B]

    # Traversal: reward large intended shifts (1/shift → minimize shift^-1)
    # Per sample: [B]
    traversal_loss = 0.5 * (
        1.0 / (core_edge_shift + 1e-2) +
        1.0 / (detail_color_shift + 1e-2)
    )
    g_traversal = goals.goal(traversal_loss, 'traversal')  # [B]

    # ========== GROUP F: SEPARATION (PER-SAMPLE) ==========
    if tc_logits is not None:
        sep_core = tc_logits.get('core', torch.zeros(B, device=x.device))
        sep_mid = tc_logits.get('mid', torch.zeros(B, device=x.device))
        sep_detail = tc_logits.get('detail', torch.zeros(B, device=x.device))
    else:
        sep_core = torch.zeros(B, device=x.device)
        sep_mid = torch.zeros(B, device=x.device)
        sep_detail = torch.zeros(B, device=x.device)

    g_sep_core = goals.goal(sep_core, 'sep_core')  # [B]
    g_sep_mid = goals.goal(sep_mid, 'sep_mid')  # [B]
    g_sep_detail = goals.goal(sep_detail, 'sep_detail')  # [B]

    # ========== GROUP G: LATENT (Hierarchical, PER-SAMPLE) ==========

    # SUB-GROUP F1: KL (Distribution Matching) - PER SAMPLE
    logvar_core_safe = torch.clamp(logvar_core, min=-30.0, max=20.0)
    kl_per_dim_core = -0.5 * (1 + logvar_core_safe - mu_core.pow(2) - logvar_core_safe.exp())
    kl_core_val = kl_per_dim_core.sum(dim=1)  # [B] - KL per sample
    g_kl_core = goals.goal(kl_core_val, 'kl_core')  # [B]

    logvar_detail_safe = torch.clamp(logvar_detail, min=-30.0, max=20.0)
    kl_per_dim_detail = -0.5 * (1 + logvar_detail_safe - mu_detail.pow(2) - logvar_detail_safe.exp())
    kl_detail_val = kl_per_dim_detail.sum(dim=1)  # [B] - KL per sample
    g_kl_detail = goals.goal(kl_detail_val, 'kl_detail')  # [B]

    prior_kl_val = block_diag_prior_kl(
        mu_all,
        logvar_all,
        config.PRIOR_BLOCK_SIZE,
        config.PRIOR_INTRA_CORR
    )
    g_prior_kl = goals.goal(prior_kl_val, 'prior_kl')  # [B]

    # Direct logvar constraints - BATCH LEVEL (global property)
    # Expand to [B] for consistent shape
    logvar_core_mean = logvar_core.mean()
    g_logvar_core = goals.goal(logvar_core_mean, 'logvar_core')  # scalar
    if not isinstance(g_logvar_core, torch.Tensor) or g_logvar_core.dim() == 0:
        g_logvar_core = g_logvar_core * torch.ones(B, device=x.device)  # [B]

    logvar_detail_mean = logvar_detail.mean()
    g_logvar_detail = goals.goal(logvar_detail_mean, 'logvar_detail')  # scalar
    if not isinstance(g_logvar_detail, torch.Tensor) or g_logvar_detail.dim() == 0:
        g_logvar_detail = g_logvar_detail * torch.ones(B, device=x.device)  # [B]

    group_kl = geometric_mean([g_kl_core, g_kl_detail, g_prior_kl, g_logvar_core, g_logvar_detail])  # [B]

    # SUB-GROUP F2: STRUCTURE (Independence & Consistency) - BATCH LEVEL
    # These are global latent space properties - computed as batch scalars, expanded to [B]
    z_c = z_core - z_core.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-2)
    diag = torch.diag(cov) + 1e-2
    cov_penalty = (cov.pow(2).sum() - diag.pow(2).sum()) / torch.clamp(diag.pow(2).sum(), min=1e-1)
    g_cov = goals.goal(cov_penalty, 'cov')  # scalar
    if not isinstance(g_cov, torch.Tensor) or g_cov.dim() == 0:
        g_cov = g_cov * torch.ones(B, device=x.device)  # [B]

    # DIFFERENTIABLE weak dims: use soft threshold instead of hard comparison
    weak_penalty = 1.0 - torch.sigmoid((mu_core.var(0) - 0.1) / 0.05).mean()  # Soft version
    g_weak = goals.goal(weak_penalty, 'weak')  # scalar
    if not isinstance(g_weak, torch.Tensor) or g_weak.dim() == 0:
        g_weak = g_weak * torch.ones(B, device=x.device)  # [B]

    if x_aug is not None:
        with torch.no_grad():
            mu_aug, _ = model.encode(x_aug)
        mu_aug_core = structure_latents(mu_aug)
        # Per-sample consistency: [B]
        consistency_loss = mse_per_sample_1d(mu_core, mu_aug_core)
        g_consistency = goals.goal(consistency_loss, 'core_consistency')  # [B]
    else:
        g_consistency = torch.full((B,), 0.5, device=x.device)
        consistency_loss = torch.zeros(B, device=x.device)

    group_structure = geometric_mean([g_cov, g_weak, g_consistency])  # [B]

    # SUB-GROUP F3: CAPACITY (Dimension Utilization) - BATCH LEVEL + DIFFERENTIABLE
    # Global latent space properties - use soft thresholds for differentiability
    core_var_per_dim = mu_core.var(0)  # [D]
    detail_var_per_dim = mu_detail.var(0)  # [D]

    # DIFFERENTIABLE active count: use soft sigmoid instead of hard threshold
    core_active_count = soft_active_count(core_var_per_dim, threshold=0.1, temperature=0.05)
    detail_active_count = soft_active_count(detail_var_per_dim, threshold=0.1, temperature=0.05)
    total_dims_core = float(mu_core.shape[1])
    total_dims_detail = float(mu_detail.shape[1])

    core_inactive_ratio = (total_dims_core - core_active_count) / total_dims_core
    detail_inactive_ratio = (total_dims_detail - detail_active_count) / total_dims_detail
    g_core_active = goals.goal(core_inactive_ratio, 'core_active')  # scalar
    g_detail_active = goals.goal(detail_inactive_ratio, 'detail_active')  # scalar

    # Expand to [B]
    if not isinstance(g_core_active, torch.Tensor) or g_core_active.dim() == 0:
        g_core_active = g_core_active * torch.ones(B, device=x.device)
    if not isinstance(g_detail_active, torch.Tensor) or g_detail_active.dim() == 0:
        g_detail_active = g_detail_active * torch.ones(B, device=x.device)

    # Effective dimensions (exponential of entropy) - already differentiable
    core_var_norm = core_var_per_dim / (core_var_per_dim.sum() + 1e-2) + 1e-2
    detail_var_norm = detail_var_per_dim / (detail_var_per_dim.sum() + 1e-2) + 1e-2
    core_var_norm_safe = torch.clamp(core_var_norm, min=1e-2, max=1.0)
    detail_var_norm_safe = torch.clamp(detail_var_norm, min=1e-2, max=1.0)
    core_effective = torch.exp(-torch.sum(core_var_norm_safe * torch.log(core_var_norm_safe)))
    detail_effective = torch.exp(-torch.sum(detail_var_norm_safe * torch.log(detail_var_norm_safe)))

    core_ineffective_ratio = (total_dims_core - core_effective) / total_dims_core
    detail_ineffective_ratio = (total_dims_detail - detail_effective) / total_dims_detail
    g_core_effective = goals.goal(core_ineffective_ratio, 'core_effective')  # scalar
    g_detail_effective = goals.goal(detail_ineffective_ratio, 'detail_effective')  # scalar

    # Expand to [B]
    if not isinstance(g_core_effective, torch.Tensor) or g_core_effective.dim() == 0:
        g_core_effective = g_core_effective * torch.ones(B, device=x.device)
    if not isinstance(g_detail_effective, torch.Tensor) or g_detail_effective.dim() == 0:
        g_detail_effective = g_detail_effective * torch.ones(B, device=x.device)

    group_capacity = geometric_mean([g_core_active, g_detail_active, g_core_effective, g_detail_effective])  # [B]

    # SUB-GROUP F4: DETAIL STATS - BATCH LEVEL
    detail_mean_val = mu_detail.mean(0).abs().mean()
    g_detail_mean = goals.goal(detail_mean_val, 'detail_mean')  # scalar
    if not isinstance(g_detail_mean, torch.Tensor) or g_detail_mean.dim() == 0:
        g_detail_mean = g_detail_mean * torch.ones(B, device=x.device)  # [B]

    detail_var_mean_val = mu_detail.var(0).mean()
    g_detail_var_mean = goals.goal(detail_var_mean_val, 'detail_var_mean')  # scalar
    if not isinstance(g_detail_var_mean, torch.Tensor) or g_detail_var_mean.dim() == 0:
        g_detail_var_mean = g_detail_var_mean * torch.ones(B, device=x.device)  # [B]

    z_d = z_detail - z_detail.mean(0, keepdim=True)
    cov_d = (z_d.T @ z_d) / (B - 1 + 1e-2)
    diag_d = torch.diag(cov_d) + 1e-2
    detail_cov_penalty = (cov_d.pow(2).sum() - diag_d.pow(2).sum()) / torch.clamp(diag_d.pow(2).sum(), min=1e-1)
    g_detail_cov = goals.goal(detail_cov_penalty, 'detail_cov')  # scalar
    if not isinstance(g_detail_cov, torch.Tensor) or g_detail_cov.dim() == 0:
        g_detail_cov = g_detail_cov * torch.ones(B, device=x.device)  # [B]

    group_detail_stats = geometric_mean([g_detail_mean, g_detail_var_mean, g_detail_cov, g_traversal])  # [B]

    # ========== GROUP H: HEALTH - BATCH LEVEL ==========
    # Variance statistics (global properties)
    detail_contrib = (recon - recon_core).abs().mean()
    detail_ratio = detail_contrib / torch.clamp(recon_core.abs().mean(), min=1e-2)
    g_detail_ratio = goals.goal(detail_ratio, 'detail_ratio')  # scalar
    if not isinstance(g_detail_ratio, torch.Tensor) or g_detail_ratio.dim() == 0:
        g_detail_ratio = g_detail_ratio * torch.ones(B, device=x.device)  # [B]

    core_var_median = mu_core.var(0).median()
    detail_var_median = mu_detail.var(0).median()
    g_core_var = goals.goal(core_var_median, 'core_var_health')  # scalar
    g_detail_var = goals.goal(detail_var_median, 'detail_var_health')  # scalar
    if not isinstance(g_core_var, torch.Tensor) or g_core_var.dim() == 0:
        g_core_var = g_core_var * torch.ones(B, device=x.device)  # [B]
    if not isinstance(g_detail_var, torch.Tensor) or g_detail_var.dim() == 0:
        g_detail_var = g_detail_var * torch.ones(B, device=x.device)  # [B]

    # ========== AGGREGATE GROUPS (PER-SAMPLE) ==========
    # All groups now have shape [B]
    group_recon = geometric_mean([g_pixel, g_edge, g_perceptual])  # [B]
    group_core = geometric_mean([g_core_mse, g_core_edge])  # [B]
    group_swap = geometric_mean([g_swap_structure, g_swap_appearance, g_swap_color_hist])  # [B]
    group_realism = geometric_mean([g_realism_recon, g_realism_swap])  # [B]
    group_disentangle = geometric_mean([g_core_color_leak, g_detail_edge_leak])  # [B]
    group_separation = geometric_mean([g_sep_core, g_sep_mid, g_sep_detail, g_prior_kl])  # [B]
    group_latent = geometric_mean([group_kl, group_structure, group_capacity, group_detail_stats])  # [B]
    group_health = geometric_mean([g_detail_ratio, g_core_var, g_detail_var])  # [B] - core_var_max/detail_var_max removed per main

    # Stack as [B, n_groups] for per-sample bottleneck selection
    groups = torch.stack([group_recon, group_core, group_swap, group_realism, group_disentangle, group_separation, group_latent, group_health], dim=1)  # [B, 8]

    # ========== CHECK FOR NaN/Inf ==========
    if torch.isnan(groups).any() or torch.isinf(groups).any():
        print(f"    [NaN/Inf DETECTED] in groups tensor")
        return None

    # ========== LBO LOSS (GLOBAL BOTTLENECK) ==========
    # Pure LBO: -log(min(all_scores))
    # Find the single worst score across ALL samples and ALL groups
    global_min = groups.min()  # Scalar - worst score in entire batch

    # Safety check for NaN/Inf in global_min before taking log
    if torch.isnan(global_min) or torch.isinf(global_min):
        print(f"    [LBO BARRIER] global_min is NaN/Inf")
        return None

    # Discrete rollback if global minimum ≤ 0
    if global_min <= 0:
        # Find which sample and group failed
        min_per_sample, idx_per_sample = groups.min(dim=1)  # [B], [B]
        failed_mask = min_per_sample <= 0
        n_failed = failed_mask.sum().item()
        failed_indices = idx_per_sample[failed_mask]
        if len(failed_indices) > 0:
            most_common_failure = failed_indices.mode().values.item()
            group_name = group_names[most_common_failure] if most_common_failure < len(group_names) else f"group_{most_common_failure}"
        else:
            group_name = "unknown"

        print(f"    [LBO BARRIER] {n_failed}/{B} samples failed, most common: '{group_name}'")

        # DETAILED DIAGNOSTICS: Show sub-group/goal breakdown (batch mean)
        if group_name == 'latent':
            latent_subgroups = {
                'kl': group_kl.mean().item(),
                'structure': group_structure.mean().item(),
                'capacity': group_capacity.mean().item(),
                'detail_stats': group_detail_stats.mean().item()
            }
            failed_subgroups = [name for name, val in latent_subgroups.items() if val <= 0]

            print(f"    └─ Sub-groups (batch mean): kl={latent_subgroups['kl']:.6f}, structure={latent_subgroups['structure']:.6f}, capacity={latent_subgroups['capacity']:.6f}, detail_stats={latent_subgroups['detail_stats']:.6f}")

            if failed_subgroups:
                print(f"    └─ Failed sub-groups: {', '.join(failed_subgroups)}")

                # Show individual goals in failed sub-groups (batch mean)
                if 'kl' in failed_subgroups:
                    print(f"       KL goals: kl_core={g_kl_core.mean():.6f}, kl_detail={g_kl_detail.mean():.6f}, logvar_core={g_logvar_core.mean():.6f}, logvar_detail={g_logvar_detail.mean():.6f}")
                if 'structure' in failed_subgroups:
                    print(f"       Structure goals: cov={g_cov.mean():.6f}, weak={g_weak.mean():.6f}, consistency={g_consistency.mean():.6f}")
                if 'capacity' in failed_subgroups:
                    print(f"       Capacity goals: core_active={g_core_active.mean():.6f}, detail_active={g_detail_active.mean():.6f}, core_effective={g_core_effective.mean():.6f}, detail_effective={g_detail_effective.mean():.6f}")
                if 'detail_stats' in failed_subgroups:
                    print(f"       Detail stats goals: detail_mean={g_detail_mean.mean():.6f}, detail_var_mean={g_detail_var_mean.mean():.6f}, detail_cov={g_detail_cov.mean():.6f}, traversal={g_traversal.mean():.6f}")

        elif group_name == 'health':
            health_goals = {
                'detail_ratio': g_detail_ratio.mean().item(),
                'core_var': g_core_var.mean().item(),
                'detail_var': g_detail_var.mean().item(),
            }
            failed_goals = [name for name, val in health_goals.items() if val <= 0]

            print(f"    └─ Goals (batch mean): detail_ratio={health_goals['detail_ratio']:.6f}, core_var={health_goals['core_var']:.6f}, detail_var={health_goals['detail_var']:.6f}")

            if failed_goals:
                print(f"    └─ Failed goals: {', '.join(failed_goals)}")
        else:
            print(f"    └─ Min score: {min_per_sample.min():.6f}")

        return None

    # Directive #1: Pure logarithmic barrier
    # LBO loss: -log(min(all_scores))
    loss = -torch.log(global_min)  # Scalar

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"    [LOSS NaN/Inf] Found NaN/Inf in -log(global_min={global_min:.6f})")
        return None

    # ========== RETURN RESULTS (BATCH MEANS FOR LOGGING) ==========
    # Most common bottleneck group across samples (for diagnostics)
    min_per_sample, idx_per_sample = groups.min(dim=1)  # [B], [B]
    min_group_idx = idx_per_sample.mode().values.item()

    # Convert [B] tensors to batch-mean scalars for logging
    individual_goals = {
        'pixel': g_pixel.mean().item(), 'edge': g_edge.mean().item(), 'perceptual': g_perceptual.mean().item(),
        'core_mse': g_core_mse.mean().item(), 'core_edge': g_core_edge.mean().item(),
        'swap_structure': g_swap_structure.mean().item(),
        'swap_appearance': g_swap_appearance.mean().item(),
        'swap_color_hist': g_swap_color_hist.mean().item(),
        'realism_recon': g_realism_recon.mean().item(),
        'realism_swap': g_realism_swap.mean().item(),
        'core_color_leak': g_core_color_leak.mean().item(), 'detail_edge_leak': g_detail_edge_leak.mean().item(),
        'traversal': g_traversal.mean().item(),
        'sep_core': g_sep_core.mean().item(), 'sep_mid': g_sep_mid.mean().item(), 'sep_detail': g_sep_detail.mean().item(),
        'prior_kl': g_prior_kl.mean().item(),
        'kl_core': g_kl_core.mean().item(), 'kl_detail': g_kl_detail.mean().item(),
        'prior_kl': g_prior_kl.mean().item(),
        'logvar_core': g_logvar_core.mean().item(), 'logvar_detail': g_logvar_detail.mean().item(),
        'cov': g_cov.mean().item(), 'weak': g_weak.mean().item(),
        'consistency': g_consistency.mean().item(),
        'core_active': g_core_active.mean().item(), 'detail_active': g_detail_active.mean().item(),
        'core_effective': g_core_effective.mean().item(), 'detail_effective': g_detail_effective.mean().item(),
        'detail_mean': g_detail_mean.mean().item(), 'detail_var_mean': g_detail_var_mean.mean().item(), 'detail_cov': g_detail_cov.mean().item(),
        'detail_ratio': g_detail_ratio.mean().item(),
        'core_var': g_core_var.mean().item(), 'detail_var': g_detail_var.mean().item(),
    }

    # groups is [B, 8] - compute batch mean per group
    group_values = {n: groups[:, i].mean().item() for i, n in enumerate(group_names)}

    raw_values = {
        'kl_core_raw': kl_core_val.mean().item(), 'kl_detail_raw': kl_detail_val.mean().item(),
        'prior_kl_raw': prior_kl_val.mean().item(),
        'logvar_core_raw': logvar_core_mean.item(), 'logvar_detail_raw': logvar_detail_mean.item(),
        'core_active_raw': core_active_count.item(), 'detail_active_raw': detail_active_count.item(),
        'core_effective_raw': core_effective.item(), 'detail_effective_raw': detail_effective.item(),
        'detail_ratio_raw': detail_ratio.item(),
        'core_var_raw': core_var_median.item(), 'detail_var_raw': detail_var_median.item(),
        'structure_loss': structure_loss.mean().item(),
        'appearance_loss': appearance_loss.mean().item(),
        'color_hist_loss': color_hist_loss.mean().item(),
        'consistency_raw': consistency_loss.mean().item(),
        'realism_recon_raw': realism_loss_recon.mean().item(),
        'realism_swap_raw': realism_loss_swap.mean().item(),
        'core_color_leak_raw': core_color_leak.mean().item(), 'detail_edge_leak_raw': detail_edge_leak.mean().item(),
        'traversal_raw': traversal_loss.mean().item(),
        'traversal_core_effect_raw': core_edge_shift.mean().item(),
        'traversal_detail_effect_raw': detail_color_shift.mean().item(),
        'detail_mean_raw': detail_mean_val.item(), 'detail_var_mean_raw': detail_var_mean_val.item(),
        'detail_cov_raw': detail_cov_penalty.item(),
        'sep_core_raw': sep_core.mean().item(), 'sep_mid_raw': sep_mid.mean().item(), 'sep_detail_raw': sep_detail.mean().item(),
        'prior_kl_raw': prior_kl_val.mean().item(),
    }

    return {
        'loss': loss, 'groups': groups, 'min_idx': min_group_idx,
        'group_values': group_values, 'individual_goals': individual_goals,
        'raw_values': raw_values, 'ssim': ssim_per_sample.mean().item(),
        'mse': pixel_mse_per_sample.mean().item(), 'edge_loss': mse_per_sample(edges(recon), edges_x).mean().item(),
    }
