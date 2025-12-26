# losses/bom_loss.py
# v13: EXPLICIT structure/appearance separation
# r_sw should have: x1's STRUCTURE + x2's APPEARANCE

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

def compute_raw_losses(recon, x, mu, logvar, z, model, vgg, split_idx, x_aug=None):
    """Compute all raw losses for calibration."""
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core = logvar[:, :split_idx]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = torch.clamp(model.decode(z_core_only), 0, 1)
    recon = torch.clamp(recon, 0, 1)

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
    
    # v13: EXPLICIT structure/appearance on swapped reconstruction
    if B >= 4:
        # Random permutation for robust pairing
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]
        
        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = torch.clamp(model.decode(z_sw), 0, 1)
        
        # STRUCTURE from x1: edges should match
        losses['swap_structure'] = F.mse_loss(edges(r_sw), edges(x1)).item()
        
        # APPEARANCE from x2: colors should match
        losses['swap_appearance'] = F.mse_loss(mean_color(r_sw), mean_color(x2)).item()
        losses['swap_color_hist'] = F.mse_loss(color_histogram(r_sw), color_histogram(x2)).item()
    else:
        losses['swap_structure'] = 0.1
        losses['swap_appearance'] = 0.01
        losses['swap_color_hist'] = 0.01
    
    kl_per_dim = torch.clamp(-0.5 * (1 + logvar_core - mu_core.pow(2) - logvar_core.exp()), 0, 50)
    losses['kl'] = kl_per_dim.sum(dim=1).mean().item()
    
    z_c = torch.clamp(z_core, -10, 10)
    z_c = z_c - z_c.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    losses['cov'] = torch.clamp((cov.pow(2).sum() - diag.pow(2).sum()) / diag.pow(2).sum(), 0, 50).item()
    losses['weak'] = (mu_core.var(0) < 0.1).float().mean().item()
    
    if x_aug is not None:
        with torch.no_grad():
            mu_aug, _ = model.encode(x_aug)
            mu_aug_core = mu_aug[:, :split_idx]
        losses['core_consistency'] = F.mse_loss(mu_core, mu_aug_core).item()
    else:
        losses['core_consistency'] = 0.01
    
    detail_contrib = (recon - recon_core).abs().mean()
    losses['detail_ratio'] = (detail_contrib / (recon_core.abs().mean() + 1e-8)).item()
    losses['core_var_health'] = mu_core.var(0).median().item()
    losses['detail_var_health'] = mu_detail.var(0).median().item()
    losses['core_var_max'] = mu_core.var(0).max().item()
    losses['detail_var_max'] = mu_detail.var(0).max().item()
    losses['_ssim'] = ssim_val.item()
    
    return losses

def grouped_bom_loss(recon, x, mu, logvar, z, model, goals, vgg, split_idx, group_names, x_aug=None):
    """Compute BOM loss with grouped goals. v13: EXPLICIT structure/appearance."""
    if not all([check_tensor(t) for t in [recon, x, mu, logvar, z]]):
        return None
    
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core = logvar[:, :split_idx]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)
    if not check_tensor(recon_core): return None
    recon = torch.clamp(recon, 0, 1)
    recon_core = torch.clamp(recon_core, 0, 1)

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
    
    # GROUP C: SWAP - EXPLICIT structure/appearance separation
    if B >= 4:
        # Random permutation for robust pairing
        perm = torch.randperm(B, device=x.device)
        x1, x2 = x, x[perm]
        z1_core, z2_detail = z_core, z_detail[perm]
        
        z_sw = torch.cat([z1_core, z2_detail], dim=1)
        r_sw = torch.clamp(model.decode(z_sw), 0, 1)
        
        # STRUCTURE from x1: edges should match
        structure_loss = F.mse_loss(edges(r_sw), edges(x1))
        g_swap_structure = goals.goal(structure_loss, 'swap_structure')
        
        # APPEARANCE from x2: colors should match
        appearance_loss = F.mse_loss(mean_color(r_sw), mean_color(x2))
        g_swap_appearance = goals.goal(appearance_loss, 'swap_appearance')
        
        color_hist_loss = F.mse_loss(color_histogram(r_sw), color_histogram(x2))
        g_swap_color_hist = goals.goal(color_hist_loss, 'swap_color_hist')
    else:
        g_swap_structure = torch.tensor(0.5, device=x.device)
        g_swap_appearance = torch.tensor(0.5, device=x.device)
        g_swap_color_hist = torch.tensor(0.5, device=x.device)
        structure_loss = appearance_loss = color_hist_loss = torch.tensor(0.0, device=x.device)

    # GROUP D: LATENT QUALITY
    kl_per_dim = torch.clamp(-0.5 * (1 + logvar_core - mu_core.pow(2) - logvar_core.exp()), 0, 50)
    kl_core = kl_per_dim.sum(dim=1).mean()
    g_kl = goals.goal(kl_core, 'kl')
    
    z_c = torch.clamp(z_core, -10, 10) - torch.clamp(z_core, -10, 10).mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    cov_penalty = torch.clamp((cov.pow(2).sum() - diag.pow(2).sum()) / diag.pow(2).sum(), 0, 50)
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

    # GROUP E: HEALTH
    detail_contrib = (recon - recon_core).abs().mean()
    detail_ratio = detail_contrib / (recon_core.abs().mean() + 1e-8)
    g_detail_ratio = goals.goal(detail_ratio, 'detail_ratio')
    
    core_var_median, detail_var_median = mu_core.var(0).median(), mu_detail.var(0).median()
    g_core_var = goals.goal(core_var_median, 'core_var_health')
    g_detail_var = goals.goal(detail_var_median, 'detail_var_health')
    
    core_var_max, detail_var_max = mu_core.var(0).max(), mu_detail.var(0).max()
    g_core_var_max = goals.goal(core_var_max, 'core_var_max')
    g_detail_var_max = goals.goal(detail_var_max, 'detail_var_max')

    # GROUPED BOM - now with explicit SWAP group
    group_recon = geometric_mean([g_pixel, g_edge, g_perceptual])
    group_core = geometric_mean([g_core_mse, g_core_edge])
    group_swap = geometric_mean([g_swap_structure, g_swap_appearance, g_swap_color_hist])
    group_latent = geometric_mean([g_kl, g_cov, g_weak, g_consistency])
    group_health = geometric_mean([g_detail_ratio, g_core_var, g_detail_var, g_core_var_max, g_detail_var_max])
    
    groups = torch.stack([group_recon, group_core, group_swap, group_latent, group_health])
    if torch.isnan(groups).any() or torch.isinf(groups).any(): return None
    
    min_group = groups.min()
    min_group_idx = groups.argmin()
    loss = -torch.log(min_group)
    if torch.isnan(loss): return None

    individual_goals = {
        'pixel': g_pixel.item(), 'edge': g_edge.item(), 'perceptual': g_perceptual.item(),
        'core_mse': g_core_mse.item(), 'core_edge': g_core_edge.item(),
        'swap_structure': g_swap_structure.item() if isinstance(g_swap_structure, torch.Tensor) else g_swap_structure,
        'swap_appearance': g_swap_appearance.item() if isinstance(g_swap_appearance, torch.Tensor) else g_swap_appearance,
        'swap_color_hist': g_swap_color_hist.item() if isinstance(g_swap_color_hist, torch.Tensor) else g_swap_color_hist,
        'kl': g_kl.item(), 'cov': g_cov.item(), 'weak': g_weak.item(),
        'consistency': g_consistency.item() if isinstance(g_consistency, torch.Tensor) else g_consistency,
        'detail_ratio': g_detail_ratio.item(),
        'core_var': g_core_var.item(), 'detail_var': g_detail_var.item(),
        'core_var_max': g_core_var_max.item(), 'detail_var_max': g_detail_var_max.item(),
    }
    
    group_values = {n: g.item() for n, g in zip(group_names, groups)}
    
    raw_values = {
        'kl_raw': kl_core.item(), 'detail_ratio_raw': detail_ratio.item(),
        'core_var_raw': core_var_median.item(), 'detail_var_raw': detail_var_median.item(),
        'core_var_max_raw': core_var_max.item(), 'detail_var_max_raw': detail_var_max.item(),
        'structure_loss': structure_loss.item() if isinstance(structure_loss, torch.Tensor) else structure_loss,
        'appearance_loss': appearance_loss.item() if isinstance(appearance_loss, torch.Tensor) else appearance_loss,
        'color_hist_loss': color_hist_loss.item() if isinstance(color_hist_loss, torch.Tensor) else color_hist_loss,
        'consistency_raw': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
    }
    
    return {
        'loss': loss, 'groups': groups, 'min_idx': min_group_idx,
        'group_values': group_values, 'individual_goals': individual_goals,
        'raw_values': raw_values, 'ssim': ssim_val.item(),
        'mse': pixel_mse.item(), 'edge_loss': edge_loss.item(),
    }
