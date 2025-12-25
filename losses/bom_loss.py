# losses/bom_loss.py
# BOM Loss - Bottleneck Optimization Method

import torch
import torch.nn.functional as F
from losses.goals import geometric_mean
from models.vgg import texture_distance

# Sobel filters for edge detection
_sobel_x = None
_sobel_y = None

def _get_sobel(device):
    global _sobel_x, _sobel_y
    if _sobel_x is None or _sobel_x.device != device:
        _sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3).to(device)
        _sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3).to(device)
    return _sobel_x, _sobel_y

def edges(img):
    """Compute edge magnitude using Sobel filters."""
    sobel_x, sobel_y = _get_sobel(img.device)
    g = img.mean(1, keepdim=True)
    return (F.conv2d(g, sobel_x, padding=1).pow(2) + F.conv2d(g, sobel_y, padding=1).pow(2)).sqrt()

def compute_ssim(x, y, window_size=11):
    """Compute SSIM between two images."""
    C1, C2 = 0.01**2, 0.03**2
    sigma = 1.5
    gauss = torch.exp(-torch.arange(window_size, device=x.device, dtype=torch.float32).sub(window_size//2).pow(2) / (2*sigma**2))
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
    """Check for NaN/Inf."""
    return not (torch.isnan(t).any() or torch.isinf(t).any())

def compute_texture_loss(r_sw, x1, x2, vgg):
    """Compute contrastive texture loss and absolute distance."""
    r_sw_feat = vgg(r_sw, return_all=True)
    x1_feat = vgg(x1, return_all=True)
    x2_feat = vgg(x2, return_all=True)
    
    dist_to_x2 = texture_distance(r_sw_feat, x2_feat)
    dist_to_x1 = texture_distance(r_sw_feat, x1_feat)
    
    margin = 0.1
    contrastive_loss = F.relu(dist_to_x2 - dist_to_x1 + margin)
    
    return contrastive_loss.mean(), dist_to_x2.mean(), dist_to_x1.mean()


def compute_raw_losses(recon, x, mu, logvar, z, model, vgg, split_idx):
    """Compute all raw losses for calibration."""
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core = logvar[:, :split_idx]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)
    recon = torch.clamp(recon, 0, 1)
    recon_core = torch.clamp(recon_core, 0, 1)

    with torch.no_grad():
        x_feat = vgg(x)
    recon_feat = vgg(recon)
    edges_x = edges(x)

    losses = {}
    
    # Reconstruction losses
    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val):
        ssim_val = torch.tensor(0.0, device=x.device)
    losses['pixel'] = (pixel_mse + 0.1 * (1.0 - ssim_val)).item()
    losses['edge'] = F.mse_loss(edges(recon), edges_x).item()
    losses['perceptual'] = F.mse_loss(recon_feat, x_feat).item()
    
    # Core losses
    losses['core_mse'] = F.mse_loss(recon_core, x).item()
    losses['core_edge'] = F.mse_loss(edges(recon_core), edges_x).item()
    
    # Cross-reconstruction and texture
    if B >= 4:
        h = B // 2
        z1_c, z2_d = z_core[:h], z_detail[h:2*h]
        x1, x2 = x[:h], x[h:2*h]
        z_sw = torch.cat([z1_c, z2_d], dim=1)
        r_sw = torch.clamp(model.decode(z_sw), 0, 1)
        
        losses['cross'] = (F.mse_loss(r_sw, x1) + F.mse_loss(edges(r_sw), edges(x1))).item()
        
        with torch.no_grad():
            tex_loss, dist_x2, _ = compute_texture_loss(r_sw, x1, x2, vgg)
        losses['texture_contrastive'] = tex_loss.item()
        losses['texture_match'] = dist_x2.item()
    else:
        losses['cross'] = 0.1
        losses['texture_contrastive'] = 0.1
        losses['texture_match'] = 0.1
    
    # Latent losses
    kl_per_dim = -0.5 * (1 + logvar_core - mu_core.pow(2) - logvar_core.exp())
    kl_per_dim = torch.clamp(kl_per_dim, 0, 50)
    losses['kl'] = kl_per_dim.sum(dim=1).mean().item()
    
    z_c = torch.clamp(z_core, -10, 10)
    z_c = z_c - z_c.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    off_diag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    losses['cov'] = torch.clamp(off_diag_sq / diag.pow(2).sum(), 0, 50).item()
    
    mu_var = mu_core.var(0) + 1e-8
    losses['weak'] = (mu_var < 0.1).float().mean().item()
    
    # Health metrics
    detail_contrib = (recon - recon_core).abs().mean()
    core_mag = recon_core.abs().mean() + 1e-8
    losses['detail_ratio'] = (detail_contrib / core_mag).item()
    losses['core_var_health'] = mu_core.var(0).median().item()
    losses['detail_var_health'] = mu_detail.var(0).median().item()
    losses['core_var_max'] = mu_core.var(0).max().item()
    losses['detail_var_max'] = mu_detail.var(0).max().item()
    
    losses['_ssim'] = ssim_val.item()
    
    return losses


def grouped_bom_loss(recon, x, mu, logvar, z, model, goals, vgg, split_idx, group_names):
    """
    Compute BOM loss with grouped goals.
    
    Returns None if tensors contain NaN/Inf.
    """
    if not all([check_tensor(t) for t in [recon, x, mu, logvar, z]]):
        return None
    
    B = x.shape[0]
    z_core, z_detail = z[:, :split_idx], z[:, split_idx:]
    mu_core, mu_detail = mu[:, :split_idx], mu[:, split_idx:]
    logvar_core = logvar[:, :split_idx]

    z_core_only = torch.cat([z_core, torch.zeros_like(z_detail)], dim=1)
    recon_core = model.decode(z_core_only)
    if not check_tensor(recon_core):
        return None
    
    recon = torch.clamp(recon, 0, 1)
    recon_core = torch.clamp(recon_core, 0, 1)

    with torch.no_grad():
        x_feat = vgg(x)
    recon_feat = vgg(recon)
    if not all([check_tensor(t) for t in [x_feat, recon_feat]]):
        return None
    
    edges_x = edges(x)

    # === GROUP A: RECONSTRUCTION ===
    pixel_mse = F.mse_loss(recon, x)
    ssim_val = compute_ssim(recon, x)
    if torch.isnan(ssim_val):
        ssim_val = torch.tensor(0.0, device=x.device)
    pixel_loss = pixel_mse + 0.1 * (1.0 - ssim_val)
    g_pixel = goals.goal(pixel_loss, 'pixel')
    
    edge_loss = F.mse_loss(edges(recon), edges_x)
    g_edge = goals.goal(edge_loss, 'edge')
    
    perceptual_loss = F.mse_loss(recon_feat, x_feat)
    g_perceptual = goals.goal(perceptual_loss, 'perceptual')

    # === GROUP B: CORE STRUCTURE ===
    g_core_mse = goals.goal(F.mse_loss(recon_core, x), 'core_mse')
    g_core_edge = goals.goal(F.mse_loss(edges(recon_core), edges_x), 'core_edge')
    
    if B >= 4:
        h = B // 2
        z1_c, z2_d = z_core[:h], z_detail[h:2*h]
        x1, x2 = x[:h], x[h:2*h]
        z_sw = torch.cat([z1_c, z2_d], dim=1)
        r_sw = torch.clamp(model.decode(z_sw), 0, 1)
        
        cross_loss = F.mse_loss(r_sw, x1) + F.mse_loss(edges(r_sw), edges(x1))
        g_cross = goals.goal(cross_loss, 'cross')
        
        texture_loss, dist_to_x2, dist_to_x1 = compute_texture_loss(r_sw, x1, x2, vgg)
        g_texture_contrastive = goals.goal(texture_loss, 'texture_contrastive')
        g_texture_match = goals.goal(dist_to_x2, 'texture_match')
    else:
        g_cross = torch.tensor(0.5, device=x.device)
        g_texture_contrastive = torch.tensor(0.5, device=x.device)
        g_texture_match = torch.tensor(0.5, device=x.device)
        texture_loss = dist_to_x2 = dist_to_x1 = torch.tensor(0.0, device=x.device)

    # === GROUP C: LATENT QUALITY ===
    kl_per_dim = -0.5 * (1 + logvar_core - mu_core.pow(2) - logvar_core.exp())
    kl_per_dim = torch.clamp(kl_per_dim, 0, 50)
    kl_core = kl_per_dim.sum(dim=1).mean()
    g_kl = goals.goal(kl_core, 'kl')
    
    z_c = torch.clamp(z_core, -10, 10)
    z_c = z_c - z_c.mean(0, keepdim=True)
    cov = (z_c.T @ z_c) / (B - 1 + 1e-8)
    diag = torch.diag(cov) + 1e-8
    off_diag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    cov_penalty = torch.clamp(off_diag_sq / diag.pow(2).sum(), 0, 50)
    g_cov = goals.goal(cov_penalty, 'cov')
    
    mu_var = mu_core.var(0) + 1e-8
    g_weak = goals.goal((mu_var < 0.1).float().mean(), 'weak')

    # === GROUP D: HEALTH ===
    detail_contrib = (recon - recon_core).abs().mean()
    core_mag = recon_core.abs().mean() + 1e-8
    detail_ratio = detail_contrib / core_mag
    g_detail_ratio = goals.goal(detail_ratio, 'detail_ratio')
    
    core_var_median = mu_core.var(0).median()
    detail_var_median = mu_detail.var(0).median()
    g_core_var = goals.goal(core_var_median, 'core_var_health')
    g_detail_var = goals.goal(detail_var_median, 'detail_var_health')
    
    core_var_max = mu_core.var(0).max()
    detail_var_max = mu_detail.var(0).max()
    g_core_var_max = goals.goal(core_var_max, 'core_var_max')
    g_detail_var_max = goals.goal(detail_var_max, 'detail_var_max')

    # === GROUPED BOM ===
    group_recon = geometric_mean([g_pixel, g_edge, g_perceptual])
    group_core = geometric_mean([g_core_mse, g_core_edge, g_cross, g_texture_contrastive, g_texture_match])
    group_latent = geometric_mean([g_kl, g_cov, g_weak])
    group_health = geometric_mean([g_detail_ratio, g_core_var, g_detail_var, g_core_var_max, g_detail_var_max])
    
    groups = torch.stack([group_recon, group_core, group_latent, group_health])
    if torch.isnan(groups).any() or torch.isinf(groups).any():
        return None
    
    min_group = groups.min()
    min_group_idx = groups.argmin()
    loss = -torch.log(min_group)
    
    if torch.isnan(loss):
        return None

    # Collect metrics
    individual_goals = {
        'pixel': g_pixel.item(), 'edge': g_edge.item(), 'perceptual': g_perceptual.item(),
        'core_mse': g_core_mse.item(), 'core_edge': g_core_edge.item(),
        'cross': g_cross.item() if isinstance(g_cross, torch.Tensor) else g_cross,
        'texture_contrastive': g_texture_contrastive.item() if isinstance(g_texture_contrastive, torch.Tensor) else g_texture_contrastive,
        'texture_match': g_texture_match.item() if isinstance(g_texture_match, torch.Tensor) else g_texture_match,
        'kl': g_kl.item(), 'cov': g_cov.item(), 'weak': g_weak.item(),
        'detail_ratio': g_detail_ratio.item(),
        'core_var': g_core_var.item(), 'detail_var': g_detail_var.item(),
        'core_var_max': g_core_var_max.item(), 'detail_var_max': g_detail_var_max.item(),
    }
    
    group_values = {n: g.item() for n, g in zip(group_names, groups)}
    
    raw_values = {
        'kl_raw': kl_core.item(),
        'detail_ratio_raw': detail_ratio.item(),
        'core_var_raw': core_var_median.item(),
        'detail_var_raw': detail_var_median.item(),
        'core_var_max_raw': core_var_max.item(),
        'detail_var_max_raw': detail_var_max.item(),
        'texture_loss': texture_loss.item() if isinstance(texture_loss, torch.Tensor) else texture_loss,
        'dist_x2': dist_to_x2.item() if isinstance(dist_to_x2, torch.Tensor) else dist_to_x2,
        'dist_x1': dist_to_x1.item() if isinstance(dist_to_x1, torch.Tensor) else dist_to_x1,
    }
    
    return {
        'loss': loss,
        'groups': groups,
        'min_idx': min_group_idx,
        'group_values': group_values,
        'individual_goals': individual_goals,
        'raw_values': raw_values,
        'ssim': ssim_val.item(),
        'mse': pixel_mse.item(),
        'edge_loss': edge_loss.item(),
    }
