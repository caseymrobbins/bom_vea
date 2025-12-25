# utils/viz.py
# Visualization utilities

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_group_balance(histories, group_names, output_path, title="BOM VAE: Group Balance"):
    """Plot group balance over training."""
    ep = range(1, len(histories['loss']) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ep, histories['min_group'], 'k-', lw=3, label='Min Group')
    
    colors = ['blue', 'red', 'green', 'purple']
    for i, n in enumerate(group_names):
        ax.plot(ep, histories[f'group_{n}'], color=colors[i], lw=2, label=n)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Group Score (0-1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_goal_details(histories, group_names, output_path):
    """Plot detailed goal metrics."""
    ep = range(1, len(histories['loss']) + 1)
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    
    # Reconstruction
    axs[0,0].plot(ep, histories['pixel'], label='pixel')
    axs[0,0].plot(ep, histories['edge_goal'], label='edge')
    axs[0,0].plot(ep, histories['perceptual'], label='perceptual')
    axs[0,0].plot(ep, histories['group_recon'], 'k--', lw=2, label='group')
    axs[0,0].set_title('Reconstruction Group')
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].set_ylim(0, 1)
    
    # Core
    axs[0,1].plot(ep, histories['core_mse'], label='core_mse')
    axs[0,1].plot(ep, histories['cross'], label='cross')
    axs[0,1].plot(ep, histories['texture_contrastive'], label='tex_contrastive', lw=2)
    axs[0,1].plot(ep, histories['texture_match'], label='tex_match', lw=2, ls='--')
    axs[0,1].plot(ep, histories['group_core'], 'k--', lw=2, label='group')
    axs[0,1].set_title('Core Group')
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3)
    axs[0,1].set_ylim(0, 1)
    
    # Latent
    axs[0,2].plot(ep, histories['kl_goal'], label='kl')
    axs[0,2].plot(ep, histories['cov_goal'], label='cov')
    axs[0,2].plot(ep, histories['weak'], label='weak')
    axs[0,2].plot(ep, histories['group_latent'], 'k--', lw=2, label='group')
    axs[0,2].set_title('Latent Group')
    axs[0,2].legend()
    axs[0,2].grid(True, alpha=0.3)
    axs[0,2].set_ylim(0, 1)
    
    # Health
    axs[1,0].plot(ep, histories['detail_ratio_goal'], label='detail_ratio')
    axs[1,0].plot(ep, histories['core_var_goal'], label='core_var')
    axs[1,0].plot(ep, histories['detail_var_goal'], label='detail_var')
    axs[1,0].plot(ep, histories['group_health'], 'k--', lw=2, label='group')
    axs[1,0].set_title('Health Group')
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)
    axs[1,0].set_ylim(0, 1)
    
    # Texture distances
    axs[1,1].plot(ep, histories['texture_dist_x2'], 'b-', lw=2, label='dist to x2 (want LOW)')
    axs[1,1].plot(ep, histories['texture_dist_x1'], 'r-', lw=2, label='dist to x1')
    axs[1,1].set_title('Texture Distances')
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)
    
    # Raw values
    ax2 = axs[1,2]
    ax2.plot(ep, histories['detail_ratio_raw'], 'g-', lw=2, label='detail_ratio')
    ax2.axhline(0.1, color='g', ls='--', alpha=0.5)
    ax2.axhline(0.5, color='g', ls='--', alpha=0.5)
    ax2.set_ylabel('Detail Ratio', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0, 0.6)
    ax2.legend(loc='upper left')
    ax2.set_title('Raw Values')
    ax2.grid(True, alpha=0.3)
    
    ax3 = ax2.twinx()
    ax3.plot(ep, histories['kl_raw'], 'b-', lw=2, label='KL')
    ax3.axhline(50, color='b', ls='--', alpha=0.3)
    ax3.axhline(2000, color='b', ls='-', alpha=0.3)
    ax3.set_ylabel('KL', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstructions(model, samples, split_idx, output_path, device='cuda'):
    """Plot original, full reconstruction, core-only, and detail contribution."""
    model.eval()
    with torch.no_grad():
        samples = samples[:16].to(device)
        rec_f, _, _, z = model(samples)
        rec_f = torch.clamp(rec_f, 0, 1)
        
        z_core_only = z.clone()
        z_core_only[:, split_idx:] = 0
        rec_c = torch.clamp(model.decode(z_core_only), 0, 1)
        
        detail_contrib = torch.clamp((rec_f - rec_c).abs() * 5, 0, 1)
        
        fig, axs = plt.subplots(4, 16, figsize=(20, 5.5))
        fig.suptitle("Original | Full | Core-Only | Detail×5", fontsize=14, y=1.02)
        
        for i in range(16):
            axs[0,i].imshow(samples[i].cpu().permute(1,2,0).numpy())
            axs[0,i].axis('off')
            axs[1,i].imshow(rec_f[i].cpu().permute(1,2,0).numpy())
            axs[1,i].axis('off')
            axs[2,i].imshow(rec_c[i].cpu().permute(1,2,0).numpy())
            axs[2,i].axis('off')
            axs[3,i].imshow(detail_contrib[i].cpu().permute(1,2,0).numpy())
            axs[3,i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_traversals(model, sample, split_idx, output_path_core, output_path_detail, 
                    num_dims=15, device='cuda'):
    """Plot latent traversals for core and detail dimensions."""
    model.eval()
    with torch.no_grad():
        sample = sample[:1].to(device)
        mu, _ = model.encode(sample)
        scales = np.linspace(-3, 3, 11)
        
        # Core traversals
        fig, axs = plt.subplots(num_dims, 11, figsize=(22, num_dims*2))
        fig.suptitle("Core Traversals", fontsize=16, y=1.01)
        
        for d in range(num_dims):
            for j, sc in enumerate(scales):
                z = mu.clone()
                z[0, d] = mu[0, d] + sc
                img = torch.clamp(model.decode(z), 0, 1)[0].cpu().numpy()
                axs[d,j].imshow(np.transpose(img, (1,2,0)))
                axs[d,j].axis('off')
                if j == 0:
                    axs[d,j].set_ylabel(f'D{d}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path_core, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Detail traversals
        fig, axs = plt.subplots(num_dims, 11, figsize=(22, num_dims*2))
        fig.suptitle("Detail Traversals", fontsize=16, y=1.01)
        
        for di in range(num_dims):
            d = split_idx + di
            for j, sc in enumerate(scales):
                z = mu.clone()
                z[0, d] = mu[0, d] + sc
                img = torch.clamp(model.decode(z), 0, 1)[0].cpu().numpy()
                axs[di,j].imshow(np.transpose(img, (1,2,0)))
                axs[di,j].axis('off')
                if j == 0:
                    axs[di,j].set_ylabel(f'D{d}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path_detail, dpi=150, bbox_inches='tight')
        plt.close()


def plot_cross_reconstruction(model, samples, split_idx, output_path, device='cuda'):
    """Plot cross-reconstruction: x1 core + x2 detail."""
    model.eval()
    with torch.no_grad():
        n = 8
        x1 = samples[:n].to(device)
        x2 = samples[n:2*n].to(device) if samples.shape[0] >= 2*n else x1
        
        _, _, _, z1 = model(x1)
        _, _, _, z2 = model(x2)
        
        z_sw = torch.cat([z1[:, :split_idx], z2[:, split_idx:]], dim=1)
        r_sw = torch.clamp(model.decode(z_sw), 0, 1)
        
        fig, axs = plt.subplots(3, n, figsize=(n*2.5, 8))
        fig.suptitle("Cross-Recon: x1 Core + x2 Detail", fontsize=14)
        
        for i in range(n):
            axs[0,i].imshow(x1[i].cpu().permute(1,2,0).numpy())
            axs[0,i].axis('off')
            if i == 0:
                axs[0,i].set_ylabel('x1 (structure)', fontsize=12)
            
            axs[1,i].imshow(x2[i].cpu().permute(1,2,0).numpy())
            axs[1,i].axis('off')
            if i == 0:
                axs[1,i].set_ylabel('x2 (texture)', fontsize=12)
            
            axs[2,i].imshow(r_sw[i].cpu().permute(1,2,0).numpy())
            axs[2,i].axis('off')
            if i == 0:
                axs[2,i].set_ylabel('x1_core+x2_detail', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_dimension_activity(histories, dim_variance_history, split_idx, output_path):
    """Plot dimension activity and variance distributions."""
    ep = range(1, len(histories['loss']) + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Active dimensions
    axs[0,0].plot(ep, histories['core_active'], 'b-', lw=2, label='Core')
    axs[0,0].plot(ep, histories['detail_active'], 'r-', lw=2, label='Detail')
    axs[0,0].axhline(split_idx, color='k', ls='--', alpha=0.3)
    axs[0,0].set_title('Active Dimensions')
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].set_ylim(0, split_idx + 6)
    
    # Effective dimensionality
    axs[0,1].plot(ep, histories['core_effective'], 'b-', lw=2, label='Core')
    axs[0,1].plot(ep, histories['detail_effective'], 'r-', lw=2, label='Detail')
    axs[0,1].set_title('Effective Dimensionality')
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3)
    
    # Final variance distributions
    if dim_variance_history['core']:
        final_core_var = dim_variance_history['core'][-1]
        axs[1,0].bar(range(split_idx), final_core_var, color='blue', alpha=0.7)
        axs[1,0].axhline(0.5, color='g', ls='--', lw=2)
        axs[1,0].axhline(50, color='r', ls='--', lw=2)
        active = (final_core_var > 0.1).sum()
        axs[1,0].set_title(f'Core Dims Variance - {active}/{split_idx} active')
        axs[1,0].grid(True, alpha=0.3)
    
    if dim_variance_history['detail']:
        final_detail_var = dim_variance_history['detail'][-1]
        axs[1,1].bar(range(split_idx), final_detail_var, color='red', alpha=0.7)
        axs[1,1].axhline(0.5, color='g', ls='--', lw=2)
        axs[1,1].axhline(50, color='r', ls='--', lw=2)
        active = (final_detail_var > 0.1).sum()
        axs[1,1].set_title(f'Detail Dims Variance - {active}/{split_idx} active')
        axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(histories, output_path):
    """Plot training history summary."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    
    axs[0,0].plot(histories['loss'], 'b-', lw=2)
    axs[0,0].set_title('Loss')
    axs[0,0].grid(True, alpha=0.3)
    
    axs[0,1].plot(histories['ssim'], 'g-', lw=2)
    axs[0,1].set_title('SSIM')
    axs[0,1].grid(True, alpha=0.3)
    
    axs[0,2].plot(histories['texture_contrastive'], 'r-', lw=2, label='contrastive')
    axs[0,2].plot(histories['texture_match'], 'orange', lw=2, label='match')
    axs[0,2].set_title('Texture Goals')
    axs[0,2].legend()
    axs[0,2].grid(True, alpha=0.3)
    
    axs[1,0].plot(histories['core_var_raw'], 'b-', lw=2, label='core')
    axs[1,0].plot(histories['detail_var_raw'], 'r-', lw=2, label='detail')
    axs[1,0].axhline(0.5, c='g', ls=':', alpha=0.5)
    axs[1,0].axhline(50, c='g', ls=':', alpha=0.5)
    axs[1,0].set_title('Dim Variance')
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)
    
    axs[1,1].plot(histories['kl_raw'], 'c-', lw=2)
    axs[1,1].axhline(50, c='r', ls='--', alpha=0.5)
    axs[1,1].axhline(2000, c='g', ls='-', alpha=0.5)
    axs[1,1].set_title('KL')
    axs[1,1].grid(True, alpha=0.3)
    
    axs[1,2].plot(histories['min_group'], 'k-', lw=2)
    axs[1,2].set_title('Min Group')
    axs[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
