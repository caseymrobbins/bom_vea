# utils/viz.py - Same as v11, abbreviated
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_group_balance(histories, group_names, output_path, title="BOM VAE"):
    ep = range(1, len(histories['loss']) + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ep, histories['min_group'], 'k-', lw=3, label='Min Group')
    for i, n in enumerate(group_names):
        ax.plot(ep, histories[f'group_{n}'], lw=2, label=n)
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.1)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()

def plot_reconstructions(model, samples, split_idx, output_path, device='cuda'):
    model.eval()
    with torch.no_grad():
        samples = samples[:16].to(device)
        rec_f, _, _, z = model(samples)
        rec_f = torch.clamp(rec_f, 0, 1)
        z_co = z.clone(); z_co[:, split_idx:] = 0
        rec_c = torch.clamp(model.decode(z_co), 0, 1)
        detail = torch.clamp((rec_f - rec_c).abs() * 5, 0, 1)
        
        fig, axs = plt.subplots(4, 16, figsize=(20, 5.5))
        for i in range(16):
            axs[0,i].imshow(samples[i].cpu().permute(1,2,0).numpy()); axs[0,i].axis('off')
            axs[1,i].imshow(rec_f[i].cpu().permute(1,2,0).numpy()); axs[1,i].axis('off')
            axs[2,i].imshow(rec_c[i].cpu().permute(1,2,0).numpy()); axs[2,i].axis('off')
            axs[3,i].imshow(detail[i].cpu().permute(1,2,0).numpy()); axs[3,i].axis('off')
        plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()

def plot_cross_reconstruction(model, samples, split_idx, output_path, device='cuda'):
    model.eval()
    with torch.no_grad():
        n = 8
        x1, x2 = samples[:n].to(device), samples[n:2*n].to(device)
        _, _, _, z1 = model(x1)
        _, _, _, z2 = model(x2)
        z_sw = torch.cat([z1[:, :split_idx], z2[:, split_idx:]], dim=1)
        r_sw = torch.clamp(model.decode(z_sw), 0, 1)
        
        fig, axs = plt.subplots(3, n, figsize=(n*2.5, 8))
        for i in range(n):
            axs[0,i].imshow(x1[i].cpu().permute(1,2,0).numpy()); axs[0,i].axis('off')
            axs[1,i].imshow(x2[i].cpu().permute(1,2,0).numpy()); axs[1,i].axis('off')
            axs[2,i].imshow(r_sw[i].cpu().permute(1,2,0).numpy()); axs[2,i].axis('off')
        plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()

def plot_traversals(model, sample, split_idx, output_path_core, output_path_detail, num_dims=15, device='cuda'):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(sample[:1].to(device))
        scales = np.linspace(-3, 3, 11)
        for prefix, offset, path in [('Core', 0, output_path_core), ('Detail', split_idx, output_path_detail)]:
            fig, axs = plt.subplots(num_dims, 11, figsize=(22, num_dims*2))
            for d in range(num_dims):
                for j, sc in enumerate(scales):
                    z = mu.clone(); z[0, offset+d] = mu[0, offset+d] + sc
                    img = torch.clamp(model.decode(z), 0, 1)[0].cpu().numpy()
                    axs[d,j].imshow(np.transpose(img, (1,2,0))); axs[d,j].axis('off')
            plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_training_history(histories, output_path):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0,0].plot(histories['loss']); axs[0,0].set_title('Loss'); axs[0,0].grid(True, alpha=0.3)
    axs[0,1].plot(histories['ssim']); axs[0,1].set_title('SSIM'); axs[0,1].grid(True, alpha=0.3)
    axs[0,2].plot(histories['texture_contrastive'], label='contrastive')
    axs[0,2].plot(histories['texture_match'], label='match')
    axs[0,2].set_title('Texture'); axs[0,2].legend(); axs[0,2].grid(True, alpha=0.3)
    axs[1,0].plot(histories['core_var_raw'], label='core')
    axs[1,0].plot(histories['detail_var_raw'], label='detail')
    axs[1,0].set_title('Variance'); axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)
    axs[1,1].plot(histories['kl_raw']); axs[1,1].set_title('KL'); axs[1,1].grid(True, alpha=0.3)
    axs[1,2].plot(histories['min_group']); axs[1,2].set_title('Min Group'); axs[1,2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
