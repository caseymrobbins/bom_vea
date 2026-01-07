# ==================== LATENT SPACE ANALYSIS ====================
# Add this cell to your Colab notebook after training

# Install dependencies if needed
# !pip install scikit-learn seaborn -q

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

print("="*70)
print("LATENT SPACE ANALYSIS")
print("="*70)

# ==================== 1. EXTRACT LATENTS ====================

def extract_latents(model, loader, device, max_batches=50):
    model.eval()
    all_mu, all_images = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            x = batch[0].to(device)
            h = model.enc(x).view(x.size(0), -1)
            mu = model.fc_mu(h)
            all_mu.append(mu.cpu().numpy())
            all_images.append(x.cpu().numpy())
    return np.vstack(all_mu), np.vstack(all_images)

latents, images = extract_latents(model_bom, test_loader, device, max_batches=50)
print(f"✓ Extracted {len(latents)} latent vectors (dim={latents.shape[1]})\n")

# ==================== 2. LATENT STATISTICS ====================

print("Latent Statistics:")
print(f"  Mean: {latents.mean(axis=0).mean():.4f} ± {latents.mean(axis=0).std():.4f}")
print(f"  Std:  {latents.std(axis=0).mean():.4f} ± {latents.std(axis=0).std():.4f}")

# Plot per-dimension statistics
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(128), latents.mean(axis=0), alpha=0.7)
axes[0].set_title('Mean per Dimension'); axes[0].set_xlabel('Dimension')
axes[0].axhline(0, color='r', linestyle='--', alpha=0.5)

axes[1].bar(range(128), latents.std(axis=0), alpha=0.7)
axes[1].set_title('Std per Dimension'); axes[1].set_xlabel('Dimension')
axes[1].axhline(1, color='r', linestyle='--', alpha=0.5, label='Unit variance')
axes[1].legend()

plt.tight_layout()
plt.savefig('/content/latent_stats.png', dpi=150)
plt.show()

# ==================== 3. t-SNE PROJECTION ====================

print("\nComputing t-SNE projection (this may take a minute)...")
n_samples = min(2000, len(latents))
indices = np.random.choice(len(latents), n_samples, replace=False)
latents_subset = latents[indices]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latents_2d = tsne.fit_transform(latents_subset)

plt.figure(figsize=(10, 8))
plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=np.arange(len(latents_2d)),
            cmap='viridis', alpha=0.6, s=20)
plt.colorbar(label='Sample Index')
plt.title('Latent Space (t-SNE Projection)', fontsize=14)
plt.xlabel('Component 1'); plt.ylabel('Component 2')
plt.tight_layout()
plt.savefig('/content/latent_tsne.png', dpi=150)
plt.show()

# ==================== 4. LATENT TRAVERSALS ====================

print("\nGenerating latent traversals...")
model_bom.eval()
base_z = torch.zeros(1, 128, device=device)

# Show 8 dimensions
dims_to_show = [0, 16, 32, 48, 64, 80, 96, 112]
n_steps = 10

fig, axes = plt.subplots(len(dims_to_show), n_steps, figsize=(n_steps*1.2, len(dims_to_show)*1.2))

with torch.no_grad():
    for i, dim in enumerate(dims_to_show):
        values = np.linspace(-3, 3, n_steps)
        for j, val in enumerate(values):
            z = base_z.clone()
            z[0, dim] = val
            # Decode using the model's decoder
            recon = model_bom.dec(model_bom.fc_dec(z).view(-1, 256, 4, 4))
            axes[i, j].imshow(recon[0].cpu().permute(1, 2, 0).numpy())
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(f'Dim {dim}', fontsize=8)
            if i == 0:
                axes[i, j].set_title(f'{val:.1f}', fontsize=7)

plt.suptitle('Latent Dimension Traversals', fontsize=14)
plt.tight_layout()
plt.savefig('/content/latent_traversals.png', dpi=150)
plt.show()

# ==================== 5. INTERPOLATIONS ====================

print("\nCreating interpolations between faces...")
batch = next(iter(test_loader))[0][:10].to(device)

model_bom.eval()
with torch.no_grad():
    h = model_bom.enc(batch).view(batch.size(0), -1)
    latents_batch = model_bom.fc_mu(h)

n_pairs = 5
n_steps = 10
fig, axes = plt.subplots(n_pairs, n_steps + 2, figsize=((n_steps+2)*1.2, n_pairs*1.2))

with torch.no_grad():
    for i in range(n_pairs):
        z1, z2 = latents_batch[i*2], latents_batch[i*2 + 1]

        # Show start image
        axes[i, 0].imshow(batch[i*2].cpu().permute(1, 2, 0).numpy())
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title('Start', fontsize=8)

        # Interpolate
        for j, alpha in enumerate(np.linspace(0, 1, n_steps)):
            z_interp = (1 - alpha) * z1 + alpha * z2
            # Decode using the model's decoder
            recon = model_bom.dec(model_bom.fc_dec(z_interp.unsqueeze(0)).view(-1, 256, 4, 4))
            axes[i, j+1].imshow(recon[0].cpu().permute(1, 2, 0).numpy())
            axes[i, j+1].axis('off')

        # Show end image
        axes[i, -1].imshow(batch[i*2 + 1].cpu().permute(1, 2, 0).numpy())
        axes[i, -1].axis('off')
        if i == 0: axes[i, -1].set_title('End', fontsize=8)

plt.suptitle('Latent Space Interpolations', fontsize=14)
plt.tight_layout()
plt.savefig('/content/latent_interpolations.png', dpi=150)
plt.show()

# ==================== 6. RANDOM SAMPLES ====================

print("\nSampling from prior N(0, I)...")
z_random = torch.randn(16, 128, device=device)

with torch.no_grad():
    # Decode using the model's decoder
    samples = model_bom.dec(model_bom.fc_dec(z_random).view(-1, 256, 4, 4))

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    ax = axes[i // 8, i % 8]
    ax.imshow(samples[i].cpu().permute(1, 2, 0).numpy())
    ax.axis('off')

plt.suptitle('Random Samples from Prior', fontsize=14)
plt.tight_layout()
plt.savefig('/content/latent_samples.png', dpi=150)
plt.show()

print("\n" + "="*70)
print("✓ Analysis complete! Saved:")
print("  - /content/latent_stats.png")
print("  - /content/latent_tsne.png")
print("  - /content/latent_traversals.png")
print("  - /content/latent_interpolations.png")
print("  - /content/latent_samples.png")
print("="*70)
