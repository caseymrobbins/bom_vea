"""
Latent Space Analysis for BOM-VAE

Visualize and analyze the learned latent representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 1. EXTRACT LATENTS ====================

def extract_latents(model, loader, device, max_batches=50):
    """
    Extract latent representations (mu, logvar, z) from dataset.

    Returns:
        latents_mu: Mean of latent distribution (deterministic)
        latents_z: Sampled latent vectors
        latents_logvar: Log-variance of latent distribution
        images: Original images
    """
    model.eval()
    all_mu, all_z, all_logvar, all_images = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            x = batch[0].to(device)

            # Encode
            h = model.enc(x).view(x.size(0), -1)
            mu = model.fc_mu(h)
            logvar = model.fc_logvar(h)

            # Sample
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

            all_mu.append(mu.cpu().numpy())
            all_z.append(z.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_images.append(x.cpu().numpy())

    latents_mu = np.vstack(all_mu)
    latents_z = np.vstack(all_z)
    latents_logvar = np.vstack(all_logvar)
    images = np.vstack(all_images)

    print(f"✓ Extracted {len(latents_mu)} latent vectors (dim={latents_mu.shape[1]})")
    return latents_mu, latents_z, latents_logvar, images


# ==================== 2. LATENT SPACE VISUALIZATION ====================

def visualize_latent_space_2d(latents, method='tsne', n_samples=2000):
    """
    Project high-dimensional latents to 2D for visualization.

    Args:
        latents: (N, latent_dim) array
        method: 'tsne' or 'pca'
        n_samples: Number of samples to visualize
    """
    if len(latents) > n_samples:
        indices = np.random.choice(len(latents), n_samples, replace=False)
        latents = latents[indices]

    print(f"Projecting {len(latents)} points to 2D using {method.upper()}...")

    if method == 'tsne':
        projector = TSNE(n_components=2, random_state=42, perplexity=30)
        latents_2d = projector.fit_transform(latents)
    elif method == 'pca':
        projector = PCA(n_components=2, random_state=42)
        latents_2d = projector.fit_transform(latents)
        print(f"  Explained variance: {projector.explained_variance_ratio_.sum():.2%}")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                        c=np.arange(len(latents_2d)), cmap='viridis',
                        alpha=0.6, s=20)
    ax.set_title(f'Latent Space Projection ({method.upper()})', fontsize=14)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    plt.tight_layout()

    return latents_2d, fig


# ==================== 3. LATENT STATISTICS ====================

def analyze_latent_statistics(latents_mu, latents_logvar):
    """Analyze statistical properties of the latent space."""
    latent_dim = latents_mu.shape[1]

    # Compute statistics per dimension
    means = latents_mu.mean(axis=0)
    stds = latents_mu.std(axis=0)
    vars = np.exp(latents_logvar.mean(axis=0))  # Average variance per dimension

    print("="*70)
    print("LATENT SPACE STATISTICS")
    print("="*70)
    print(f"Latent dimensions: {latent_dim}")
    print(f"Mean activation: {means.mean():.4f} ± {means.std():.4f}")
    print(f"Std activation: {stds.mean():.4f} ± {stds.std():.4f}")
    print(f"Average variance: {vars.mean():.4f} ± {vars.std():.4f}")
    print()

    # Plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mean per dimension
    axes[0].bar(range(latent_dim), means, alpha=0.7)
    axes[0].set_title('Mean per Latent Dimension')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Mean')
    axes[0].axhline(0, color='r', linestyle='--', alpha=0.5)

    # Std per dimension
    axes[1].bar(range(latent_dim), stds, alpha=0.7)
    axes[1].set_title('Std per Latent Dimension')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Std')
    axes[1].axhline(1, color='r', linestyle='--', alpha=0.5, label='Unit variance')
    axes[1].legend()

    # Variance per dimension
    axes[2].bar(range(latent_dim), vars, alpha=0.7)
    axes[2].set_title('Model Variance per Dimension')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Variance')

    plt.tight_layout()

    return means, stds, vars, fig


# ==================== 4. LATENT TRAVERSALS ====================

def latent_traversal(model, device, latent_dim=128, n_steps=10, traversal_range=(-3, 3)):
    """
    Traverse along individual latent dimensions to see what they encode.

    Shows how varying each dimension affects the generated image.
    """
    model.eval()

    # Start from zero latent (mean of prior)
    base_z = torch.zeros(1, latent_dim, device=device)

    # Select a few interesting dimensions to visualize
    n_dims_to_show = min(10, latent_dim)
    dims_to_show = np.linspace(0, latent_dim-1, n_dims_to_show, dtype=int)

    fig, axes = plt.subplots(n_dims_to_show, n_steps, figsize=(n_steps*1.5, n_dims_to_show*1.5))

    with torch.no_grad():
        for i, dim in enumerate(dims_to_show):
            # Traverse this dimension
            values = np.linspace(traversal_range[0], traversal_range[1], n_steps)

            for j, val in enumerate(values):
                z = base_z.clone()
                z[0, dim] = val

                # Decode using the model's decoder
                recon = model.dec(model.fc_dec(z).view(-1, 256, 4, 4))

                # Plot
                ax = axes[i, j] if n_dims_to_show > 1 else axes[j]
                ax.imshow(recon[0].cpu().permute(1, 2, 0).numpy())
                ax.axis('off')

                if j == 0:
                    ax.set_ylabel(f'Dim {dim}', fontsize=10)
                if i == 0:
                    ax.set_title(f'{val:.1f}', fontsize=8)

    plt.suptitle('Latent Dimension Traversals', fontsize=14)
    plt.tight_layout()

    return fig


# ==================== 5. LATENT INTERPOLATIONS ====================

def interpolate_latents(model, loader, device, n_pairs=5, n_steps=10):
    """
    Interpolate between pairs of real images in latent space.

    Shows smooth transitions between different faces.
    """
    model.eval()

    # Get some random images
    batch = next(iter(loader))[0].to(device)
    n_images = min(n_pairs * 2, len(batch))
    images = batch[:n_images]

    # Encode to latents
    with torch.no_grad():
        h = model.enc(images).view(images.size(0), -1)
        latents = model.fc_mu(h)  # Use mean (deterministic)

    fig, axes = plt.subplots(n_pairs, n_steps + 2, figsize=((n_steps+2)*1.5, n_pairs*1.5))

    with torch.no_grad():
        for i in range(n_pairs):
            # Get two latents
            z1 = latents[i*2]
            z2 = latents[i*2 + 1]

            # Interpolate
            alphas = np.linspace(0, 1, n_steps)

            for j, alpha in enumerate(alphas):
                z_interp = (1 - alpha) * z1 + alpha * z2
                # Decode using the model's decoder
                recon = model.dec(model.fc_dec(z_interp.unsqueeze(0)).view(-1, 256, 4, 4))

                ax = axes[i, j+1] if n_pairs > 1 else axes[j+1]
                ax.imshow(recon[0].cpu().permute(1, 2, 0).numpy())
                ax.axis('off')

            # Show original images at endpoints
            ax_start = axes[i, 0] if n_pairs > 1 else axes[0]
            ax_end = axes[i, -1] if n_pairs > 1 else axes[-1]

            ax_start.imshow(images[i*2].cpu().permute(1, 2, 0).numpy())
            ax_start.axis('off')
            ax_start.set_title('Start', fontsize=8)

            ax_end.imshow(images[i*2 + 1].cpu().permute(1, 2, 0).numpy())
            ax_end.axis('off')
            ax_end.set_title('End', fontsize=8)

    plt.suptitle('Latent Space Interpolations', fontsize=14)
    plt.tight_layout()

    return fig


# ==================== 6. RANDOM SAMPLING ====================

def sample_from_prior(model, device, latent_dim=128, n_samples=16):
    """
    Sample random latent vectors from N(0, I) and decode.

    Shows the diversity of the learned generative model.
    """
    model.eval()

    # Sample from standard normal
    z = torch.randn(n_samples, latent_dim, device=device)

    with torch.no_grad():
        # Decode using the model's decoder
        samples = model.dec(model.fc_dec(z).view(-1, 256, 4, 4))

    # Plot in grid
    n_cols = 8
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(samples[i].cpu().permute(1, 2, 0).numpy())
        axes[i].axis('off')

    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Random Samples from Prior N(0, I)', fontsize=14)
    plt.tight_layout()

    return fig


# ==================== MAIN ANALYSIS ====================

def run_latent_analysis(model, loader, device, output_dir='/content'):
    """
    Run complete latent space analysis.

    Args:
        model: Trained VAE model
        loader: DataLoader
        device: torch device
        output_dir: Where to save plots
    """
    print("="*70)
    print("LATENT SPACE ANALYSIS")
    print("="*70)

    # 1. Extract latents
    print("\n1. Extracting latent representations...")
    latents_mu, latents_z, latents_logvar, images = extract_latents(model, loader, device, max_batches=50)

    # 2. Statistics
    print("\n2. Computing statistics...")
    means, stds, vars, fig_stats = analyze_latent_statistics(latents_mu, latents_logvar)
    fig_stats.savefig(f'{output_dir}/latent_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3. 2D Projection (t-SNE)
    print("\n3. Creating t-SNE projection...")
    latents_2d_tsne, fig_tsne = visualize_latent_space_2d(latents_mu, method='tsne', n_samples=2000)
    fig_tsne.savefig(f'{output_dir}/latent_tsne.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 4. 2D Projection (PCA)
    print("\n4. Creating PCA projection...")
    latents_2d_pca, fig_pca = visualize_latent_space_2d(latents_mu, method='pca', n_samples=2000)
    fig_pca.savefig(f'{output_dir}/latent_pca.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 5. Latent traversals
    print("\n5. Generating latent traversals...")
    fig_trav = latent_traversal(model, device, latent_dim=model.latent_dim, n_steps=10)
    fig_trav.savefig(f'{output_dir}/latent_traversals.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 6. Interpolations
    print("\n6. Creating latent interpolations...")
    fig_interp = interpolate_latents(model, loader, device, n_pairs=5, n_steps=10)
    fig_interp.savefig(f'{output_dir}/latent_interpolations.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 7. Random samples
    print("\n7. Sampling from prior...")
    fig_samples = sample_from_prior(model, device, latent_dim=model.latent_dim, n_samples=16)
    fig_samples.savefig(f'{output_dir}/latent_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Saved plots to: {output_dir}/")
    print("  - latent_statistics.png")
    print("  - latent_tsne.png")
    print("  - latent_pca.png")
    print("  - latent_traversals.png")
    print("  - latent_interpolations.png")
    print("  - latent_samples.png")

    return {
        'latents_mu': latents_mu,
        'latents_z': latents_z,
        'latents_logvar': latents_logvar,
        'images': images,
        'means': means,
        'stds': stds,
        'vars': vars,
    }


# ==================== USAGE ====================

if __name__ == '__main__':
    # Assuming you have a trained model from the main script
    # If running in Colab after training:

    # Run analysis
    analysis_results = run_latent_analysis(
        model=model_bom,  # Your trained BOM-VAE model
        loader=test_loader,
        device=device,
        output_dir='/content'
    )

    # Download results
    from google.colab import files
    files.download('/content/latent_statistics.png')
    files.download('/content/latent_tsne.png')
    files.download('/content/latent_pca.png')
    files.download('/content/latent_traversals.png')
    files.download('/content/latent_interpolations.png')
    files.download('/content/latent_samples.png')
