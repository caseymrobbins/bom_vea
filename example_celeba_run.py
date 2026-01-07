"""
Simple example script to run BOM-VAE CelebA comparison.

This is a simplified version for quick testing and demonstration.
"""

from bom_vae_celeba_comparison import run_comparison

# Example 1: Local dataset
# Assumes CelebA is downloaded to ./data/celeba/
if __name__ == '__main__':
    import os

    # Configure paths
    celeba_path = './data/celeba'  # Change this to your CelebA path
    output_dir = './outputs'

    # Check if dataset exists
    if not os.path.exists(celeba_path):
        print(f"ERROR: CelebA dataset not found at {celeba_path}")
        print("\nDownload CelebA first:")
        print("1. Download from: https://drive.google.com/file/d/1xJs_8JB0HYXiaAmU8PTG9qbk0WJ2Wo1U")
        print("2. Extract to ./data/celeba/")
        print("\nOr in Colab:")
        print("  !pip install gdown")
        print("  import gdown")
        print('  gdown.download("https://drive.google.com/uc?id=1xJs_8JB0HYXiaAmU8PTG9qbk0WJ2Wo1U", "celeba.zip")')
        print("  !unzip celeba.zip -d ./data/celeba")
        exit(1)

    # Run comparison
    print("="*70)
    print("BOM-VAE vs β-VAE Comparison on CelebA")
    print("="*70)

    results = run_comparison(
        celeba_path=celeba_path,
        n_epochs=20,  # Reduce to 5 for quick test
        batch_size=64,  # Reduce to 32 if OOM
        output_dir=output_dir
    )

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - {output_dir}/training_comparison.png")
    print(f"  - {output_dir}/pareto_comparison.png")
    print(f"  - {output_dir}/reconstructions_comparison.png")
    print(f"  - {output_dir}/samples_comparison.png")
