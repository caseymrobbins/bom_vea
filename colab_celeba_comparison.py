"""
Complete Colab-ready script for BOM-VAE CelebA comparison.

Copy this entire file into a Colab cell or run as a standalone script.
"""

# ==================== SETUP (Run first in Colab) ====================

import os
import sys

# Install dependencies
try:
    import gdown
except ImportError:
    print("Installing dependencies...")
    os.system("pip install gdown -q")
    import gdown

# Download and extract CelebA
celeba_path = '/content/celeba'
zip_path = '/content/celeba.zip'

if not os.path.exists(celeba_path) or not os.listdir(celeba_path):
    print("Downloading CelebA dataset...")
    if not os.path.exists(zip_path):
        gdown.download(
            "https://drive.google.com/uc?id=1xJs_8JB0HYXiaAmU8PTG9qbk0WJ2Wo1U",
            zip_path,
            quiet=False
        )

    print("Extracting dataset...")
    import zipfile
    os.makedirs(celeba_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(celeba_path)
    print(f"Extracted to {celeba_path}")
else:
    print(f"CelebA already exists at {celeba_path}")

# Verify dataset
import glob
num_images = len(glob.glob(f"{celeba_path}/**/*.jpg", recursive=True))
print(f"Found {num_images:,} images")

if num_images == 0:
    print("ERROR: No images found. Please check the dataset.")
    sys.exit(1)

# ==================== Import Main Script ====================

# If running in Colab, you'll need to upload bom_vae_celeba_comparison.py
# or copy the entire script here

# For now, we'll just show the import and run command
print("\n" + "="*70)
print("Setup complete! Now run the comparison:")
print("="*70)

print("""
# Option 1: If you uploaded bom_vae_celeba_comparison.py to Colab:
from bom_vae_celeba_comparison import run_comparison

results = run_comparison(
    celeba_path='/content/celeba',
    n_epochs=10,  # Start with 10 epochs for testing
    batch_size=128,  # Use larger batch size on Colab GPU
    output_dir='/content/outputs'
)

# Option 2: Run as command line:
!python bom_vae_celeba_comparison.py \\
    --celeba_path /content/celeba \\
    --n_epochs 10 \\
    --batch_size 128 \\
    --output_dir /content/outputs

# View results:
from IPython.display import Image, display
display(Image('/content/outputs/training_comparison.png'))
display(Image('/content/outputs/pareto_comparison.png'))
display(Image('/content/outputs/reconstructions_comparison.png'))
display(Image('/content/outputs/samples_comparison.png'))
""")

# ==================== Alternative: Inline Full Script ====================

# If you prefer a self-contained Colab notebook, copy the entire content
# of bom_vae_celeba_comparison.py here, then run:

if __name__ == '__main__' and 'google.colab' in sys.modules:
    print("\n" + "="*70)
    print("Starting BOM-VAE comparison on CelebA...")
    print("="*70)

    # Import the comparison function
    # (Assumes bom_vae_celeba_comparison.py is available)
    try:
        from bom_vae_celeba_comparison import run_comparison

        results = run_comparison(
            celeba_path='/content/celeba',
            n_epochs=10,
            batch_size=128,
            output_dir='/content/outputs'
        )

        # Display results
        from IPython.display import Image, display
        print("\n" + "="*70)
        print("Displaying results...")
        print("="*70)

        display(Image('/content/outputs/training_comparison.png'))
        display(Image('/content/outputs/pareto_comparison.png'))
        display(Image('/content/outputs/reconstructions_comparison.png'))
        display(Image('/content/outputs/samples_comparison.png'))

    except ImportError:
        print("\nERROR: bom_vae_celeba_comparison.py not found.")
        print("Please upload it to Colab or copy the code into this notebook.")
