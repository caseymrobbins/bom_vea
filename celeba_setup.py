#!/usr/bin/env python3
"""
CelebA Dataset Setup Script
Downloads and extracts CelebA dataset for BOM VAE training
"""
import os
import zipfile
import subprocess
import sys
import glob

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])

def download_celeba():
    """Download CelebA dataset from Google Drive"""
    import gdown

    zip_path = '/content/img_align_celeba.zip'
    celeba_path = '/content/celeba'

    # Download if not exists
    if not os.path.exists(zip_path):
        print("Downloading CelebA dataset (1.3GB)...")
        gdown.download(
            "https://drive.google.com/uc?id=1xJs_8JB0HYXiaAmU8PTG9qbk0WJ2Wo1U",
            zip_path,
            quiet=False
        )
        print("Download complete!")
    else:
        print(f"Zip already exists: {zip_path}")

    # Unzip if not extracted
    if not os.path.exists(celeba_path) or not os.listdir(celeba_path):
        print("Extracting dataset...")
        os.makedirs(celeba_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(celeba_path)
        print("Extraction complete!")
    else:
        print(f"Already extracted: {celeba_path}")

    # Verify
    num_images = len(glob.glob(f"{celeba_path}/**/*.jpg", recursive=True))
    print(f"✓ Found {num_images:,} images in {celeba_path}")

    return celeba_path

def create_output_dir():
    """Create output directory"""
    output_dir = '/content/outputs_bom_v15'
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    return output_dir

if __name__ == "__main__":
    print("=" * 60)
    print("CelebA Dataset Setup")
    print("=" * 60)

    install_dependencies()
    celeba_path = download_celeba()
    output_dir = create_output_dir()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"Dataset: {celeba_path}")
    print(f"Outputs: {output_dir}")
    print("=" * 60)
