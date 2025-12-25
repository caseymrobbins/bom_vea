# ==================== SETUP CELL - RUN FIRST ====================
# Install dependencies
!pip install gdown torchmetrics[image] lpips -q

# Download CelebA
import gdown
import zipfile
import os

zip_path = '/content/img_align_celeba.zip'
celeba_path = '/content/celeba'
output_dir = '/content/outputs'

# Download if not exists
if not os.path.exists(zip_path):
    print("Downloading CelebA...")
    gdown.download(
        "https://drive.google.com/uc?id=1xJs_8JB0HYXiaAmU8PTG9qbk0WJ2Wo1U",
        zip_path,
        quiet=False
    )
else:
    print(f"Zip already exists: {zip_path}")

# Unzip if not extracted
if not os.path.exists(celeba_path) or not os.listdir(celeba_path):
    print("Extracting...")
    os.makedirs(celeba_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(celeba_path)
    print("Done!")
else:
    print(f"Already extracted: {celeba_path}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Verify
import glob
num_images = len(glob.glob(f"{celeba_path}/**/*.jpg", recursive=True))
print(f"Found {num_images:,} images")
