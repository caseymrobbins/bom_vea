# CelebA BOM-VAE Comparison Usage Guide

This guide explains how to run the BOM-VAE vs β-VAE comparison on the CelebA dataset.

## Overview

The script `bom_vae_celeba_comparison.py` adapts the working MNIST BOM-VAE code to CelebA. It compares:
- **β-VAE** with multiple β values (0.0001, 0.001, 0.01, 0.1)
- **BOM-VAE** with adaptive squeeze scheduling (no hyperparameter tuning required)

## Key Changes from MNIST

1. **Dataset**: CelebA instead of MNIST
2. **Images**: RGB 64x64 (no need for PadChannels)
3. **Transform**: CenterCrop(178) → Resize(64) for CelebA's 218x178 images
4. **Architecture**: Same VAE (already supports 3 channels)

## Prerequisites

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn
```

## Download CelebA

You can use the included `celeba_setup.py` for Colab, or download manually:

```python
# In Colab
!pip install gdown
import gdown
gdown.download("https://drive.google.com/uc?id=1xJs_8JB0HYXiaAmU8PTG9qbk0WJ2Wo1U", "celeba.zip")
!unzip celeba.zip -d ./data/celeba
```

## Usage

### Basic Usage

```bash
python bom_vae_celeba_comparison.py --celeba_path /path/to/celeba
```

The `celeba_path` should point to the directory containing the `img_align_celeba` folder.

### Full Options

```bash
python bom_vae_celeba_comparison.py \
    --celeba_path /path/to/celeba \
    --n_epochs 20 \
    --batch_size 64 \
    --output_dir ./outputs
```

### In Colab

```python
# After downloading CelebA (see celeba_setup.py)
!python bom_vae_celeba_comparison.py \
    --celeba_path /content/celeba \
    --n_epochs 20 \
    --batch_size 128 \
    --output_dir /content/outputs
```

## Expected Output

The script will:

1. **Train 4 β-VAE models** (with β = 0.0001, 0.001, 0.01, 0.1)
2. **Train 1 BOM-VAE model** (with adaptive squeeze)
3. **Evaluate** all models on test set
4. **Generate visualizations**:
   - `training_comparison.png` - Training curves (MSE, KL, sharpness)
   - `pareto_comparison.png` - MSE vs KL Pareto plot
   - `reconstructions_comparison.png` - Reconstruction quality
   - `samples_comparison.png` - Samples from prior

## Results Summary

The script prints a final comparison table:

```
Method               MSE         KL    Sharpness
------------------------------------------------------
beta_0.0001       0.0234      12.3       0.0456
beta_0.001        0.0245      45.1       0.0489
beta_0.01         0.0289      89.2       0.0523
beta_0.1          0.0412     134.5       0.0567
bom               0.0251      78.6       0.0512
------------------------------------------------------
```

## BOM-VAE Key Features

### Adaptive Squeeze Rule

```
squeeze_amount = (s_min - 0.5) * k
```

- When `s_min = 0.9`: squeeze aggressively
- When `s_min = 0.55`: squeeze gently
- When `s_min ≤ 0.5`: stop squeezing

### No Hyperparameter Tuning

BOM-VAE automatically:
1. **Calibrates** initial constraints from untrained model
2. **Squeezes** constraints toward targets as training progresses
3. **Balances** all objectives without manual β tuning

### Three Objectives

1. **MSE** (lower is better) - Reconstruction quality
2. **KL** (box constraint: 50-80-150) - Latent regularization
3. **Sharpness** (higher is better) - Image detail

## Understanding the Results

### Training Curves

- **MSE curve**: Should decrease over time (better reconstruction)
- **KL curve**: BOM-VAE maintains moderate KL (50-150), β-VAE varies widely
- **Sharpness curve**: Higher is better (sharper details)

### Pareto Plot

- Shows MSE vs KL tradeoff
- **β-VAE**: Different β values span different regions
- **BOM-VAE**: Automatically finds balanced point

### Reconstructions

- Compare visual quality across methods
- BOM-VAE should maintain sharp details while avoiding KL collapse

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python bom_vae_celeba_comparison.py --celeba_path /path/to/celeba --batch_size 32
```

### Dataset Not Found

Verify path structure:
```
/path/to/celeba/
  └── img_align_celeba/
      ├── 000001.jpg
      ├── 000002.jpg
      └── ...
```

### Slow Training

- Use GPU: The script auto-detects CUDA
- Reduce epochs for quick test: `--n_epochs 5`
- Reduce number of β values in code if needed

## Next Steps

After running the comparison:

1. **Analyze results**: Check which β-VAE performs best and compare to BOM-VAE
2. **Tune targets**: Adjust KL targets in `train_bom_vae()` if needed
3. **Extend comparison**: Add more metrics (LPIPS, FID, etc.)
4. **Train longer**: Increase epochs for better convergence

## References

- Original BOM paper/implementation
- β-VAE: Higgins et al. (2017)
- CelebA: Liu et al. (2015)
