# BOM VAE v11 - Data Agnostic

**Bottleneck Optimization Method** applied to VAE with structure/texture disentanglement.

## Quick Start

```bash
# 1. Edit config (optional - defaults work)
# nano configs/config.py

# 2. Run training
python train.py
```

That's it. No hyperparameter tuning needed.

## Structure

```
bom_vae_v11/
├── train.py              # Main script - run this
├── configs/
│   └── config.py         # All settings in one place
├── models/
│   ├── vae.py            # ConvVAE with PixelShuffle decoder
│   └── vgg.py            # VGG features for perceptual/texture
├── losses/
│   ├── goals.py          # BOM goal system with auto-calibration
│   └── bom_loss.py       # Grouped BOM loss computation
└── utils/
    ├── data.py           # Universal data loader
    └── viz.py            # Visualization utilities
```

## Supported Datasets

The data loader auto-handles:

- **CelebA**: `DATASET_NAME = 'celeba'`
- **CIFAR-10/100**: `DATASET_NAME = 'cifar10'` or `'cifar100'`
- **MNIST/Fashion-MNIST**: `DATASET_NAME = 'mnist'` or `'fashion_mnist'`
- **Any folder of images**: `DATASET_NAME = 'folder'`, point `DATA_PATH` to folder
- **Auto-detect**: `DATASET_NAME = 'auto'`

## Config Options

Edit `configs/config.py`:

```python
# Data
DATASET_NAME = 'celeba'      # or 'cifar10', 'mnist', 'folder', 'auto'
DATA_PATH = '/path/to/data'

# Training
EPOCHS = 30                   # Usually enough - converges fast
BATCH_SIZE = 128
LATENT_DIM = 128              # 64 core + 64 detail

# Output
OUTPUT_DIR = '/content/outputs_bom_v11'
```

## BOM Goals

The system auto-calibrates 16 goals across 4 groups:

| Group | Goals | Purpose |
|-------|-------|---------|
| **recon** | pixel, edge, perceptual | Image reconstruction quality |
| **core** | core_mse, core_edge, cross, texture_contrastive, texture_match | Structure preservation + texture transfer |
| **latent** | kl, cov, weak | Latent space health |
| **health** | detail_ratio, variance bounds | Prevent degenerate solutions |

BOM maximizes `log(min(goals))` - the worst goal drives learning.

## Key Features

1. **Auto-calibration**: Scale factors computed from data, not hand-tuned
2. **No recalibration**: Calibrate once at epoch 1, stable targets thereafter
3. **Texture fix**: Both contrastive (relative) and match (absolute) goals prevent gaming
4. **Variance control**: BOX constraints prevent collapsed or exploding dimensions
5. **PixelShuffle decoder**: No checkerboard artifacts

## Outputs

Training produces:
- `bom_vae_v11.pt` - Model checkpoint
- `group_balance.png` - BOM group convergence
- `goal_details.png` - Individual goal metrics
- `reconstructions.png` - Original/Full/Core/Detail comparison
- `traversals_core.png` - Core dimension traversals
- `traversals_detail.png` - Detail dimension traversals
- `cross_reconstruction.png` - x1_core + x2_detail mixing
- `dimension_activity.png` - Latent dimension usage
- `training_history.png` - Loss/SSIM/KL curves

## Evaluation Metrics

Final metrics computed:
- MSE, SSIM, LPIPS, FID

## Requirements

```
torch
torchvision
torchmetrics[image]
matplotlib
numpy
tqdm
Pillow
```
