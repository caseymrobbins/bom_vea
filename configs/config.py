# configs/config.py
# v15: Progressive group-by-group tightening (one group every 2 epochs)

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# A100 Optimizations
USE_TORCH_COMPILE = True  # PyTorch 2.0+: Significant speedup on A100 (set False if PyTorch < 2.0)

# Training
EPOCHS = 30
BATCH_SIZE = 512  # A100: Increased from 256 (40GB can handle 512-1024)
LEARNING_RATE = 1e-3
LEARNING_RATE_D = 1e-4  # Discriminator learning rate (slower)
WEIGHT_DECAY = 1e-5
CALIBRATION_BATCHES = 200

# Model
LATENT_DIM = 128
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

# Data
DATASET_NAME = 'celeba'
DATA_PATH = '/content/celeba'
ZIP_PATH = '/content/img_align_celeba.zip'

# Output
OUTPUT_DIR = '/content/outputs_bom_v15'
EVAL_SAMPLES = 10000
NUM_TRAVERSE_DIMS = 15

USE_AUGMENTATION = True

# A/B Testing: Softmin vs Hard Min
USE_SOFTMIN = False  # True = softmin, False = hard min (SOFTMIN UNSTABLE - using hard min)
SOFTMIN_TEMPERATURE = 0.1  # Lower = closer to hard min, higher = smoother

from enum import Enum

class ConstraintType(Enum):
    UPPER = "upper"
    LOWER = "lower"
    BOX = "box"
    BOX_ASYMMETRIC = "box_asymmetric"
    MINIMIZE_SOFT = "min_soft"
    MINIMIZE_HARD = "min_hard"

GOAL_SPECS = {
    # Reconstruction group - full image quality
    'pixel': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'edge': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'perceptual': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},

    # Core group - structure preservation
    'core_mse': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'core_edge': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},

    # Swap group - structure from x1, appearance from x2
    'swap_structure': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # edges(r_sw) ≈ edges(x1)
    'swap_appearance': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}, # colors(r_sw) ≈ colors(x2)
    'swap_color_hist': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}, # histogram(r_sw) ≈ histogram(x2)

    # v14: Realism group - discriminator goals
    'realism_recon': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # D should classify recon as real
    'realism_swap': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},   # D should classify swap as real

    # NEW: Disentanglement group - behavioral walls (what core/detail actually DO)
    'core_color_leak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # Δz_core shouldn't change colors
    'detail_edge_leak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}, # Δz_detail shouldn't change edges

    # Latent group - KL and statistical health
    # Start with only LOWER bound, add upper bound at epoch 25 once KL stabilizes
    'kl_core': {'type': ConstraintType.LOWER, 'lower': 10},
    'kl_detail': {'type': ConstraintType.LOWER, 'lower': 10},
    'cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'weak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.1},
    'core_consistency': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},

    # v14: Detail contracts - WIDE initial bounds for feasible initialization
    'detail_mean': {'type': ConstraintType.BOX, 'lower': -15.0, 'upper': 15.0},
    'detail_var_mean': {'type': ConstraintType.BOX, 'lower': 0.01, 'upper': 300.0},
    'detail_cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 1.0},

    # Health group - WIDE initial bounds, tighten at epoch 27
    'detail_ratio': {'type': ConstraintType.BOX, 'lower': 0.001, 'upper': 0.70},
    'core_var_health': {'type': ConstraintType.BOX, 'lower': 0.01, 'upper': 300.0},
    'detail_var_health': {'type': ConstraintType.BOX, 'lower': 0.01, 'upper': 300.0},
    'core_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
    'detail_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
}

# Progressive tightening: one group per epoch, spaced 2 epochs apart
# BOM will focus on each group for 1-2 epochs as it becomes the bottleneck
TIGHTENING_SCHEDULE = {
    15: 'recon',      # Epoch 15: tighten reconstruction
    17: 'core',       # Epoch 17: tighten core structure
    19: 'swap',       # Epoch 19: tighten swap goals
    21: 'realism',    # Epoch 21: tighten discriminator goals
    23: 'disentangle',# Epoch 23: tighten behavioral walls
    25: 'latent',     # Epoch 25: tighten KL and statistics
    27: 'health',     # Epoch 27: tighten variance/ratio health
}

RECALIBRATION_EPOCHS = list(TIGHTENING_SCHEDULE.keys())  # [15, 17, 19, 21, 23, 25, 27]
GROUP_NAMES = ['recon', 'core', 'swap', 'realism', 'disentangle', 'latent', 'health']
