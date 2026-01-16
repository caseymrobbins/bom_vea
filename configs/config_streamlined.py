# configs/config_streamlined.py
# STREAMLINED VERSION - Reduced from 35 goals to 9 for cleaner gradient flow
# Priority: Useful latents > Perfect reconstruction

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    # Use new TF32 API (PyTorch 2.9+)
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'

# GPU Optimizations
USE_TORCH_COMPILE = False  # DISABLED: Causes inplace operation errors during backward pass with LBO rollback mechanism

# Training
EPOCHS = 25
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
LEARNING_RATE_D = 2e-4
WEIGHT_DECAY = 1e-5
CALIBRATION_BATCHES = 79
MAX_GRAD_NORM = 1.0
MIN_GROUP_GRAD_THRESHOLD = 1e-6

# Model
LATENT_DIM = 128
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
PRIOR_BLOCK_SIZE = 8
PRIOR_INTRA_CORR = 0.2

# Data
DATASET_NAME = 'celeba'
DATA_PATH = '/content/celeba'
ZIP_PATH = '/content/img_align_celeba.zip'

# Output
OUTPUT_DIR = '/content/outputs_bom_streamlined'
EVAL_SAMPLES = 10000
NUM_TRAVERSE_DIMS = 15

USE_AUGMENTATION = True
DEBUG_RAW_NORMALIZED = True

from enum import Enum

class ConstraintType(Enum):
    UPPER = "upper"
    LOWER = "lower"
    BOX = "box"
    BOX_ASYMMETRIC = "box_asymmetric"
    MINIMIZE_SOFT = "min_soft"
    MINIMIZE_HARD = "min_hard"

# ========================================
# STREAMLINED GOALS (9 total, down from 35)
# ========================================
# Organized into 5 groups for clearer optimization

GOAL_SPECS = {
    # ===== GROUP 1: LATENT QUALITY (5 goals) =====
    # Priority: Useful, disentangled latents

    # 1. KL divergence - Single merged KL budget
    # Combines: kl_core + kl_detail + prior_kl → unified KL target
    # Adaptive squeeze: 15k → 3k over epochs 3-15
    'kl_divergence': {
        'type': ConstraintType.BOX_ASYMMETRIC,
        'lower': 0.0,
        'upper': 1e9,  # Set dynamically by squeeze schedule
        'healthy': 2e8,
        'lower_scale': 2.0
    },

    # 2. Disentanglement - TC discriminator penalty
    # Combines: sep_core + sep_mid + sep_detail → single TC score
    'disentanglement': {
        'type': ConstraintType.MINIMIZE_SOFT,
        'scale': 'auto'
    },

    # 3. Capacity - Latent utilization
    # Combines: core_active + detail_active + core_effective + detail_effective
    # Ensures latent dimensions are actually used (not collapsed)
    'capacity': {
        'type': ConstraintType.MINIMIZE_SOFT,
        'scale': 0.4  # Allows 60% active dims (relaxed from 70%)
    },

    # 4. Behavioral separation - Structure vs Appearance
    # Combines: core_color_leak + detail_edge_leak
    # Core affects structure not color, detail affects color not structure
    'behavioral_separation': {
        'type': ConstraintType.MINIMIZE_SOFT,
        'scale': 'auto'
    },

    # 5. Latent stats - Statistical health
    # Combines: logvar_core + logvar_detail + cov + weak
    # Prevents variance explosion and dimension collapse
    'latent_stats': {
        'type': ConstraintType.MINIMIZE_SOFT,
        'scale': 'auto'
    },

    # ===== GROUP 2: RECONSTRUCTION (2 goals) =====

    # 6. Reconstruction - Image quality
    # Combines: pixel + edge + perceptual
    # Weighted: 0.2*pixel + 0.3*edge + 0.5*perceptual (perceptual dominates)
    'reconstruction': {
        'type': ConstraintType.MINIMIZE_SOFT,
        'scale': 'auto'
    },

    # 7. Cross-reconstruction - Swap consistency
    # Combines: swap_structure + swap_appearance + swap_color_hist
    # Tests if latent factorization actually works
    'cross_recon': {
        'type': ConstraintType.MINIMIZE_SOFT,
        'scale': 'auto'
    },

    # ===== GROUP 3: STABILITY (2 goals) =====

    # 8. Realism - GAN adversarial loss
    # Combines: realism_recon + realism_swap
    # Keeps reconstructions realistic (not blurry)
    'realism': {
        'type': ConstraintType.MINIMIZE_SOFT,
        'scale': 2.0  # Fixed: discriminator untrained during calibration
    },

    # 9. Consistency - Augmentation invariance
    # Core structure should be robust to augmentations
    'consistency': {
        'type': ConstraintType.MINIMIZE_SOFT,
        'scale': 500.0  # Large fixed scale for augmentation variance
    },
}

# ========================================
# FLAT STRUCTURE - NO GROUPING
# ========================================
# With only 9 goals, we can use them flat without grouping
# Pure LBO: loss = -log(min(all 9 goals))

GOAL_NAMES = [
    'kl_divergence',
    'disentanglement',
    'capacity',
    'behavioral_separation',
    'latent_stats',
    'reconstruction',
    'cross_recon',
    'realism',
    'consistency',
]

# For backwards compatibility with training code
GROUP_NAMES = GOAL_NAMES  # Flat structure: each "group" is one goal

# ========================================
# ADAPTIVE SQUEEZE & TIGHTENING
# ========================================

# KL Adaptive Squeeze Schedule (starts epoch 3)
KL_SQUEEZE_SCHEDULE = {
    1: None,      # No ceiling - discovery phase
    2: None,      # Ceiling set dynamically from epoch 1 max
    3: 13000,     # Start squeeze
    4: 11000,
    5: 9500,
    6: 8200,
    7: 7000,
    8: 6000,
    9: 5200,
    10: 4600,
    11: 4100,
    12: 3700,
    13: 3400,
    14: 3200,
    15: 3000,     # Target reached
}

# KL lower-bound warmup
KL_LOWER_WARMUP_START = 3
KL_LOWER_WARMUP_END = 10
KL_LOWER_FINAL = 100.0

# Adaptive tightening
ADAPTIVE_TIGHTENING_START = 6
ADAPTIVE_TIGHTENING_RATE = 0.95
ROLLBACK_THRESHOLD_MAX = 0.15
ROLLBACK_THRESHOLD_TARGET = 0.05
MIN_GROUP_STABILITY_THRESHOLD = 0.50
STABILITY_WINDOW = 3
