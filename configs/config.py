# configs/config.py
# v15: Hierarchical latent constraints + dimension utilization enforcement
# NEW: Added 4 capacity goals (core_active, detail_active, core_effective, detail_effective)

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# GPU Optimizations
USE_TORCH_COMPILE = False  # DISABLED: Causes inplace operation errors during backward pass with LBO rollback mechanism

# Training
EPOCHS = 35
BATCH_SIZE = 512  # A100: 40GB VRAM (L4 used 256 with 24GB)
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
OUTPUT_DIR = '/content/outputs_bom_v14'
EVAL_SAMPLES = 10000
NUM_TRAVERSE_DIMS = 15

USE_AUGMENTATION = True

# LBO Directive #1: Must use pure min() - softmin violates LBO by smoothing the barrier

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

    # Behavioral walls (what core/detail actually DO)
    'core_color_leak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # Δz_core shouldn't change colors
    'detail_edge_leak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}, # Δz_detail shouldn't change edges

    # Latent group - KL and statistical health
    # Start with only LOWER bound, add upper bound at epoch 25 once KL stabilizes
    # Margin lowered to 1.0 since ultra-conservative init (logvar.bias=-5.0) gives KL~0.4
    'kl_core': {'type': ConstraintType.LOWER, 'margin': 1.0},
    'kl_detail': {'type': ConstraintType.LOWER, 'margin': 1.0},

    # Direct logvar constraints to prevent exp(logvar) explosion
    # logvar∈[-15,10] → std∈[0.0003, 148] → prevents numerical overflow
    # Lower bound widened to -15 to contain calibration phase values
    'logvar_core': {'type': ConstraintType.BOX, 'lower': -15.0, 'upper': 10.0},
    'logvar_detail': {'type': ConstraintType.BOX, 'lower': -15.0, 'upper': 10.0},

    'cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'weak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.1},
    'core_consistency': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},

    # v15: Dimension capacity utilization (inactive/ineffective ratios - minimize these)
    'core_active': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'detail_active': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'core_effective': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'detail_effective': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},

    # v14: Detail contracts - WIDE initial bounds for feasible initialization
    'detail_mean': {'type': ConstraintType.BOX, 'lower': -15.0, 'upper': 15.0},
    'detail_var_mean': {'type': ConstraintType.BOX, 'lower': 0.0, 'upper': 350.0},  # Allow 0.0 at init
    'detail_cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 1.0},
    'traversal': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},

    # Health group - v16: WIDER bounds to prevent collapse (was 300, now 600)
    'detail_ratio': {'type': ConstraintType.BOX, 'lower': 0.0, 'upper': 0.70},  # Allow exactly 0.0 at init
    'core_var_health': {'type': ConstraintType.BOX, 'lower': 0.0, 'upper': 600.0},  # v16: Doubled to prevent squeeze death
    'detail_var_health': {'type': ConstraintType.BOX, 'lower': 0.0, 'upper': 600.0},  # v16: Doubled to prevent squeeze death
    'core_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
    'detail_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
}

# LBO Directive #6: Adaptive Squeeze with rollback monitoring
# Constitution: "move Failure threshold 10% closer when S_min > 0.5"
# v16 fix: Reduced aggression (10% instead of 5%), added stability check, earlier backoff
ADAPTIVE_TIGHTENING_START = 8  # Start tightening after epoch 7 (more warmup)
ADAPTIVE_TIGHTENING_RATES = [0.90, 0.92, 0.94, 0.96, 0.98]  # 10%, 8%, 6%, 4%, 2% tightening
ROLLBACK_THRESHOLD_MAX = 0.15  # If rollback rate > 15%, back off immediately (was 50% - too late!)
ROLLBACK_THRESHOLD_TARGET = 0.05  # Target: 5% rollback rate (optimal squeeze)
MIN_GROUP_STABILITY_THRESHOLD = 0.50  # Only tighten if avg min_group > 0.5 (Directive #6)
STABILITY_WINDOW = 3  # Check last 3 epochs for stability

GROUP_NAMES = ['recon', 'core', 'swap', 'realism', 'disentangle', 'latent', 'health']
