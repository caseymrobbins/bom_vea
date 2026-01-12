# configs/config.py
# v17: "Lazy Optimizer" design - clear targets and asymmetric squeeze
# NEW: KL upper bounds with aggressive→gentle squeeze, appearance hard requirement, relaxed capacity

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# GPU Optimizations
USE_TORCH_COMPILE = False  # DISABLED: Causes inplace operation errors during backward pass with LBO rollback mechanism

# Training
EPOCHS = 25  # v17: Target KL=3k by epoch 15
BATCH_SIZE = 512  # A100: 40GB VRAM (L4 used 256 with 24GB)
LEARNING_RATE = 1e-3  # Reduced from 2e-3 to stabilize gradients with small geometric mean terms
LEARNING_RATE_D = 2e-4  # Discriminator learning rate (10x slower than main)
WEIGHT_DECAY = 1e-5
CALIBRATION_BATCHES = 79  # 20% of 395 batches for CelebA
MAX_GRAD_NORM = 1.0
MIN_GROUP_GRAD_THRESHOLD = 1e-6  # Skip backward if S_min too small (avoids non-finite grads)

# Model
LATENT_DIM = 128
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

# Data
DATASET_NAME = 'celeba'
DATA_PATH = '/content/celeba'
ZIP_PATH = '/content/img_align_celeba.zip'

# Output
OUTPUT_DIR = '/content/outputs_bom_v17'  # v17: Lazy optimizer design with KL squeeze
EVAL_SAMPLES = 10000
NUM_TRAVERSE_DIMS = 15

USE_AUGMENTATION = True
DEBUG_RAW_NORMALIZED = True  # v16: Print raw and normalized values once per epoch for debugging

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
    # v17e: Keep appearance as MINIMIZE_SOFT (BOX_ASYMMETRIC caused 100% rollbacks)
    # Detail usage will be enforced by: (1) KL squeeze forcing latent usage, (2) capacity constraints
    'swap_appearance': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # colors(r_sw) ≈ colors(x2)
    'swap_color_hist': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}, # histogram(r_sw) ≈ histogram(x2)

    # v14: Realism group - discriminator goals
    # Fixed scale=2.0: Discriminator untrained during calibration (D≈0.5), but learns quickly
    # After training starts, D(recon) can reach 0.99. Need large scale to handle this range.
    'realism_recon': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 2.0},  # D should classify recon as real
    'realism_swap': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 2.0},   # D should classify swap as real

    # Behavioral walls (what core/detail actually DO)
    'core_color_leak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # Δz_core shouldn't change colors
    'detail_edge_leak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}, # Δz_detail shouldn't change edges

    # Latent group - KL and statistical health
    # DISCOVERY STRATEGY: No upper cap until epoch 3
    # Epoch 1-2: NO upper constraint (healthy=1e8, upper=1e9 → effectively unlimited)
    #            Lower=100 prevents KL collapse, but no pull toward any target value
    # Epoch 3+: Apply KL_SQUEEZE_SCHEDULE to squeeze from discovered ceiling → 3000
    # Upper bounds prevent "high KL collapse" (all inputs → same point far from prior)
    # Lower bounds prevent "low KL collapse" (ignore latent space)
    'kl_core': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 100.0, 'upper': 1e9, 'healthy': 1e8, 'lower_scale': 2.0},
    'kl_detail': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 100.0, 'upper': 1e9, 'healthy': 1e8, 'lower_scale': 2.0},

    # Direct logvar constraints to prevent exp(logvar) explosion
    # logvar∈[-15,10] → std∈[0.0003, 148] → prevents numerical overflow
    # Lower bound widened to -15 to contain calibration phase values
    'logvar_core': {'type': ConstraintType.BOX, 'lower': -15.0, 'upper': 10.0},
    'logvar_detail': {'type': ConstraintType.BOX, 'lower': -15.0, 'upper': 10.0},

    'cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'weak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.1},
    # Consistency: Use large fixed scale to handle augmentation-induced variance
    # Auto-calibration sees p95≈115 but training reaches 200-250 due to augmentation strength
    'core_consistency': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 500.0},

    # v15: Dimension capacity utilization (inactive/ineffective ratios - minimize these)
    # v17 FIX: Relaxed from 0.3 to 0.4 (70%→60% active) to reduce conflict with variance constraints
    # Scale = 0.4 means: inactive_ratio < 0.28 to get score > 0.5 (requires 46+/64 dims active)
    # Still enforces capacity, but gives more breathing room for detail variance
    'core_active': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.4},
    'detail_active': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.4},
    'core_effective': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.4},
    'detail_effective': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.4},

    # v14: Detail contracts - WIDE initial bounds for feasible initialization
    # v17b: Widened bounds (observed violations: detail_mean=21.32, detail_var_mean=354.0)
    'detail_mean': {'type': ConstraintType.BOX, 'lower': -30.0, 'upper': 30.0},  # Was [-20, 20], violated by 21.32
    # detail_var_mean: Changed BOX→LOWER to allow initialization at ~0
    # At init, variance is near 0 - only enforce it stays above -1.0 (effectively non-negative)
    # No upper bound needed - let it grow naturally during training
    'detail_var_mean': {'type': ConstraintType.LOWER, 'lower': -1.0, 'margin': 10.0},
    'detail_cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # v17f: Fixed scale=1.0 → auto (calibration saw 4.9-30.7)
    'traversal': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},

    # Health group - v17: WIDER variance bounds to allow latent space spreading
    # detail_ratio: Changed from BOX to MINIMIZE_SOFT - we don't need upper bound
    # Raw values reach 0.97-0.99, BOX upper=1.0 causes gradient explosion at boundary
    'detail_ratio': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    # core/detail_var_health: Changed BOX→LOWER to allow initialization at ~0 variance
    # At init, median variance ≈ 0 - only enforce non-negative with breathing room
    # No upper bound needed - let variance grow naturally during training
    'core_var_health': {'type': ConstraintType.LOWER, 'lower': -10.0, 'margin': 100.0},
    'detail_var_health': {'type': ConstraintType.LOWER, 'lower': -10.0, 'margin': 100.0},
    'core_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # v17g: Fixed scale=100.0 → auto (calibration saw p95≈494)
    'detail_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},  # v17g: Fixed scale=100.0 → auto (calibration saw p95≈577)
}

# v17d: KL Adaptive Squeeze Schedule (starts epoch 3, after ceiling discovery)
# Epoch 1: NO ceiling (upper=1e9) - discover natural KL
# Epoch 2: Ceiling = max(KL_epoch_1) - set dynamically in train.py
# Epoch 3+: Squeeze from discovered ceiling → 3k (aggressive early, gentle late)
KL_SQUEEZE_SCHEDULE = {
    1: None,      # No ceiling - discovery phase
    2: None,      # Ceiling set dynamically from epoch 1 max
    3: 13000,     # Start squeeze (aggressive -2k if ceiling was ~15k)
    4: 11000,     # -2000
    5: 9500,      # -1500
    6: 8200,      # -1300
    7: 7000,      # -1200
    8: 6000,      # -1000
    9: 5200,      # -800
    10: 4600,     # -600
    11: 4100,     # -500
    12: 3700,     # -400
    13: 3400,     # -300
    14: 3200,     # -200
    15: 3000,     # -200 (target reached!)
}

# LBO Directive #6: Adaptive Squeeze with rollback monitoring
# v17: Simplified to constant 5% squeeze every epoch after convergence (epoch 5)
ADAPTIVE_TIGHTENING_START = 6  # Start after epoch 5 (convergence)
ADAPTIVE_TIGHTENING_RATE = 0.95  # Constant 5% squeeze per epoch (simple and predictable)
ROLLBACK_THRESHOLD_MAX = 0.15  # If rollback rate > 15%, back off immediately
ROLLBACK_THRESHOLD_TARGET = 0.05  # Target: 5% rollback rate (optimal squeeze)
MIN_GROUP_STABILITY_THRESHOLD = 0.50  # Only tighten if avg min_group > 0.5 (Directive #6)
STABILITY_WINDOW = 3  # Check last 3 epochs for stability

GROUP_NAMES = ['recon', 'core', 'swap', 'realism', 'disentangle', 'latent', 'health']
