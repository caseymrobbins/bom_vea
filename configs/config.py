# configs/config.py
# v14: Discriminator + Detail contracts

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# Training
EPOCHS = 30
BATCH_SIZE = 256
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

    # Latent group - BOTH core and detail now have KL!
    'kl_core': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 50, 'upper': 8000, 'healthy': 2000},
    'kl_detail': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 50, 'upper': 8000, 'healthy': 2000},
    'cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'weak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.1},
    'core_consistency': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},

    # v14: Detail contracts - ensure detail channel has proper statistics
    'detail_mean': {'type': ConstraintType.BOX, 'lower': -3.0, 'upper': 3.0},    # Mean should be near 0
    'detail_var_mean': {'type': ConstraintType.BOX, 'lower': 0.1, 'upper': 10.0}, # Variance should be reasonable
    'detail_cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},       # Low covariance

    # Health group
    'detail_ratio': {'type': ConstraintType.BOX, 'lower': 0.10, 'upper': 0.50},
    'core_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0},
    'detail_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0},
    'core_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
    'detail_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
}

RECALIBRATION_EPOCHS = []
GROUP_NAMES = ['recon', 'core', 'swap', 'realism', 'latent', 'health']
