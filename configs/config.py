# configs/config.py
# v12: + LPIPS texture + Consistency Regularization + Data Augmentation

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# Training
EPOCHS = 30
BATCH_SIZE = 256  # A100 can handle more
LEARNING_RATE = 1e-3
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
OUTPUT_DIR = '/content/outputs_bom_v12'
EVAL_SAMPLES = 10000
NUM_TRAVERSE_DIMS = 15

# v12: Enable augmentation
USE_AUGMENTATION = True

# Constraint types
from enum import Enum

class ConstraintType(Enum):
    UPPER = "upper"
    LOWER = "lower"
    BOX = "box"
    BOX_ASYMMETRIC = "box_asymmetric"
    MINIMIZE_SOFT = "min_soft"
    MINIMIZE_HARD = "min_hard"

GOAL_SPECS = {
    # Reconstruction group
    'pixel': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'edge': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'perceptual': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    
    # Core group - structure + texture
    'core_mse': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'core_edge': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'cross': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'texture_contrastive': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'texture_match': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    
    # Latent group + consistency
    'kl': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 50, 'upper': 8000, 'healthy': 2000},
    'cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'weak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.1},
    'core_consistency': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    
    # Health group
    'detail_ratio': {'type': ConstraintType.BOX, 'lower': 0.10, 'upper': 0.50},
    'core_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0},
    'detail_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0},
    'core_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
    'detail_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
}

RECALIBRATION_EPOCHS = []
GROUP_NAMES = ['recon', 'core', 'latent', 'health']
