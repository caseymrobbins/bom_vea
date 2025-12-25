# configs/config.py
# All configuration in one place - the ONLY file you should need to edit

import torch

# ==================== DEVICE ====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPU optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ==================== TRAINING ====================
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
CALIBRATION_BATCHES = 200

# ==================== MODEL ====================
LATENT_DIM = 128
IMAGE_SIZE = 64  # Images will be resized to this
IMAGE_CHANNELS = 3

# ==================== DATA ====================
# Set ONE of these - the loader will auto-detect which to use
DATASET_NAME = 'celeba'  # Options: 'celeba', 'cifar10', 'mnist', 'folder', 'auto'
DATA_PATH = '/content/celeba'  # Path to data (folder of images, or will auto-download)
ZIP_PATH = '/content/img_align_celeba.zip'  # Optional: extract from zip first

# For 'folder' mode: expects DATA_PATH/class_name/images.jpg structure
# For 'auto' mode: point DATA_PATH to any folder of images

# ==================== OUTPUT ====================
OUTPUT_DIR = '/content/outputs_bom_v11'
EVAL_SAMPLES = 10000
NUM_TRAVERSE_DIMS = 15

# ==================== BOM GOALS ====================
# These rarely need changing - BOM auto-calibrates scales
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
    
    # Latent group
    'kl': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 50, 'upper': 8000, 'healthy': 2000},
    'cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'},
    'weak': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.1},
    
    # Health group
    'detail_ratio': {'type': ConstraintType.BOX, 'lower': 0.10, 'upper': 0.50},
    'core_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0},
    'detail_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0},
    'core_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
    'detail_var_max': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 100.0},
}

# Recalibration epochs (empty = calibrate only at epoch 1)
RECALIBRATION_EPOCHS = []

# Group names for logging
GROUP_NAMES = ['recon', 'core', 'latent', 'health']
