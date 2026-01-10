#!/usr/bin/env python3
"""
single.py - Quick experimentation CLI for BOM VAE

Usage examples:
  python single.py --kl-ceiling 20000 --lr 4e-3 --epochs 10
  python single.py --capacity-scale 0.5 --detail-var-upper 1500
  python single.py --lr 5e-3 --lr-d 5e-4 --squeeze-rate 0.93
"""

import argparse
import sys
import os

# Parse arguments before importing config (so we can override values)
parser = argparse.ArgumentParser(description='Quick BOM VAE experimentation with CLI overrides')

# Learning rates
parser.add_argument('--lr', type=float, help='Main learning rate (default: from config)')
parser.add_argument('--lr-d', type=float, help='Discriminator learning rate (default: from config)')

# Training
parser.add_argument('--epochs', type=int, help='Number of epochs (default: from config)')
parser.add_argument('--batch-size', type=int, help='Batch size (default: from config)')

# KL squeeze
parser.add_argument('--kl-ceiling', type=int, help='Initial KL upper bound for epoch 2 (overrides schedule)')
parser.add_argument('--kl-target', type=int, help='Target KL value (default: 3000)')

# Capacity constraints
parser.add_argument('--capacity-scale', type=float, help='Capacity constraint scale (default: 0.4)')

# Variance bounds
parser.add_argument('--core-var-upper', type=int, help='Core variance health upper bound (default: 1200)')
parser.add_argument('--detail-var-upper', type=int, help='Detail variance health upper bound (default: 1200)')

# Adaptive squeeze
parser.add_argument('--squeeze-rate', type=float, help='Adaptive tightening rate (default: 0.95 = 5%%)')
parser.add_argument('--squeeze-start', type=int, help='Epoch to start adaptive squeeze (default: 6)')

# Appearance constraint
parser.add_argument('--appearance-upper', type=float, help='Appearance error upper bound (default: 0.15)')

# Output
parser.add_argument('--output-suffix', type=str, help='Suffix for output directory (e.g., "_test1")')

args = parser.parse_args()

# Import config after parsing args
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import config

# Apply CLI overrides to config module
if args.lr is not None:
    config.LEARNING_RATE = args.lr
    print(f"Override: LEARNING_RATE = {args.lr}")

if args.lr_d is not None:
    config.LEARNING_RATE_D = args.lr_d
    print(f"Override: LEARNING_RATE_D = {args.lr_d}")

if args.epochs is not None:
    config.EPOCHS = args.epochs
    print(f"Override: EPOCHS = {args.epochs}")

if args.batch_size is not None:
    config.BATCH_SIZE = args.batch_size
    print(f"Override: BATCH_SIZE = {args.batch_size}")

if args.kl_ceiling is not None:
    # Override the KL squeeze schedule to use custom ceiling
    for epoch_num in range(2, 16):
        if epoch_num in config.KL_SQUEEZE_SCHEDULE:
            config.KL_SQUEEZE_SCHEDULE[epoch_num] = args.kl_ceiling
    print(f"Override: KL_SQUEEZE_SCHEDULE (epoch 2-15) = {args.kl_ceiling}")

if args.kl_target is not None:
    config.GOAL_SPECS['kl_core']['healthy'] = float(args.kl_target)
    config.GOAL_SPECS['kl_detail']['healthy'] = float(args.kl_target)
    print(f"Override: KL healthy target = {args.kl_target}")

if args.capacity_scale is not None:
    config.GOAL_SPECS['core_active']['scale'] = args.capacity_scale
    config.GOAL_SPECS['detail_active']['scale'] = args.capacity_scale
    config.GOAL_SPECS['core_effective']['scale'] = args.capacity_scale
    config.GOAL_SPECS['detail_effective']['scale'] = args.capacity_scale
    print(f"Override: Capacity scales = {args.capacity_scale}")

if args.core_var_upper is not None:
    config.GOAL_SPECS['core_var_health']['upper'] = float(args.core_var_upper)
    print(f"Override: Core variance upper = {args.core_var_upper}")

if args.detail_var_upper is not None:
    config.GOAL_SPECS['detail_var_health']['upper'] = float(args.detail_var_upper)
    print(f"Override: Detail variance upper = {args.detail_var_upper}")

if args.squeeze_rate is not None:
    config.ADAPTIVE_TIGHTENING_RATE = args.squeeze_rate
    print(f"Override: ADAPTIVE_TIGHTENING_RATE = {args.squeeze_rate} ({(1-args.squeeze_rate)*100:.0f}% squeeze)")

if args.squeeze_start is not None:
    config.ADAPTIVE_TIGHTENING_START = args.squeeze_start
    print(f"Override: ADAPTIVE_TIGHTENING_START = {args.squeeze_start}")

if args.appearance_upper is not None:
    config.GOAL_SPECS['swap_appearance']['upper'] = args.appearance_upper
    print(f"Override: Appearance upper bound = {args.appearance_upper}")

if args.output_suffix is not None:
    config.OUTPUT_DIR = config.OUTPUT_DIR.rstrip('/') + args.output_suffix
    print(f"Override: OUTPUT_DIR = {config.OUTPUT_DIR}")

print("\n" + "="*60)
print("Starting training with overrides applied...")
print("="*60 + "\n")

# Now import and run the training script
# This imports train.py which will use the modified config values
import train
