# BOM VAE v14 - Discriminator + Detail Contracts

## Summary

This version adds:
1. **PatchGAN Discriminator** with spectral normalization
2. **KL divergence for BOTH core and detail channels** (critical fix - v13 only had KL for core)
3. **Detail channel contracts**: mean, variance, covariance constraints
4. **Aggressive BOX constraint verification** to prevent crashes

## Key Changes

### New Files
- `models/discriminator.py`: PatchGAN discriminator with spectral norm

### Modified Files

#### `configs/config.py`
- Added `LEARNING_RATE_D = 1e-4` for discriminator
- Updated `OUTPUT_DIR` to `/content/outputs_bom_v14`
- Renamed `kl` → `kl_core` and added `kl_detail` (both BOX_ASYMMETRIC)
- Added realism goals: `realism_recon`, `realism_swap` (MINIMIZE_SOFT)
- Added detail contracts: `detail_mean` (BOX), `detail_var_mean` (BOX), `detail_cov` (MINIMIZE_SOFT)
- Updated `GROUP_NAMES` to include 'realism': `['recon', 'core', 'swap', 'realism', 'latent', 'health']`

#### `models/vae.py`
- **CRITICAL**: Added `mu` clamping in `encode()`: `torch.clamp(self.mu(h), -10, 10)` to prevent explosion

#### `losses/bom_loss.py`
- Updated `compute_raw_losses()`:
  - Added `discriminator` parameter
  - Added `logvar_detail` split
  - Compute `kl_core` and `kl_detail` separately
  - Compute detail contracts: `detail_mean`, `detail_var_mean`, `detail_cov`
  - Compute discriminator scores: `realism_recon`, `realism_swap`

- Updated `grouped_bom_loss()`:
  - Added `discriminator` parameter
  - Added `logvar_detail` split
  - New GROUP D: REALISM with discriminator goals
  - Updated GROUP E: LATENT with separate KL for core and detail
  - Added detail contract goals to latent group
  - Updated group stack: `[group_recon, group_core, group_swap, group_realism, group_latent, group_health]`

#### `losses/goals.py`
- **CRITICAL**: Added BOX constraint verification in `calibrate()`:
  - Prints actual initialization ranges vs BOX bounds
  - Warns if initial values are outside BOX (will cause crash)
  - Example: `detail_mean: BOX [-3.00, 3.00] | init=[-0.12, 0.15] median=0.02`

#### `train.py`
- Import `create_discriminator`
- Create discriminator and optimizer_d
- Add discriminator training loop (every other step):
  - Train D to classify real images as 1
  - Train D to classify fake (recon) as 0
  - Uses BCE loss with logits
- Pass `discriminator` to `compute_raw_losses()` and `grouped_bom_loss()`
- Update history tracking for new metrics
- Save discriminator state in checkpoint
- Print KL_core and KL_detail in epoch summary

## BOM Philosophy Compliance

✅ **No epsilon in log()**: Loss is `-torch.log(min_group)` with no softening
✅ **BOX constraints contain initialization**: Verification system warns if violated
✅ **Print raw values**: Calibration shows actual ranges vs constraints
✅ **mu clamping**: Prevents encoder explosion
✅ **Detail channel KL**: Both channels now have KL divergence

## What to Expect

### During Calibration (first 200 batches of epoch 1):
```
==============================================================
CALIBRATING GOALS (#1, epoch 1)
==============================================================
  pixel               : scale=0.0523
  edge                : scale=0.0089
  ...
  detail_mean         : BOX [-3.00, 3.00] | init=[-0.05, 0.08] median=0.01 ✓
  detail_var_mean     : BOX [0.10, 10.00] | init=[1.23, 2.45] median=1.67 ✓
  kl_core             : BOX_ASYM [50, 8000] h=2000 | init=[1850, 2150] median=2000 ✓
  kl_detail           : BOX_ASYM [50, 8000] h=2000 | init=[1823, 2189] median=1998 ✓
==============================================================
```

### If BOX violation occurs:
```
⚠️  WARNING: BOX CONSTRAINT VIOLATIONS ⚠️
    detail_var_mean: init range [15.23, 51.45] outside BOX [0.10, 10.00]
    These constraints will return goal=0 → loss=inf → crash!
    ACTION: Widen BOX bounds to contain initialization values.
```

**ACTION**: Update `configs/config.py` BOX bounds to be wider, then restart training.

### During Training:
```
Epoch  1 | Loss: 1.523 | Min: 0.217 | SSIM: 0.832
         Structure: 0.0123 | Appearance: 0.0456
         KL_core: 2050.3 | KL_detail: 1987.1
         Groups: recon:0.65 | core:0.58 | swap:0.47 | realism:0.42 | latent:0.52 | health:0.61
```

## Testing Checklist

Before running full training:

1. **Verify imports**: `from models.discriminator import create_discriminator`
2. **Check BOX bounds**: Run 1 epoch and check calibration output for violations
3. **Monitor groups**: All groups should be 0.4-0.8 range, none hitting 0
4. **Watch KL values**: Both core and detail should be in [50, 8000] range
5. **Check detail contracts**:
   - `detail_mean_raw` should approach 0
   - `detail_var_mean_raw` should be ~1
   - `detail_cov_raw` should be low

## Known Potential Issues

1. **Detail contracts too tight**: If `detail_var_mean` BOX [0.1, 10.0] is violated, widen to [0.01, 50.0]
2. **Discriminator too strong**: If realism group always bottleneck, reduce LEARNING_RATE_D
3. **KL explosion**: If KL > 8000, check mu clamping is working (should be in vae.py:65)

## Next Steps (v15)

Planned improvements:
- Attention layers in encoder/decoder
- Skip connections for better reconstruction
- Higher resolution (128/256)
- 50 epochs instead of 30
