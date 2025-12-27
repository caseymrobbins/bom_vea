# BOM VAE v15 - Loosened Constraints

**Date**: 2025-12-27

## Changes from v14

v14 introduced discriminator + detail contracts but had overly tight constraints causing:
- Core channel collapse (traversals showed no variation)
- Over-compression (KL_detail=572 vs KL_core=1942)
- Reconstruction artifacts (discriminator too aggressive)

v15 loosens all constraints to allow more model freedom while maintaining BOM structure.

## Specific Changes

### 1. Discriminator Constraints (Realism Group)
```python
# v14
'realism_recon': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}
'realism_swap': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}

# v15
'realism_recon': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.3}  # 70% reduction
'realism_swap': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.3}
```
**Impact**: Less aggressive discriminator, fewer artifacts

### 2. KL Divergence (Latent Group)
```python
# v14
'kl_core': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 50, 'upper': 8000, 'healthy': 2000}
'kl_detail': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 50, 'upper': 8000, 'healthy': 2000}

# v15
'kl_core': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 10, 'upper': 15000, 'healthy': 3000}
'kl_detail': {'type': ConstraintType.BOX_ASYMMETRIC, 'lower': 10, 'upper': 15000, 'healthy': 3000}
```
**Impact**: Much wider latent space, prevents core collapse, allows higher KL values

### 3. Detail Contracts
```python
# v14
'detail_mean': {'type': ConstraintType.BOX, 'lower': -3.0, 'upper': 3.0}
'detail_var_mean': {'type': ConstraintType.BOX, 'lower': 0.1, 'upper': 10.0}
'detail_cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 'auto'}

# v15
'detail_mean': {'type': ConstraintType.BOX, 'lower': -5.0, 'upper': 5.0}
'detail_var_mean': {'type': ConstraintType.BOX, 'lower': 0.01, 'upper': 20.0}
'detail_cov': {'type': ConstraintType.MINIMIZE_SOFT, 'scale': 0.5}
```
**Impact**: More flexible detail statistics, less constraint on mean/variance

### 4. Health Constraints
```python
# v14
'detail_ratio': {'type': ConstraintType.BOX, 'lower': 0.10, 'upper': 0.50}
'core_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0}
'detail_var_health': {'type': ConstraintType.BOX, 'lower': 0.5, 'upper': 50.0}

# v15
'detail_ratio': {'type': ConstraintType.BOX, 'lower': 0.05, 'upper': 0.60}
'core_var_health': {'type': ConstraintType.BOX, 'lower': 0.1, 'upper': 100.0}
'detail_var_health': {'type': ConstraintType.BOX, 'lower': 0.1, 'upper': 100.0}
```
**Impact**: More flexible channel split, wider variance allowed

### 5. Visualization Fix
```python
# v14 (crashed with KeyError: 'kl_raw')
axs[1,1].plot(histories['kl_raw'])

# v15
axs[1,1].plot(histories['kl_core_raw'], label='core')
axs[1,1].plot(histories['kl_detail_raw'], label='detail')
axs[1,1].legend()
```
**Impact**: No more crash, can see both KL metrics separately

## Expected Results

With loosened constraints, v15 should show:
- ✅ More variation in core channel traversals (no more collapse)
- ✅ Better balance between KL_core and KL_detail
- ✅ Cleaner reconstructions with fewer artifacts
- ✅ All groups still balanced via BOM's min-group principle
- ✅ More expressive latent space overall

## Files Modified

- `configs/config.py`: All constraint changes, version bump to v15
- `utils/viz.py`: KL plot fix for split metrics

## BOM Philosophy Maintained

Despite loosening constraints, v15 still follows strict BOM principles:
- Pure log barrier: `-log(min(goals))`
- No epsilon softening in the barrier function
- BOX constraints use soft exponential tails (steepness=20.0)
- All 6 goal groups balanced via geometric mean
- Hard barriers force all constraints to improve simultaneously
