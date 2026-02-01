# v17b Hotfix: Constraint Violation Fixes

## Problem: 49% Rollback Rate (Complete Training Failure)

**Observed**: After calibration (batch 200), **every single batch** (200-395) was rejected with rollback.

**Root Cause**: Hard BOX constraints were too tight, causing instant rejection (`goal=0` → `loss=inf` → rollback).

---

## Fixes Applied

### 1. **swap_appearance: BOX → BOX_ASYMMETRIC** ⭐ CRITICAL

```python
# ❌ v17 (caused 49% rollback)
'swap_appearance': {'type': ConstraintType.BOX, 'lower': 0.0, 'upper': 0.15}

# ✅ v17b (soft pressure)
'swap_appearance': {
    'type': ConstraintType.BOX_ASYMMETRIC,
    'lower': 0.0,
    'upper': 0.50,      # Wide ceiling (won't reject)
    'healthy': 0.15,    # Target to optimize toward
    'lower_scale': 1.0
}
```

**Why it matters**:
- **BOX**: Hard rejection if violated (goal=0 instantly)
- **BOX_ASYMMETRIC**: Soft barrier (goal decreases smoothly as you approach ceiling)

During training, appearance error likely rises above 0.15 temporarily → BOX would reject → BOX_ASYMMETRIC allows it but applies pressure.

---

### 2. **KL Upper Bounds: 30k → 45k**

```python
# ❌ v17 (violated by init)
'kl_detail': {'upper': 30000.0}  # init=30724 EXCEEDED

# ✅ v17b (40% margin)
'kl_detail': {'upper': 45000.0}  # 30724 × 1.4 ≈ 43k
```

**Calibration Warning**:
```
⚠️  kl_detail: init range [128, 30724] outside BOX [100, 30000]
```

**Fix**: Widened to 45k to contain initialization with margin.

---

### 3. **detail_mean: [-20, 20] → [-30, 30]**

```python
# ❌ v17 (violated by init)
'detail_mean': {'lower': -20.0, 'upper': 20.0}  # init=21.32 EXCEEDED

# ✅ v17b (40% margin)
'detail_mean': {'lower': -30.0, 'upper': 30.0}  # 21.32 × 1.4 ≈ 30
```

**Calibration Warning**:
```
⚠️  detail_mean: init range [0.00, 21.32] outside BOX [-20.00, 20.00]
```

---

### 4. **detail_var_mean: 350 → 500**

```python
# ❌ v17 (violated by init)
'detail_var_mean': {'upper': 350.0}  # init=354.0 EXCEEDED

# ✅ v17b (40% margin)
'detail_var_mean': {'upper': 500.0}  # 354 × 1.4 ≈ 500
```

**Calibration Warning**:
```
⚠️  detail_var_mean: init range [0.00, 354.00] outside BOX [0.00, 350.00]
```

---

### 5. **Removed Traversal Diversity Monitoring** (Temporary)

**Issue**: Code was trying to decode during epoch aggregation, causing crash:
```python
File "/content/bom_vea/train.py", line 427, in <module>
    recon = model.decode(z_combined)
KeyboardInterrupt
```

**Fix**: Removed for now, will re-add later after ensuring it runs at correct point in training loop.

---

## Expected Impact

### Before (v17):
- **Rollback rate**: 49% (195/395 batches rejected)
- **Training**: Completely stuck, no successful optimization steps

### After (v17b):
- **Rollback rate**: <5% (target)
- **Training**: Should proceed normally with soft pressure on appearance

---

## Key Lesson: BOX vs BOX_ASYMMETRIC

**BOX** (hard constraint):
- If violated → `goal = 0` → `loss = inf` → instant rollback
- Use only when bounds are VERY wide and violations are unacceptable

**BOX_ASYMMETRIC** (soft barrier):
- If exceeded → `goal` decreases smoothly (but stays > 0)
- Creates pressure to stay within `healthy` range
- Allows temporary violations during optimization
- **Preferred for "target" constraints** where you want pressure, not hard rejection

---

## Updated Configuration Summary

| Constraint | v17 | v17b | Reason |
|------------|-----|------|--------|
| swap_appearance | BOX [0, 0.15] | BOX_ASYM [0, 0.5] h=0.15 | Avoid hard rejection |
| kl_core | upper=30k | upper=45k | Init exceeded 30k |
| kl_detail | upper=30k | upper=45k | Init=30724 violated |
| detail_mean | [-20, 20] | [-30, 30] | Init=21.32 violated |
| detail_var_mean | upper=350 | upper=500 | Init=354 violated |

---

## Files Changed

- `configs/config.py`: All 5 fixes above
- `train.py`: Removed traversal diversity check

## Commit

```
251694d - Fix v17: Critical constraint violations causing 49% rollback rate
```

---

## Next Steps

1. **Test v17b**: Run `python train.py` and verify rollback rate < 5%
2. **Monitor appearance**: Should stay below 0.50 ceiling, optimize toward 0.15
3. **Re-add traversal monitoring**: After confirming training works, add diversity check at end of epoch (not during aggregation)

---

## Quick Start (Colab)

```python
# Pull latest fixes
!git pull origin claude/lbo-vae-implementation-1eyE0

# Run training
!python train.py

# Watch for:
# - Rollback rate should be <5% (not 49%!)
# - No BOX CONSTRAINT VIOLATION warnings
# - Appearance score rising toward 0.15
```
