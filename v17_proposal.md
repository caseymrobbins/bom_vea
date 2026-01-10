# v17 Configuration Proposal: "Point the Lazy Optimizer"

## Philosophy: LBO Finds the Easiest Path - Make That Path Lead to Quality

**User Insight**: "LBO is lazy - spell out what the goal is and where it fails. It will roll to the endpoint while avoiding the walls."

---

## Proposed Changes

### 1. **KL Upper Bounds with Aggressive Squeeze** ⭐ CRITICAL

**Current v16**:
```python
'kl_core': {'type': ConstraintType.LOWER, 'margin': 100.0},   # No upper!
'kl_detail': {'type': ConstraintType.LOWER, 'margin': 100.0}, # No ceiling!
```

**Proposed v17**:
```python
# Epoch 1: Let natural descent happen (init ~18k → ~12k)
# Epoch 2+: Add upper bounds and squeeze aggressively

'kl_core': {
    'type': ConstraintType.BOX_ASYMMETRIC,
    'lower': 100.0,
    'upper': 25000.0,  # Epoch 1 only (contains init)
    'healthy': 3000.0, # Final target
    'lower_scale': 2.0
},
'kl_detail': {
    'type': ConstraintType.BOX_ASYMMETRIC,
    'lower': 100.0,
    'upper': 25000.0,  # Epoch 1 only
    'healthy': 3000.0, # Final target
    'lower_scale': 2.0
}
```

**Squeeze Schedule** (in training loop):
```python
# After epoch 1, replace with tightening upper bounds
kl_upper_schedule = {
    2: 15000,  # Drop from 25k → 15k (current observed value)
    3: 13000,  # -2k per epoch (aggressive)
    4: 11000,  # -2k
    5: 10000,  # Target floor
    # Then adaptive squeeze takes over if stable
}
```

**Why**: Currently KL sits at 15k with no pressure to descend. This forces the descent.

---

### 2. **Relax Capacity Constraints** ⭐ CRITICAL

**Current v16**:
```python
'core_active': {'scale': 0.3},    # Forces 70%+ active dims
'detail_active': {'scale': 0.3},  # Very aggressive!
```

**Proposed v17**:
```python
'core_active': {'scale': 0.4},     # Requires 60%+ active (looser)
'detail_active': {'scale': 0.4},   # Allows more flexibility
'core_effective': {'scale': 0.4},  # Requires 60%+ effective
'detail_effective': {'scale': 0.4} # More breathing room
```

**Why**:
- scale=0.3 requires high variance per dim to hit 70% active
- But detail_var_health upper=600 caps total variance
- **Conflict**: Can't satisfy both → detail collapse
- scale=0.4 still enforces 60%+ usage but allows lower per-dim variance

**Math**:
```
score = exp(-inactive_ratio / scale)
For score > 0.5:
  scale=0.3 → inactive < 21% → active > 79% (51/64 dims)
  scale=0.4 → inactive < 28% → active > 72% (46/64 dims)
```

---

### 3. **Make Appearance a Hard Requirement** ⭐ HIGH PRIORITY

**Current v16**: Swap group is geometric mean
```python
S_swap = (S_structure * S_appearance * S_color)^(1/3)
```
Result: Can satisfy with structure alone (appearance=0.01!)

**Proposed v17 Option A**: Separate appearance into latent group
```python
# In GOAL_SPECS, add appearance to latent goals
'swap_appearance_hard': {
    'type': ConstraintType.MINIMIZE_SOFT,
    'scale': 0.05  # Tight - force variation
}
```

**Proposed v17 Option B**: Add lower bound to appearance
```python
'swap_appearance': {
    'type': ConstraintType.BOX,
    'lower': 0.0,
    'upper': 0.15,  # Must achieve < 0.15 appearance error
}
```

**Why**: Current lazy path ignores appearance entirely. Make it mandatory.

---

### 4. **Widen Detail Variance Bounds** ⭐ MEDIUM PRIORITY

**Current v16**:
```python
'detail_var_health': {'upper': 600.0}  # Caps total variance
```

**Proposed v17**:
```python
'detail_var_health': {'upper': 1200.0}  # Double the ceiling
# OR remove upper entirely, only enforce lower=0
```

**Why**:
- Detail needs high variance to spread points in latent space
- Current upper=600 keeps variance low → clustering → identical traversals
- Doubling to 1200 allows more spread while still preventing explosion

---

### 5. **Increase Learning Rate (20x Efficiency)** ⭐ MEDIUM PRIORITY

**Current v16**:
```python
LEARNING_RATE = 2e-3      # Already 4x standard VAE
LEARNING_RATE_D = 2e-4    # 10x slower
```

**Proposed v17**:
```python
LEARNING_RATE = 3e-3      # 50% increase (1.5x current)
LEARNING_RATE_D = 3e-4    # Maintain 10x ratio
```

**Why**:
- User observation: "Convergence at epoch 4-5 = 20x more efficient"
- Current 2e-3 might still be conservative
- Rollback mechanism protects from overshooting
- Start at 3e-3, can push to 4e-3 if stable

---

### 6. **Training Loop: Dynamic KL Squeeze Logic**

**New function** (add to train.py):
```python
def apply_kl_squeeze(epoch, goal_specs):
    """Apply aggressive KL upper bound squeeze after epoch 1"""
    kl_schedule = {
        1: 25000,  # Calibration (wide margin)
        2: 15000,  # Start squeeze
        3: 13000,  # -2k per epoch
        4: 11000,
        5: 10000,  # Floor (adaptive takes over)
    }

    if epoch in kl_schedule:
        new_upper = kl_schedule[epoch]
        goal_specs['kl_core']['upper'] = new_upper
        goal_specs['kl_detail']['upper'] = new_upper
        print(f"🔽 KL upper bounds squeezed to {new_upper}")
```

**Call in training loop** (before epoch starts):
```python
# After epoch 1, start KL squeeze
if epoch >= 1:
    apply_kl_squeeze(epoch, goal_specs)
```

---

## Summary: What Changes

| Goal | v16 | v17 | Impact |
|------|-----|-----|--------|
| KL bounds | Lower only | Upper squeeze 15k→10k | Forces KL descent |
| Capacity scale | 0.3 (70% active) | 0.4 (60% active) | Reduces variance conflict |
| Appearance | Optional (in group) | Hard requirement | Forces detail usage |
| Detail var upper | 600 | 1200 | Allows spreading in latent space |
| Learning rate | 2e-3 | 3e-3 | Faster exploration (20x efficiency) |

---

## Expected v17 Trajectory

**Epochs 1-2**: Natural KL descent (18k → 12k), upper bound at 25k → 15k
**Epochs 3-5**: Aggressive KL squeeze (15k → 10k), appearance variation increases
**Epochs 6-10**: Adaptive squeeze takes over, detail space opens up
**Epochs 11-20**: Convergence at KL~3-5k, appearance~0.3+, SSIM~0.75+

**Key Metric to Watch**:
- Appearance score should rise from 0.01 → 0.20+ by epoch 5
- Detail traversals should show high-frequency variation (not blur)
- KL should follow squeeze schedule downward

---

## Implementation Priority

1. **CRITICAL**: KL upper bounds + squeeze schedule (enables descent)
2. **CRITICAL**: Relax capacity scale 0.3→0.4 (resolves conflict)
3. **HIGH**: Make appearance mandatory (forces detail usage)
4. **MEDIUM**: Widen detail_var_health (allows spreading)
5. **MEDIUM**: Increase LR to 3e-3 (faster convergence)

---

## Next Steps

1. Modify `configs/config.py` with v17 changes
2. Add KL squeeze logic to `train.py`
3. Run 25 epoch training on CelebA
4. Monitor: appearance score, detail traversals, KL descent
5. If successful: Push to v18 with further refinements
