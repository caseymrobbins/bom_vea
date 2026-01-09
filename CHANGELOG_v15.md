# BOM VAE v15 - Hierarchical Latent Constraints + Dimension Utilization Enforcement

## 📋 Summary

Version 15 addresses a critical issue discovered in v14: **dimension utilization was tracked but not enforced**. The latent group had 10 flat goals whose geometric mean diluted the optimization signal, preventing the model from effectively utilizing all 128 latent dimensions.

## 🎯 Problem Identified

**From v14 training analysis:**
- Latent bottleneck: **0-2.5%** (almost never the constraint - too loose!)
- KL divergence stayed high: Core=3535, Detail=8057
- Active/effective dimensions were **logged but not enforced as LBO goals**
- 10 flat goals in latent group → weak gradient signal when individual goals fail

## ✨ Key Changes

### 1. Hierarchical Latent Group Structure

**Before (v14) - Flat structure:**
```python
group_latent = geometric_mean([
    g_kl_core, g_kl_detail, g_logvar_core, g_logvar_detail,  # 4 KL goals
    g_cov, g_weak, g_consistency,                             # 3 structure goals
    g_detail_mean, g_detail_var_mean, g_detail_cov           # 3 detail stats
])  # 10-way geometric mean → diluted signal
```

**After (v15) - Hierarchical structure:**
```python
# SUB-GROUP F1: Distribution Matching (KL constraints)
group_kl = geometric_mean([g_kl_core, g_kl_detail, g_logvar_core, g_logvar_detail])

# SUB-GROUP F2: Independence & Consistency
group_structure = geometric_mean([g_cov, g_weak, g_consistency])

# SUB-GROUP F3: Dimension Capacity Utilization (NEW!)
group_capacity = geometric_mean([g_core_active, g_detail_active, g_core_effective, g_detail_effective])

# SUB-GROUP F4: Detail Statistics
group_detail_stats = geometric_mean([g_detail_mean, g_detail_var_mean, g_detail_cov])

# FINAL: Combine 4 sub-groups
group_latent = geometric_mean([group_kl, group_structure, group_capacity, group_detail_stats])
```

### 2. New Capacity Goals (Previously Missing!)

Four new goals added to enforce dimension utilization:

| Goal | Measures | Target | LBO Constraint |
|------|----------|--------|----------------|
| `core_active` | Fraction of inactive core dims (var < 0.1) | Minimize | MINIMIZE_SOFT (auto-scaled) |
| `detail_active` | Fraction of inactive detail dims (var < 0.1) | Minimize | MINIMIZE_SOFT (auto-scaled) |
| `core_effective` | Effective dimension deficit (64 - eff_dims)/64 | Minimize | MINIMIZE_SOFT (auto-scaled) |
| `detail_effective` | Effective dimension deficit (64 - eff_dims)/64 | Minimize | MINIMIZE_SOFT (auto-scaled) |

**Active dimensions:** Count of dims with variance > 0.1
**Effective dimensions:** Exponential of entropy (measures usage uniformity)

### 3. Implementation Changes

**Files Modified:**
- `losses/bom_loss.py`: Added hierarchical latent structure + 4 capacity goals
- `configs/config.py`: Added 4 new goal specs
- `train.py`: Added tracking for capacity metrics

**New metrics computed:**
```python
# Active dimensions
core_active_count = (mu_core.var(0) > 0.1).sum()  # How many dims have variance > 0.1
detail_active_count = (mu_detail.var(0) > 0.1).sum()

# Effective dimensions (entropy-based)
core_var_norm = mu_core.var(0) / mu_core.var(0).sum()
core_effective = exp(-sum(p * log(p)))  # Exponential of Shannon entropy
```

## 📊 Expected Improvements

### Stronger Latent Enforcement

**Current (v14):**
- Latent bottleneck: 0-2.5% (ignored)
- Unknown dimension wastage
- KL = 3500+ (loose posterior)

**Expected (v15):**
- Latent bottleneck: 15-25% (enforced!)
- All 64 core + 64 detail dims utilized
- KL should decrease (tighter posterior)
- Possibly improved SSIM (better latent organization)

### Gradient Signal Strength

**Scenario:** 1 failing goal out of many

**v14 (10 flat goals):**
```
If 9 goals = 0.9, 1 goal = 0.2:
geometric_mean = 0.9^9 × 0.2 = 0.077 → Loss = 2.56
```

**v15 (4 sub-groups of 2-4 goals):**
```
Failing goal isolated to its sub-group → 4× stronger gradient
```

### Better Diagnostics

Can now identify which **sub-group** is the bottleneck:
- KL sub-group → distribution matching issues
- Structure sub-group → independence/consistency issues
- **Capacity sub-group → dimension utilization issues** ← NEW!
- Detail stats sub-group → detail channel issues

## 🧪 Testing Plan

### 1. Verify Capacity Enforcement

After 5-10 epochs, check:
```python
print(f"Core active: {core_active_count}/64")
print(f"Detail active: {detail_active_count}/64")
print(f"Core effective: {core_effective:.1f}/64")
print(f"Detail effective: {detail_effective:.1f}/64")
```

**Success criteria:**
- Active dims: 55-64/64 (>85% utilization)
- Effective dims: 45-60/64 (>70% effective capacity)

### 2. Monitor Latent Bottleneck %

Should increase from ~2% to 15-25% of batches, indicating capacity goals are now active constraints.

### 3. Watch KL Divergence

Should decrease from ~3500 to 1000-2000 range as posterior tightens (due to better dim utilization).

### 4. SSIM Improvement (Possible)

Better latent organization might push SSIM from 0.60 → 0.62-0.65.

## 🚀 Training with v15

**Recommended settings:**
- Use 10% adaptive squeeze (user already configured this)
- Monitor capacity sub-group specifically during training
- Expect more rollbacks initially as capacity constraints activate
- May need 5-10 more epochs than v14 for convergence

**Watch for:**
- Capacity bottleneck appearing in logs (this is GOOD!)
- Dimension counts increasing toward 64/64
- KL divergence coming down
- Latent group score decreasing (becoming a real constraint)

## 🎓 Why This Matters

**LBO Philosophy:**
> "Track it? Then enforce it. Otherwise you're just writing logs."

We were computing active/effective dimensions but not using them in the loss. This violates the LBO principle that all metrics should be constraints. The hierarchical bundling also follows your intuition - group related goals together for stronger, clearer optimization signals.

## 🔄 Backward Compatibility

**Breaking changes:**
- New goal specs required in config
- Loss computation changed (hierarchical)
- Saved models from v14 compatible, but training will differ

**Migration from v14:**
- Update config.py (done)
- Update loss computation (done)
- Retrain from scratch recommended for fair comparison

---

**Version:** v15
**Date:** 2026-01-09
**Author:** LBO Constitutional Framework Implementation
