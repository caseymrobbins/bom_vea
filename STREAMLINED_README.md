# Streamlined BOM-VAE: 9 Goals Instead of 35

## Overview

This streamlined version reduces the VAE training complexity from **35 individual goals** to **9 consolidated goals**, organized into **3 groups** instead of 8.

**Problem**: The original 35 goals created excessive gradient competition, resulting in blurry reconstructions because the optimizer was pulled in too many directions simultaneously.

**Solution**: Merge related goals into meaningful composites while prioritizing **useful, disentangled latents** over perfect reconstruction.

---

## Goal Reduction: 35 → 9

### Original Structure (35 goals, 8 groups)

```
recon (3): pixel, edge, perceptual
core (2): core_mse, core_edge
swap (3): swap_structure, swap_appearance, swap_color_hist
realism (2): realism_recon, realism_swap
disentangle (2): core_color_leak, detail_edge_leak
separation (4): sep_core, sep_mid, sep_detail, prior_kl
latent (16):
  - kl (5): kl_core, kl_detail, prior_kl, logvar_core, logvar_detail
  - structure (3): cov, weak, core_consistency
  - capacity (4): core_active, detail_active, core_effective, detail_effective
  - detail_stats (4): detail_mean, detail_var_mean, detail_cov, traversal
health (3): detail_ratio, core_var_health, detail_var_health
```

### Streamlined Structure (9 goals - FLAT, no grouping)

**Pure LBO**: `loss = -log(min(all 9 goals))`

With only 9 goals, we don't need grouping. All goals are equal:

```
1. kl_divergence: Merged kl_core + kl_detail + prior_kl
2. disentanglement: Merged sep_core + sep_mid + sep_detail
3. capacity: Merged core_active + detail_active + core_effective + detail_effective
4. behavioral_separation: Merged core_color_leak + detail_edge_leak
5. latent_stats: Merged logvar_core + logvar_detail + cov + weak
6. reconstruction: Merged pixel + edge + perceptual (0.2*pixel + 0.3*edge + 0.5*perceptual)
7. cross_recon: Merged swap_structure + swap_appearance + swap_color_hist
8. realism: Merged realism_recon + realism_swap
9. consistency: core_consistency (augmentation invariance)
```

**No grouping** - Each goal competes equally for attention via the global min() barrier.

---

## Key Changes

### 1. **Merged Goals**

**KL Divergence** (was 5 goals → now 1):
- Combines core KL, detail KL, and prior KL into single budget
- Adaptive squeeze from 15k → 3k over epochs 3-15
- Simpler than tracking 3 separate KL components

**Reconstruction** (was 3 goals → now 1):
- Weighted combo: `0.2*pixel + 0.3*edge + 0.5*perceptual`
- Perceptual loss dominates (faces care more about features than pixels)

**Capacity** (was 4 goals → now 1):
- Merges active/effective counts for both core and detail
- Single metric: % of dimensions actively used
- Prevents low-rank collapse

**Disentanglement** (was 3 TC discriminators → now 1):
- Average of sep_core, sep_mid, sep_detail
- Single TC penalty for total correlation

**Behavioral Separation** (was 2 goals → now 1):
- Combines core_color_leak + detail_edge_leak
- Enforces: core affects structure not color, detail affects color not structure

**Latent Stats** (was multiple goals → now 1):
- Merges logvar, covariance, weak dimension penalties
- Prevents variance explosion and dimension collapse

### 2. **Flat Structure - No Grouping**

**Old**: 35 goals in 8 groups with hierarchical geometric means
**New**: 9 flat goals with pure min() across all

- **No artificial hierarchy** - LBO treats all 9 goals equally
- **Pure barrier optimization** - The single worst goal determines loss
- **Simpler math** - No geometric mean grouping, just `loss = -log(min(all 9))`

### 3. **Removed Goals**

- ❌ `core_mse, core_edge` - Redundant with reconstruction
- ❌ Individual capacity metrics - Merged into single capacity score
- ❌ `detail_mean, detail_var_mean, detail_cov, traversal` - Merged into latent_stats
- ❌ `core_var_health, detail_var_health` - Merged into latent_stats
- ❌ `detail_ratio` - Not critical for core objectives

---

## File Structure

```
configs/
  config_streamlined.py         # 9 goals, 3 groups

losses/
  bom_loss_streamlined.py       # Streamlined loss computation

train_streamlined.py            # Main training script using streamlined config
```

---

## How to Use

### Run streamlined training:
```bash
python train_streamlined.py
```

### Compare with original:
```bash
# Original (35 goals)
python train.py

# Streamlined (9 goals)
python train_streamlined.py
```

---

## Expected Benefits

### 1. **Clearer Gradient Flow**
- 9 goals instead of 35 means each gets ~11% of gradient budget (vs 2.8%)
- No grouping means **pure competition** - the worst goal gets fixed first
- LBO naturally balances all 9 goals without artificial hierarchies

### 2. **Better Reconstructions**
- Less gradient competition → less blur
- Perceptual loss weighted 0.5 → prioritizes feature similarity over pixel MSE
- Original: reconstruction starved at ~3% budget
- Streamlined: reconstruction gets ~22% budget

### 3. **Faster Convergence**
- Fewer constraints to satisfy simultaneously
- Muon optimizer can focus updates on fewer objectives
- Barrier optimization has fewer bottlenecks to avoid

### 4. **Easier Debugging**
- 3 groups easier to monitor than 8
- Clear hierarchy: latent quality > reconstruction > stability

---

## Configuration Details

### Goal Specifications

All goals use `MINIMIZE_SOFT` with auto-calibrated scales except:

**kl_divergence**: BOX_ASYMMETRIC
- Lower: 0, Upper: 1e9 (dynamically squeezed)
- Healthy: 2e8, Lower_scale: 2.0

**realism**: Fixed scale = 2.0
- Discriminator untrained during calibration

**consistency**: Fixed scale = 500.0
- Large scale for augmentation variance

---

## Monitoring Training

### Key Metrics to Watch

**Goal Scores** (all 9 should trend toward 0.5-0.7):
- `kl_divergence`: Target >0.5
- `disentanglement`: Target >0.5
- `capacity`: Target >0.5
- `behavioral_separation`: Target >0.5
- `latent_stats`: Target >0.5
- `reconstruction`: Target >0.5
- `cross_recon`: Target >0.5
- `realism`: Target >0.5
- `consistency`: Target >0.5

**Bottleneck Goal**: Watch `min_idx` to see which goal is the current bottleneck

**Raw Metrics**:
- `kl_total_raw`: 15k → 3k over epochs
- `disentangle_raw`: Should decrease (TC penalty)
- `capacity_raw`: Should decrease (more dims active)
- `ssim`: 0.3 → 0.7+ over training

**Individual Goals** (scores 0-1):
- All should trend toward 0.5-0.7
- If stuck <0.3, that goal is the bottleneck

---

## Comparison: Original vs Streamlined

| Metric | Original (35 goals) | Streamlined (9 goals) |
|--------|--------------------|-----------------------|
| **Goals** | 35 | 9 |
| **Groups** | 8 (hierarchical) | 0 (flat) |
| **Structure** | Geometric mean groups | Pure min() across all |
| **Goal budget** | ~2.8% each | ~11% each |
| **Expected SSIM** | 0.4-0.5 | 0.6-0.7 |
| **Gradient complexity** | Very high | Low |
| **Debug difficulty** | Hard | Easy |
| **LBO purity** | Diluted by grouping | Pure barrier optimization |

---

## Troubleshooting

### Blurry reconstructions?
- Check `reconstruction` goal score - should be >0.4
- If stuck <0.3, reconstruction is still starved
- Consider increasing reconstruction group weight

### Posterior collapse?
- Check `kl_total_raw` - should be 3k-15k
- If <100: low KL collapse (ignoring latent)
- If >25k: high KL collapse (everything maps to same point)

### Poor disentanglement?
- Check `disentangle_raw` (TC penalty) - should be <5.0
- Check `behavioral_sep_raw` - should be <0.1
- If high, TC discriminators may need more training

### Low capacity?
- Check `capacity_raw` - should be <0.4 (60%+ dims active)
- If >0.5, latent space collapsing to low rank

---

## Next Steps

1. **Run comparison**: Train both original and streamlined for 5 epochs
2. **Compare metrics**: SSIM, FID, reconstruction quality, latent traversals
3. **Adjust if needed**: Can further reduce to 7 goals if still too complex
4. **Production**: Use streamlined version as new baseline

---

## Philosophy

> **"Useful latents > Perfect reconstruction"**

The streamlined version prioritizes learning disentangled, interpretable latent representations over pixel-perfect reconstructions. This aligns with VAE best practices:

- A VAE with blurry reconstructions but perfect disentanglement is valuable
- A VAE with sharp reconstructions but collapsed latents is useless
- The optimal VAE balances both, but leans toward latent quality

By reducing from 35 → 9 goals, we give the optimizer room to breathe while maintaining all critical properties:
- ✅ Disentanglement (TC penalties)
- ✅ Capacity (full latent utilization)
- ✅ Behavioral separation (structure vs appearance)
- ✅ Statistical health (variance, covariance)
- ✅ Reconstruction quality (perceptual + edge)
- ✅ Consistency (augmentation invariance)
