# BOM VAE v14 - Discriminator + Detail Contracts + Meaningful Traversals

## Key Insight
Explicitly define **what** should be similar (structure vs appearance) and ensure latent
traversals actually *do* something measurable.

## Core Ideas
- **Core dims = STRUCTURE** → edges(r_sw) must match edges(x1).
- **Detail dims = APPEARANCE** → colors(r_sw) must match colors(x2).
- **Realism** enforced by PatchGAN discriminator.
- **Detail contracts** keep the appearance channel statistically healthy.
- **Traversal meaning**: perturbing latent dims should change their *intended* outputs.

## Goals (high level)

| Goal | What it measures | Target |
|------|------------------|--------|
| `swap_structure` | Edge map MSE | Match x1 |
| `swap_appearance` | Mean color MSE | Match x2 |
| `swap_color_hist` | Color histogram MSE | Match x2 |
| `realism_recon` / `realism_swap` | PatchGAN realism | High realism |
| `detail_mean` / `detail_var_mean` / `detail_cov` | Detail stats | Stable bounds |
| `traversal` | Inverse of (core→edge shift, detail→color shift) | Encourage meaningful change |

`traversal` is part of the **log(min())** barrier, so weak traversals become a bottleneck.

## Expected Results

**Traversals should show:**
- Core dims: face shape, eye positions, nose size (structure)
- Detail dims: skin tone, lighting, color temperature (appearance)

**Cross-reconstruction should show:**
- x1's face geometry with x2's coloring/lighting

## Run
```bash
python train.py
```

## Theory
More precise goals = less room for the model to cheat.
BOM ensures **all** goals are satisfied, so explicit constraints + traversal meaning
force genuine disentanglement.
