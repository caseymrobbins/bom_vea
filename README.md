# BOM VAE v13 - Explicit Structure/Appearance Separation

## Key Insight
Previous versions measured "similarity" without specifying *what* should be similar.

v13 fixes this with **explicit** constraints:
- **Core dims = STRUCTURE** → edges(r_sw) must match edges(x1)
- **Detail dims = APPEARANCE** → colors(r_sw) must match colors(x2)

## New Goals

| Goal | What it measures | Target |
|------|------------------|--------|
| `swap_structure` | Edge map MSE | Match x1 |
| `swap_appearance` | Mean color MSE | Match x2 |
| `swap_color_hist` | Color histogram MSE | Match x2 |

## Expected Results

**Traversals should now show:**
- Core dims: face shape, eye positions, nose size (structure)
- Detail dims: skin tone, lighting, color temperature (appearance)

**Cross-reconstruction should show:**
- x1's face geometry with x2's coloring/lighting

## Run
```bash
cd bom_vae_v13
python train.py
```

## Theory
More precise goals = less room for model to cheat.
BOM ensures ALL goals are satisfied, so explicit constraints force genuine disentanglement.
