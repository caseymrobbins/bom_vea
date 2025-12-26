# BOM VAE v12

**v12 Changes from v11:**
- LPIPS texture distance (replaces Gram matrix)
- Core consistency regularization (core invariant to augmentation)
- Data augmentation (horizontal flip, color jitter)
- Batch size 256 for A100

## Quick Start

```bash
cd bom_vae_v12
python train.py
```

## Key Features

| Feature | v11 | v12 |
|---------|-----|-----|
| Texture metric | Gram matrix | LPIPS |
| Consistency loss | None | Core invariance |
| Augmentation | None | Flip + color jitter |
| Batch size | 128 | 256 |

## Expected Improvements

- Better texture transfer (LPIPS aligns with perception)
- Meaningful core/detail split (consistency enforces it)
- Less overfitting (augmentation)
