# BOM VAE v14 - Colab Setup Guide

## Quick Start

1. **Upload the zip file** to your Colab session:
```python
from google.colab import files
uploaded = files.upload()  # Upload bom_vae_v14.zip
```

2. **Extract and enter directory**:
```python
!unzip bom_vae_v14.zip -d bom_vae_v14
%cd bom_vae_v14
```

3. **Install dependencies** (if needed):
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install torchmetrics[image] -q
```

4. **Setup CelebA dataset** (optional - skip if using other data):
```python
!python celeba_setup.py
```

5. **Run training**:
```python
!python train.py
```

## Expected Output

### Phase 1: Calibration (batches 0-200)
```
BOM VAE v14 - CELEBA - 30 EPOCHS
v14: Discriminator + Detail contracts
     - PatchGAN discriminator with spectral norm
     - KL divergence for BOTH core and detail channels
     - Detail contracts: mean, variance, covariance
================================================================================

Model params: 15,234,567
Discriminator params: 2,345,678

📊 Epoch 1: Calibrating...
Epoch 1/30:   5%|██▍                                           | 200/4000 [00:45<14:30,  4.37it/s, phase=CALIBRATING]

============================================================
CALIBRATING GOALS (#1, epoch 1)
============================================================
  pixel               : scale=0.0523
  edge                : scale=0.0089
  perceptual          : scale=0.1234
  ...
  detail_mean         : BOX [-3.00, 3.00] | init=[-0.05, 0.08] median=0.01
  detail_var_mean     : BOX [0.10, 10.00] | init=[1.23, 2.45] median=1.67
  kl_core             : BOX_ASYM [50, 8000] h=2000 | init=[1850, 2150] median=2000
  kl_detail           : BOX_ASYM [50, 8000] h=2000 | init=[1823, 2189] median=1998
============================================================
```

### Phase 2: Training
```
Epoch 1/30: 100%|██████████████████████████████████████| 4000/4000 [18:23<00:00,  3.62it/s, loss=1.52, min=0.217, bn=realism, ssim=0.832]

Epoch  1 | Loss: 1.523 | Min: 0.217 | SSIM: 0.832
         Structure: 0.0123 | Appearance: 0.0456
         KL_core: 2050.3 | KL_detail: 1987.1
         Groups: recon:0.65 | core:0.58 | swap:0.47 | realism:0.42 | latent:0.52 | health:0.61
```

## Troubleshooting

### ⚠️ BOX Constraint Violation
```
⚠️  WARNING: BOX CONSTRAINT VIOLATIONS ⚠️
    detail_var_mean: init range [15.23, 51.45] outside BOX [0.10, 10.00]
    These constraints will return goal=0 → loss=inf → crash!
    ACTION: Widen BOX bounds to contain initialization values.
```

**Fix**: Edit `configs/config.py`:
```python
'detail_var_mean': {'type': ConstraintType.BOX, 'lower': 0.01, 'upper': 100.0},  # Widened
```

### ⚠️ Loss becomes infinite
```
Epoch 1: loss=1.52 → 1.89 → 2.34 → inf
```

**Cause**: A group goal hit exactly 0 (happens when BOX constraint is violated)

**Fix**:
1. Check calibration output for BOX violations
2. Widen the violated BOX constraint
3. Restart training

### ⚠️ KL explodes (> 8000)
```
KL_detail: 2000 → 5000 → 12000 → 45000
```

**Cause**: Detail channel has no KL regularization

**Check**: Ensure mu clamping is present in `models/vae.py:65`:
```python
mu = torch.clamp(self.mu(h), -10, 10)  # Should be present!
```

### ⚠️ Discriminator dominates
```
Groups: recon:0.65 | core:0.58 | swap:0.47 | realism:0.15 | latent:0.52 | health:0.61
       (realism is ALWAYS the bottleneck)
```

**Fix**: Reduce discriminator learning rate in `configs/config.py`:
```python
LEARNING_RATE_D = 5e-5  # Was 1e-4
```

## Monitor These Values

✅ **All groups 0.4-0.8**: Healthy optimization
❌ **Any group < 0.1**: That constraint is failing → will crash soon
✅ **KL in [50, 8000]**: Both core and detail should stay in range
❌ **KL > 8000 or < 50**: Check BOX bounds or mu clamping
✅ **Min group rising over epochs**: System is improving
❌ **Min group falling**: Something is getting worse

## What Each Group Means

- **recon**: Full image reconstruction quality
- **core**: Structure preservation (core channel only)
- **swap**: Structure/appearance disentanglement
- **realism**: Discriminator scores (how realistic are outputs)
- **latent**: Latent space health (KL, covariance, detail contracts)
- **health**: Overall system health (variance bounds, detail ratio)

## Expected Training Time

- **CelebA 64x64, 30 epochs**: ~6-8 hours on Colab T4
- **Batch size 256**: ~15 batches/second
- **Calibration**: ~45 seconds (first 200 batches of epoch 1)

## Files Generated

After training completes, check `/content/outputs_bom_v14/`:
```
bom_vae_v14.pt              # Model checkpoint
group_balance.png           # Group performance over time
reconstructions.png         # Sample reconstructions
traversals_core.png         # Core latent traversals (structure)
traversals_detail.png       # Detail latent traversals (appearance)
cross_reconstruction.png    # Structure/appearance swapping
training_history.png        # Loss curves
```

## Next Steps After Successful Training

1. **Check visualizations**: Look at cross_reconstruction.png to verify disentanglement
2. **Inspect group balance**: All groups should be roughly equal by end of training
3. **Run evaluation**: Check FID, SSIM, LPIPS scores in training output
4. **Experiment**: Try different latent traversals, swap more images

## Version History

- **v13**: Structure/appearance separation
- **v14**: + Discriminator + Detail contracts (current)
- **v15**: Planned - Attention + Skip connections + Higher resolution
