# Muon Optimizer Experiment

## Overview
This branch experiments with replacing AdamW with the Muon optimizer for the VAE's 2D parameters (convolutional and linear weights).

## What is Muon?
- **Muon** (MomentUm Orthogonalized by Newton-schulz) is a modern optimizer designed for hidden layer weights
- Uses Newton-Schulz iterations to orthogonalize gradient momentum
- Shown to achieve faster convergence and more stable training compared to Adam
- Introduced by Keller Jordan in October 2024

## Implementation Details

### Current Configuration (Conservative)
```python
# VAE Model - Hybrid Muon + AdamW
- 2D parameters (Conv2d, Linear weights): Muon
  - Learning rate: 0.01
  - Momentum: 0.95
  - Weight decay: 1e-5

- 1D parameters (biases, BatchNorm): AdamW
  - Learning rate: 1e-3
  - Betas: (0.9, 0.95)
  - Weight decay: 1e-5

# Discriminators - Pure AdamW (unchanged)
- Learning rate: 2e-4
- Weight decay: 1e-5
```

### Hyperparameter Rationale

**Muon LR = 0.01:**
- Previous AdamW LR: 1e-3
- Muon typically uses 10-20x higher LR than Adam
- Starting conservatively at 10x (0.01) to avoid destabilizing the BOM loss system
- Can experiment with higher values (0.015, 0.02) if training is too slow

**AdamW LR = 1e-3:**
- Kept same as before for 1D parameters
- Maintains consistency for biases and BatchNorm layers

**Discriminators unchanged:**
- Keeping discriminators on pure AdamW to isolate VAE optimizer changes
- Can apply Muon to discriminators in follow-up experiments

## Alternative Configurations to Try

### Moderate (if conservative is too slow)
```python
Muon LR: 0.015
AdamW LR: 8e-4
```

### Aggressive (closer to typical Muon setups)
```python
Muon LR: 0.02
AdamW LR: 5e-4
```

## Expected Benefits
1. **Faster convergence** - Reduced training time to reach target loss
2. **More stable gradients** - Orthogonalization may complement BOM's barrier optimization
3. **Better generalization** - Improved feature learning in hidden layers

## Monitoring During Training
Watch for:
- Convergence speed compared to baseline
- Gradient stability (should see fewer NaN/Inf rollbacks)
- Reconstruction quality progression
- KL divergence behavior (may be different due to different optimization dynamics)

## Files Modified
- `train.py`: Optimizer setup (lines 58-83)
- `muon.py`: Added Muon optimizer implementation

## References
- [Muon GitHub Repository](https://github.com/KellerJordan/Muon)
- [Muon Blog Post](https://kellerjordan.github.io/posts/muon/)
- [PyTorch 2.9 Documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
