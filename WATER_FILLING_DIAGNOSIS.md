# Water-Filling Optimization Diagnosis

## Your Question: "Would this work?"

```python
def agency_calculus_step(agency_vector):
    # 1. Identify the bottleneck (The lowest bucket)
    min_val, min_idx = torch.min(agency_vector, dim=0)

    # 2. The Gradient is applied ONLY to that bottleneck
    # The optimizer ignores the "rich" variables until the "poor" one catches up.
    loss = -torch.log(min_val)

    loss.backward()
    # Result: The system lifts the floor without destabilizing the ceiling.
```

**Answer: Yes, this IS what we're doing, and it's working correctly!**

The problem isn't the water-filling logic - it's that **one of the buckets has a hole in it.**

---

## What's Actually Happening

### The Code (bom_loss_streamlined.py:395-415)

```python
goals_tensor = torch.stack([g_kl_divergence, g_disentanglement, ...], dim=1)  # [B, 9]
loss = -torch.log(goals_tensor.min())  # Exactly your water-filling logic
```

This is precisely what you described. The gradient flows ONLY to the minimum goal.

### The Training Timeline

**Epoch 1 (Calibration):**
- Discriminator: Untrained, random weights
- Realism raw value: ~0.39 (easy to fool random discriminator)
- Auto-calibration sets: `realism scale = 0.39`
- All 9 goals achieve similar satisfaction (~0.5-0.8)

**Epoch 2-3:**
- Discriminator: Trains on real vs fake images, gets skilled
- Realism raw value: Increases to ~0.6, then ~0.8 (harder to fool trained discriminator)
- Realism scale: Still 0.39 (frozen after calibration)
- **Realism becomes the bottleneck**

**Epochs 3-25:**
- Water-filling logic: ✅ Correctly identifies realism as bottleneck
- Gradient flow: ✅ 100% to realism (working as designed!)
- **But**: Generator improves → Discriminator counters → Realism gets harder again
- Result: **Realism never catches up** (adversarial arms race)
- Meanwhile: Reconstruction gets 0% gradient for 22 epochs → model forgets how to reconstruct

---

## The Bucket With a Hole

The water-filling analogy assumes **static buckets**. But:

```
Generator improves realism → Discriminator trains harder → Realism difficulty increases
     ↑                                                              ↓
     └──────────────────── Adversarial Loop ──────────────────────┘
```

**You're trying to fill a bucket while someone else is drilling holes in it.**

The optimization is working correctly, but:
- Realism is an **adversarially moving target**
- Auto-calibration based on epoch 1 (untrained discriminator) doesn't capture training difficulty
- Realism dominates forever, starving all other goals

---

## The Root Cause

**Auto-calibration assumes goal difficulty is constant.**

When a goal's difficulty changes during training, auto-calibration fails:

1. **Realism**: Discriminator untrained in epoch 1 (D≈0.5) → trains quickly → D≈0.9 by epoch 5
   - Auto: `scale = 0.39` (based on easy epoch 1)
   - Reality: Need `scale = 2.0` (for trained discriminator)

2. **Consistency**: Low augmentation variance in epoch 1 (≈115) → increases during training (≈250)
   - Auto: `scale = 115` (based on weak epoch 1 variance)
   - Reality: Need `scale = 500.0` (for strong training variance)

3. **Capacity**: Few active dimensions in epoch 1 → more activate as model learns
   - Auto: Sets scale based on random initialization
   - Reality: Need `scale = 0.4` (for trained model)

---

## The Fix

**Use manual scales for goals whose difficulty changes during training.**

The original config (configs/config.py) already solved this problem! It uses:

### Auto-Calibration (Static Difficulty)
- `kl_divergence`: ✅ auto (KL relatively stable)
- `disentanglement`: ✅ auto (TC discriminator co-evolves, not adversarial)
- `behavioral_separation`: ✅ auto (color/edge leak stable)
- `latent_stats`: ✅ auto (variance/covariance stable)
- `reconstruction`: ✅ auto (MSE/VGG relatively stable)
- `cross_recon`: ✅ auto (swap consistency stable)

### Manual Scales (Changing Difficulty)
- `realism`: 🔧 `scale=2.0` (discriminator trains → harder)
- `consistency`: 🔧 `scale=500.0` (augmentation variance increases)
- `capacity`: 🔧 `scale=0.4` (latent utilization changes)

---

## Applied Fix

I've updated `configs/config_streamlined.py`:

```python
'realism': {
    'type': ConstraintType.MINIMIZE_SOFT,
    'scale': 2.0  # Manually tuned for trained discriminator
},

'consistency': {
    'type': ConstraintType.MINIMIZE_SOFT,
    'scale': 500.0  # Manually tuned for training variance
},

'capacity': {
    'type': ConstraintType.MINIMIZE_SOFT,
    'scale': 0.4  # Match original config
},
```

---

## Why This Should Work

With manual scales accounting for **post-training difficulty** (not calibration difficulty):

1. **Realism** gets looser scale → won't dominate early epochs
2. **Reconstruction** will get gradient share → model learns to reconstruct
3. **Capacity** properly balanced → latents activate without bottlenecking
4. **Consistency** has room for augmentation variance → robust features

The water-filling optimization will now balance goals based on their **actual training difficulty**, not their misleading epoch 1 calibration values.

---

## Summary

✅ **Your water-filling logic is correct and working**
❌ **Auto-calibration fails for adversarially-trained goals**
🔧 **Fix: Manual scales for realism/consistency/capacity**
📊 **Expected: Balanced gradient flow across all 9 goals**
