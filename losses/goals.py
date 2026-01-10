# losses/goals.py
# BOM Goal System - same as v11

import torch
import numpy as np
from typing import Dict, Callable
from configs.config import ConstraintType

def make_normalizer_torch(ctype: ConstraintType, **kwargs) -> Callable:
    if ctype == ConstraintType.UPPER:
        margin = kwargs["margin"]
        return lambda x: torch.clamp((margin - x) / margin, 0.0, 1.0)
    
    if ctype == ConstraintType.LOWER:
        margin = kwargs["margin"]
        return lambda x: torch.clamp((x - margin) / margin, 0.0, 1.0)
    
    if ctype == ConstraintType.BOX:
        lower, upper = kwargs["lower"], kwargs["upper"]
        mid = (lower + upper) / 2
        half_width = (upper - lower) / 2
        steepness = 20.0
        def soft_box(x):
            dist = torch.abs(x - mid) / half_width
            inside = 1.0 - dist
            outside = torch.exp(-steepness * (dist - 1.0))
            return torch.where(dist <= 1.0, inside, outside)
        return soft_box
    
    if ctype == ConstraintType.BOX_ASYMMETRIC:
        lower, upper, healthy = kwargs["lower"], kwargs["upper"], kwargs["healthy"]
        dist_lower, dist_upper = healthy - lower, upper - healthy
        steepness = 20.0
        def soft_asymmetric_box(x):
            below = (x - lower) / dist_lower
            above = (upper - x) / dist_upper
            inside = torch.where(x <= healthy, below, above)
            left_tail = torch.exp(-steepness * (lower - x) / dist_lower)
            right_tail = torch.exp(-steepness * (x - upper) / dist_upper)
            outside = torch.where(x < lower, left_tail, right_tail)
            return torch.where((x >= lower) & (x <= upper), inside, outside)
        return soft_asymmetric_box
    
    if ctype == ConstraintType.MINIMIZE_SOFT:
        scale = max(kwargs["scale"], 1e-6)  # Enforce minimum scale to prevent numerical instability
        return lambda x: torch.exp(-torch.clamp(x, min=0.0) / scale)
    
    if ctype == ConstraintType.MINIMIZE_HARD:
        scale = kwargs.get("scale", 1.0)
        return lambda x: 1.0 / (1.0 + (torch.clamp(x, min=0.0) / scale) ** 2)
    
    raise ValueError(f"Unknown constraint type: {ctype}")

class GoalSystem:
    def __init__(self, goal_specs: Dict):
        self.specs = goal_specs
        self.scales = {}
        self.normalizers = {}
        self.samples = {name: [] for name in goal_specs.keys()}
        self.calibrated = False
        self.calibration_count = 0
        self.epoch1_margin_applied = False  # Track if 10% safety margin is active
    
    def collect(self, loss_dict: Dict[str, float]):
        for name, value in loss_dict.items():
            if name in self.samples and not np.isnan(value) and not np.isinf(value):
                self.samples[name].append(value)
    
    def calibrate(self, epoch: int = 0):
        self.calibration_count += 1
        print("\n" + "=" * 60)
        print(f"CALIBRATING GOALS (#{self.calibration_count}, epoch {epoch})")
        print("=" * 60)

        # Verify BOX constraints contain initial values
        box_violations = []

        # EPOCH 1 SAFETY MARGIN: Apply 10% wider scales during first epoch
        epoch1_margin = 1.10 if epoch == 1 else 1.0

        for name, spec in self.specs.items():
            ctype = spec['type']
            if ctype == ConstraintType.MINIMIZE_SOFT and spec.get('scale') == 'auto':
                if self.samples.get(name):
                    median = np.median(self.samples[name])
                    min_val = np.min(self.samples[name])
                    max_val = np.max(self.samples[name])
                    mean_val = np.mean(self.samples[name])
                    # Use max if median is near zero (prevents over-sensitivity)
                    # Minimum scale 1e-3 to prevent numerical instability
                    if median < 1e-4:
                        self.scales[name] = max(max_val, 1e-3)
                    else:
                        self.scales[name] = max(median, 1e-3)

                    # Apply epoch 1 safety margin
                    self.scales[name] *= epoch1_margin

                    self.normalizers[name] = make_normalizer_torch(ctype, scale=self.scales[name])
                    margin_note = " (+10% margin)" if epoch == 1 else ""
                    print(f"  {name:20s}: scale={self.scales[name]:.4f}{margin_note} | raw: [{min_val:.4f}, {max_val:.4f}] mean={mean_val:.4f}")
                else:
                    self.scales[name] = 1.0 * epoch1_margin
                    self.normalizers[name] = make_normalizer_torch(ctype, scale=self.scales[name])
                    margin_note = " (+10% margin)" if epoch == 1 else ""
                    print(f"  {name:20s}: scale={self.scales[name]:.4f}{margin_note} (no samples)")
            elif ctype == ConstraintType.MINIMIZE_SOFT:
                self.scales[name] = spec['scale']
                self.normalizers[name] = make_normalizer_torch(ctype, scale=spec['scale'])
                if self.samples.get(name):
                    min_val = np.min(self.samples[name])
                    max_val = np.max(self.samples[name])
                    mean_val = np.mean(self.samples[name])
                    median = np.median(self.samples[name])
                    print(f"  {name:20s}: scale={spec['scale']:.4f} (fixed) | raw: [{min_val:.4f}, {max_val:.4f}] mean={mean_val:.4f}")
                else:
                    print(f"  {name:20s}: scale={spec['scale']:.4f} (fixed)")
            elif ctype == ConstraintType.BOX:
                self.normalizers[name] = make_normalizer_torch(ctype, lower=spec['lower'], upper=spec['upper'])
                if self.samples.get(name):
                    median = np.median(self.samples[name])
                    min_val = np.min(self.samples[name])
                    max_val = np.max(self.samples[name])
                    print(f"  {name:20s}: BOX [{spec['lower']:.2f}, {spec['upper']:.2f}] | init=[{min_val:.2f}, {max_val:.2f}] median={median:.2f}")
                    if min_val < spec['lower'] or max_val > spec['upper']:
                        box_violations.append(f"{name}: init range [{min_val:.2f}, {max_val:.2f}] outside BOX [{spec['lower']:.2f}, {spec['upper']:.2f}]")
                else:
                    print(f"  {name:20s}: BOX [{spec['lower']:.2f}, {spec['upper']:.2f}] (no samples)")
            elif ctype == ConstraintType.BOX_ASYMMETRIC:
                self.normalizers[name] = make_normalizer_torch(ctype, lower=spec['lower'], upper=spec['upper'], healthy=spec['healthy'])
                if self.samples.get(name):
                    median = np.median(self.samples[name])
                    min_val = np.min(self.samples[name])
                    max_val = np.max(self.samples[name])
                    print(f"  {name:20s}: BOX_ASYM [{spec['lower']:.0f}, {spec['upper']:.0f}] h={spec['healthy']:.0f} | init=[{min_val:.0f}, {max_val:.0f}] median={median:.0f}")
                    if min_val < spec['lower'] or max_val > spec['upper']:
                        box_violations.append(f"{name}: init range [{min_val:.0f}, {max_val:.0f}] outside BOX [{spec['lower']:.0f}, {spec['upper']:.0f}]")
                else:
                    print(f"  {name:20s}: BOX_ASYM [{spec['lower']:.0f}, {spec['upper']:.0f}] h={spec['healthy']:.0f} (no samples)")
            elif ctype == ConstraintType.LOWER:
                self.normalizers[name] = make_normalizer_torch(ctype, **{k: v for k, v in spec.items() if k != 'type'})
                if self.samples.get(name):
                    margin = spec['margin']
                    min_val = np.min(self.samples[name])
                    max_val = np.max(self.samples[name])
                    mean_val = np.mean(self.samples[name])
                    median = np.median(self.samples[name])
                    print(f"  {name:20s}: LOWER(margin={margin:.0f}) | raw: [{min_val:.0f}, {max_val:.0f}] mean={mean_val:.0f} median={median:.0f}")
                else:
                    print(f"  {name:20s}: LOWER(margin={spec['margin']:.0f}) (no samples)")
            else:
                self.normalizers[name] = make_normalizer_torch(ctype, **{k: v for k, v in spec.items() if k != 'type'})

        if box_violations:
            print("\n⚠️  WARNING: BOX CONSTRAINT VIOLATIONS ⚠️")
            for violation in box_violations:
                print(f"    {violation}")
            print("    These constraints will return goal=0 → loss=inf → crash!")
            print("    ACTION: Widen BOX bounds to contain initialization values.\n")

        # Track if margin was applied
        if epoch == 1:
            self.epoch1_margin_applied = True
            print("\n⚠️  EPOCH 1 SAFETY MARGIN: All auto-scaled goals widened by 10%")
            print("    Will be removed at start of Epoch 2 to tighten constraints\n")

        print("=" * 60 + "\n")
        self.calibrated = True
        self.samples = {name: [] for name in self.specs.keys()}
    
    def goal(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if name not in self.normalizers:
            return 1.0 / (1.0 + torch.clamp(value, min=0.0))
        return self.normalizers[name](value)
    
    def start_recalibration(self):
        self.samples = {name: [] for name in self.specs.keys()}

    def rebuild_normalizers(self):
        """Rebuild normalizers from current specs (used after tightening)"""
        for name, spec in self.specs.items():
            ctype = spec['type']
            if ctype == ConstraintType.MINIMIZE_SOFT:
                scale = self.scales.get(name, spec.get('scale', 1.0))
                self.normalizers[name] = make_normalizer_torch(ctype, scale=scale)
            elif ctype == ConstraintType.BOX:
                self.normalizers[name] = make_normalizer_torch(ctype, lower=spec['lower'], upper=spec['upper'])
            elif ctype == ConstraintType.BOX_ASYMMETRIC:
                self.normalizers[name] = make_normalizer_torch(ctype, lower=spec['lower'], upper=spec['upper'], healthy=spec['healthy'])
            elif ctype == ConstraintType.LOWER:
                self.normalizers[name] = make_normalizer_torch(ctype, **{k: v for k, v in spec.items() if k != 'type'})
            else:
                self.normalizers[name] = make_normalizer_torch(ctype, **{k: v for k, v in spec.items() if k != 'type'})

    def remove_epoch1_margin(self):
        """Remove the 10% safety margin applied during epoch 1 calibration"""
        if not self.epoch1_margin_applied:
            print("⚠️  No epoch 1 margin to remove (was not applied)")
            return

        print("\n" + "=" * 60)
        print("🔧 REMOVING EPOCH 1 SAFETY MARGIN")
        print("=" * 60)
        print("Tightening auto-scaled goals from calibrated+10% → calibrated values\n")

        for name, spec in self.specs.items():
            if spec['type'] == ConstraintType.MINIMIZE_SOFT and spec.get('scale') == 'auto':
                old_scale = self.scales[name]
                self.scales[name] /= 1.10  # Remove 10% margin
                print(f"  {name:20s}: {old_scale:.4f} → {self.scales[name]:.4f} (-9.1%)")

        # Rebuild normalizers with tightened scales
        self.rebuild_normalizers()
        self.epoch1_margin_applied = False

        print("\n✅ All auto-scaled constraints tightened to calibrated values")
        print("=" * 60 + "\n")

def geometric_mean(goals):
    """Geometric mean with epsilon protection against underflow"""
    goals = torch.stack(goals)
    # Clamp to prevent underflow in product (min value = 1e-10 → prod^(1/n) ≥ 1e-10)
    goals = torch.clamp(goals, min=1e-10)
    product = goals.prod()
    return product ** (1.0 / len(goals))
