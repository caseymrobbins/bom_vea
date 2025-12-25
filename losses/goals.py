# losses/goals.py
# BOM Goal System - auto-calibrating constraint satisfaction

import torch
import numpy as np
from typing import Dict, Callable
from configs.config import ConstraintType


def make_normalizer_torch(ctype: ConstraintType, **kwargs) -> Callable:
    """Create PyTorch-compatible normalizer function."""
    
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
        dist_lower = healthy - lower
        dist_upper = upper - healthy
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
        scale = kwargs["scale"]
        return lambda x: 1.0 / (1.0 + torch.clamp(x, min=0.0) / scale) if scale > 0 else torch.zeros_like(x)
    
    if ctype == ConstraintType.MINIMIZE_HARD:
        scale = kwargs.get("scale", 1.0)
        return lambda x: 1.0 / (1.0 + (torch.clamp(x, min=0.0) / scale) ** 2)
    
    raise ValueError(f"Unknown constraint type: {ctype}")


class GoalSystem:
    """
    BOM Goal System with auto-calibration.
    
    Goals are normalized to [0, 1] where:
    - 1.0 = goal fully satisfied
    - 0.0 = goal completely violated
    
    BOM maximizes log(min(goals)) - the worst goal drives learning.
    """
    
    def __init__(self, goal_specs: Dict):
        self.specs = goal_specs
        self.scales = {}
        self.normalizers = {}
        self.samples = {name: [] for name in goal_specs.keys()}
        self.calibrated = False
        self.calibration_count = 0
    
    def collect(self, loss_dict: Dict[str, float]):
        """Collect samples during calibration phase."""
        for name, value in loss_dict.items():
            if name in self.samples:
                if not np.isnan(value) and not np.isinf(value):
                    self.samples[name].append(value)
    
    def calibrate(self, epoch: int = 0):
        """Calibrate 'auto' scales from collected samples."""
        self.calibration_count += 1
        print("\n" + "=" * 70)
        print(f"CALIBRATING GOALS (#{self.calibration_count}, epoch {epoch})")
        print("=" * 70)
        
        for name, spec in self.specs.items():
            ctype = spec['type']
            
            if ctype == ConstraintType.MINIMIZE_SOFT and spec.get('scale') == 'auto':
                if self.samples.get(name):
                    median = np.median(self.samples[name])
                    self.scales[name] = max(median, 1e-6)
                    self.normalizers[name] = make_normalizer_torch(ctype, scale=self.scales[name])
                    print(f"  {name:20s}: MINIMIZE_SOFT scale={self.scales[name]:.4f}")
                else:
                    self.scales[name] = 1.0
                    self.normalizers[name] = make_normalizer_torch(ctype, scale=1.0)
                    print(f"  {name:20s}: MINIMIZE_SOFT scale=1.0 (no samples)")
                    
            elif ctype == ConstraintType.MINIMIZE_SOFT:
                scale = spec['scale']
                self.scales[name] = scale
                self.normalizers[name] = make_normalizer_torch(ctype, scale=scale)
                print(f"  {name:20s}: MINIMIZE_SOFT scale={scale:.4f} (fixed)")
                
            elif ctype == ConstraintType.BOX:
                lower, upper = spec['lower'], spec['upper']
                self.normalizers[name] = make_normalizer_torch(ctype, lower=lower, upper=upper)
                print(f"  {name:20s}: BOX [{lower:.2f}, {upper:.2f}]")
                
            elif ctype == ConstraintType.BOX_ASYMMETRIC:
                lower, upper, healthy = spec['lower'], spec['upper'], spec['healthy']
                self.normalizers[name] = make_normalizer_torch(ctype, lower=lower, upper=upper, healthy=healthy)
                print(f"  {name:20s}: BOX_ASYM [{lower:.0f}, {upper:.0f}] healthy={healthy:.0f}")
                
            else:
                print(f"  {name:20s}: {ctype.value}")
                self.normalizers[name] = make_normalizer_torch(ctype, **{k: v for k, v in spec.items() if k != 'type'})
        
        print("=" * 70 + "\n")
        self.calibrated = True
        self.samples = {name: [] for name in self.specs.keys()}
    
    def goal(self, value: torch.Tensor, name: str) -> torch.Tensor:
        """Compute goal score for a value."""
        if name not in self.normalizers:
            return 1.0 / (1.0 + torch.clamp(value, min=0.0))
        return self.normalizers[name](value)
    
    def start_recalibration(self):
        """Reset samples for recalibration."""
        self.samples = {name: [] for name in self.specs.keys()}


def geometric_mean(goals):
    """Compute geometric mean of goals."""
    goals = torch.stack(goals)
    return goals.prod() ** (1.0 / len(goals))
