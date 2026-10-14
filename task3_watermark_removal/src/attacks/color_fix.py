"""
Color-Fix Attack: Histogram-Matching um MSE zu minimieren.
Wichtig um die Qualitäts-Constraint (MSE ≤ 0.08) einzuhalten!
"""

import numpy as np
from .base import BaseAttack


class ColorFixAttack(BaseAttack):
    """Histogram Matching Attack."""
    
    name = "color_fix"
    
    def is_enabled(self) -> bool:
        return self.config.enable_color_fix
    
    def apply(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        from skimage.exposure import match_histograms
        
        matched = match_histograms(image, original, channel_axis=2)
        return np.clip(matched, 0, 255).astype(np.uint8)
