"""
Rotation Attack: Kleine Rotation um Bit-Alignment zu stören.
BitMark ist nicht rotations-invariant!
"""

import numpy as np
from .base import BaseAttack


class RotationAttack(BaseAttack):
    """Kleine Rotation Attack."""
    
    name = "rotation"
    
    def is_enabled(self) -> bool:
        return self.config.enable_rotation
    
    def apply(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        from scipy.ndimage import rotate
        
        angle = self.config.rotation_angle
        rotated = rotate(image, angle, reshape=False, order=1, mode='reflect')
        return np.clip(rotated, 0, 255).astype(np.uint8)
