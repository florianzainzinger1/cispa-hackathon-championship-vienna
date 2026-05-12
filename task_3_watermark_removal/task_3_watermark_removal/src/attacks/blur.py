"""
Blur Attack: Leichter Gaussian Blur um hochfrequente Watermark-Reste zu entfernen.
"""

import numpy as np
from .base import BaseAttack


class BlurAttack(BaseAttack):
    """Gaussian Blur Attack."""
    
    name = "blur"
    
    def is_enabled(self) -> bool:
        return self.config.enable_blur
    
    def apply(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        import cv2
        
        kernel = self.config.blur_kernel
        sigma = self.config.blur_sigma
        return cv2.GaussianBlur(image, (kernel, kernel), sigma)
