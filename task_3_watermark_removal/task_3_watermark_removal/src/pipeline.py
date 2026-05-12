"""
Haupt-Pipeline die alle Attack-Schritte orchestriert.
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

from .config import AttackConfig
from .attacks import (
    BaseAttack,
    FastSandAttack,
    ImageGSAttack,
    DiffusionAttack,
    RotationAttack,
    BlurAttack,
    ColorFixAttack,
)


class AttackPipeline:
    """Orchestriert alle Attack-Schritte."""
    
    def __init__(self, config: AttackConfig):
        self.config = config
        
        # Schritte in der richtigen Reihenfolge!
        self.steps: List[BaseAttack] = [
            FastSandAttack(config),    # 1. Sand
            ImageGSAttack(config),     # 2. GS
            DiffusionAttack(config),   # 3. Diffusion
            RotationAttack(config),    # 4. Rotation
            BlurAttack(config),        # 5. Blur
            ColorFixAttack(config),    # 6. Color-Fix
        ]
    
    def get_enabled_steps(self) -> List[str]:
        """Liste der aktivierten Schritte."""
        return [s.name for s in self.steps if s.is_enabled()]
    
    def preload_models(self):
        """Lädt alle Modelle vor (spart Zeit bei vielen Bildern)."""
        print("Preloading models...")
        for step in self.steps:
            if step.is_enabled() and hasattr(step, 'load_model'):
                print(f"  Loading {step.name}...")
                step.load_model()
    
    def unload_models(self):
        """Gibt Modell-Speicher frei."""
        for step in self.steps:
            step.unload_model()
    
    @staticmethod
    def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """Berechnet MSE zwischen zwei Bildern (normalisiert auf [0,1])."""
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2) / (255 ** 2)
    
    def attack_single(
        self, 
        image: np.ndarray, 
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Führt Pipeline auf einem einzelnen Bild aus.
        
        Returns:
            Tuple aus (attacked_image, mse_history_per_step)
        """
        original = image.copy()
        current = image.copy()
        mse_history = {"input": 0.0}
        
        for step in self.steps:
            if not step.is_enabled():
                continue
            
            if verbose:
                print(f"    [{step.name}]", end=" ", flush=True)
            
            current = step(current, original)
            
            mse = self.calculate_mse(original, current)
            mse_history[step.name] = mse
            
            if verbose:
                status = "✓" if mse <= self.config.max_mse else "⚠ HIGH"
                print(f"MSE={mse:.4f} {status}")
        
        return current, mse_history
    
    def attack_batch(
        self,
        images: List[np.ndarray],
        names: List[str],
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        """
        Führt Pipeline auf mehreren Bildern aus.
        
        Returns:
            Tuple aus (list_of_attacked_images, list_of_mse_histories)
        """
        attacked_images = []
        all_mse = []
        
        iterator = tqdm(zip(images, names), total=len(images)) if not verbose else zip(images, names)
        
        for i, (img, name) in enumerate(iterator):
            if verbose:
                print(f"\n[{i+1:3d}/{len(images)}] {name}")
            
            attacked, mse_history = self.attack_single(img, verbose=verbose)
            attacked_images.append(attacked)
            all_mse.append(mse_history)
        
        return attacked_images, all_mse
