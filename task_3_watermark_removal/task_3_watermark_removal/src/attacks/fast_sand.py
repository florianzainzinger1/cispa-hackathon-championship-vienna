"""
Fast-Sand Attack: Vereinfachte Version von "Watermarks in the Sand"
Iteratives Patch-Inpainting um lokale Watermark-Bits zu zerstören.

Referenz: 
- Paper: https://arxiv.org/abs/2404.04727
- Diffusers Doku: siehe repo/diffusers/
"""

import numpy as np
from PIL import Image
from .base import BaseAttack


class FastSandAttack(BaseAttack):
    """Iteratives Patch-Inpainting Attack."""
    
    name = "fast_sand"
    
    def is_enabled(self) -> bool:
        return self.config.enable_fast_sand
    
    def load_model(self):
        if self._model is not None:
            return
            
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Loading Inpainting model on {device}...")
        
        self._model = StableDiffusionInpaintPipeline.from_pretrained(
            self.config.inpainting_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
        
        # Speicher-Optimierungen
        if device == "cuda":
            self._model.enable_attention_slicing()
    
    def _create_patch_mask(self, h: int, w: int) -> np.ndarray:
        """Erstellt zufällige Patch-Maske."""
        mask = np.zeros((h, w), dtype=np.float32)
        patch_size = self.config.sand_patch_size
        
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        total_patches = num_patches_h * num_patches_w
        num_to_mask = int(total_patches * self.config.sand_coverage)
        
        all_patches = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]
        selected = np.random.choice(len(all_patches), size=min(num_to_mask, len(all_patches)), replace=False)
        
        for idx in selected:
            i, j = all_patches[idx]
            y_start, x_start = i * patch_size, j * patch_size
            mask[y_start:y_start+patch_size, x_start:x_start+patch_size] = 1.0
        
        return mask
    
    def apply(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        import torch
        
        self.load_model()
        h, w = image.shape[:2]
        current = image.copy()
        
        for i in range(self.config.sand_iterations):
            print(f"      Sand iteration {i+1}/{self.config.sand_iterations}")
            
            mask = self._create_patch_mask(h, w)
            
            pil_image = Image.fromarray(current)
            pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
            
            with torch.no_grad():
                result = self._model(
                    prompt="high quality photo, detailed, sharp",
                    image=pil_image,
                    mask_image=pil_mask,
                    num_inference_steps=self.config.sand_steps,
                    guidance_scale=self.config.diffusion_guidance
                ).images[0]
            
            current = np.array(result)
        
        return current
