"""
Diffusion Img2Img Attack: Globale Rekonstruktion durch Stable Diffusion.

Referenz:
- Diffusers Doku: siehe repo/diffusers/
"""

import numpy as np
from PIL import Image
from .base import BaseAttack


class DiffusionAttack(BaseAttack):
    """Stable Diffusion img2img Attack."""
    
    name = "diffusion"
    
    def is_enabled(self) -> bool:
        return self.config.enable_diffusion
    
    def load_model(self):
        if self._model is not None:
            return
            
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Loading Diffusion model on {device}...")
        
        self._model = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.diffusion_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
        
        if device == "cuda":
            self._model.enable_attention_slicing()
    
    def apply(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        import torch
        
        self.load_model()
        pil_image = Image.fromarray(image)
        
        with torch.no_grad():
            result = self._model(
                prompt="a high quality photo, sharp, detailed",
                image=pil_image,
                strength=self.config.diffusion_strength,
                num_inference_steps=self.config.diffusion_steps,
                guidance_scale=self.config.diffusion_guidance
            ).images[0]
        
        return np.array(result)
