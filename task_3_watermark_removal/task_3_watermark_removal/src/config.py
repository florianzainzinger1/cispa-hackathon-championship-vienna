"""
Konfiguration für die Attack Pipeline.
Jeder Schritt kann aktiviert/deaktiviert werden.
"""

from dataclasses import dataclass, field
from typing import Optional
import copy


@dataclass
class AttackConfig:
    """Konfiguration für alle Attack-Schritte."""
    
    # ===== SCHRITT TOGGLES =====
    enable_fast_sand: bool = True
    enable_image_gs: bool = True
    enable_diffusion: bool = True
    enable_rotation: bool = True
    enable_blur: bool = True
    enable_color_fix: bool = True
    
    # ===== FAST-SAND PARAMETER =====
    sand_iterations: int = 3
    sand_patch_size: int = 128
    sand_coverage: float = 0.5
    sand_steps: int = 20
    
    # ===== IMAGE-GS PARAMETER =====
    gs_num_gaussians: int = 10000
    gs_quantize: bool = True
    gs_iterations: int = 3000
    gs_repo_path: str = "repo/image-gs"
    
    # ===== DIFFUSION PARAMETER =====
    diffusion_strength: float = 0.15
    diffusion_steps: int = 20
    diffusion_guidance: float = 7.5
    diffusion_model: str = "runwayml/stable-diffusion-v1-5"
    inpainting_model: str = "runwayml/stable-diffusion-inpainting"
    
    # ===== ROTATION PARAMETER =====
    rotation_angle: float = 1.0  # Grad
    
    # ===== BLUR PARAMETER =====
    blur_kernel: int = 3
    blur_sigma: float = 0.4
    
    # ===== QUALITY CONSTRAINT =====
    max_mse: float = 0.08
    
    # ===== SERVER CONFIG =====
    server_url: str = "http://35.192.205.84:80"
    task_id: str = "09-watermark-removal"
    api_key: Optional[str] = None


def _create_preset(base_config: dict) -> AttackConfig:
    """Helper to create preset from dict overrides."""
    return AttackConfig(**base_config)


# ===== PRESETS =====
PRESETS = {
    "minimal": AttackConfig(
        enable_fast_sand=False,
        enable_image_gs=False,
        enable_diffusion=False,
        enable_rotation=True,
        enable_blur=True,
        enable_color_fix=True,
    ),
    
    "medium": AttackConfig(
        enable_fast_sand=False,
        enable_image_gs=False,
        enable_diffusion=True,
        enable_rotation=True,
        enable_blur=True,
        enable_color_fix=True,
    ),
    
    "full": AttackConfig(
        enable_fast_sand=True,
        enable_image_gs=True,
        enable_diffusion=True,
        enable_rotation=True,
        enable_blur=True,
        enable_color_fix=True,
    ),
    
    "no_gs": AttackConfig(
        enable_fast_sand=True,
        enable_image_gs=False,  # Falls gsplat nicht funktioniert
        enable_diffusion=True,
        enable_rotation=True,
        enable_blur=True,
        enable_color_fix=True,
    ),
    
    "safe_mse": AttackConfig(
        enable_fast_sand=True,
        enable_image_gs=True,
        enable_diffusion=True,
        enable_rotation=True,
        enable_blur=True,
        enable_color_fix=True,
        # Konservative Parameter
        sand_iterations=2,
        diffusion_strength=0.10,
        rotation_angle=0.5,
        blur_sigma=0.3,
    ),
    
    "aggressive": AttackConfig(
        enable_fast_sand=True,
        enable_image_gs=True,
        enable_diffusion=True,
        enable_rotation=True,
        enable_blur=True,
        enable_color_fix=True,
        # Aggressive Parameter
        sand_iterations=4,
        diffusion_strength=0.25,
        rotation_angle=2.0,
        blur_sigma=0.5,
    ),
}
