"""Watermark Removal Attack Pipeline."""

from .config import AttackConfig, PRESETS
from .pipeline import AttackPipeline

__all__ = ["AttackConfig", "PRESETS", "AttackPipeline"]
