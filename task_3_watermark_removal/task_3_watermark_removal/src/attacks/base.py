"""
Basis-Klasse für alle Attack-Schritte.
Definiert das Interface das jeder Schritt implementieren muss.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import AttackConfig


class BaseAttack(ABC):
    """Abstrakte Basis-Klasse für Attack-Schritte."""
    
    name: str = "base"
    
    def __init__(self, config: "AttackConfig"):
        self.config = config
        self._model = None  # Lazy loading
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Prüft ob dieser Schritt aktiviert ist."""
        pass
    
    def load_model(self):
        """Lädt benötigte Modelle (lazy loading). Optional zu überschreiben."""
        pass
    
    def unload_model(self):
        """Gibt Modell-Speicher frei. Optional zu überschreiben."""
        if self._model is not None:
            del self._model
            self._model = None
            # Try to free CUDA memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    @abstractmethod
    def apply(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        Wendet den Attack auf das Bild an.
        
        Args:
            image: Aktuelles Bild (kann bereits modifiziert sein)
            original: Originales Eingabebild (für Color-Fix etc.)
            
        Returns:
            Modifiziertes Bild
        """
        pass
    
    def __call__(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Führt den Attack aus, falls aktiviert."""
        if not self.is_enabled():
            return image
        return self.apply(image, original)
