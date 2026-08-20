"""Audio Source Separation module (Stage 2 in Audio Processing Pipeline)."""

from src.separation.base import BaseSeparator, SeparationError, SeparationResult, SeparationStem
from src.separation.htdemucs import HTDemucsSeparator
from src.separation.mel_roformer import MelRoFormerSeparator
from src.separation.deepfilternet import DeepFilterNetSeparator
from src.separation.manager import SeparationManager

__all__ = [
    "BaseSeparator",
    "SeparationError",
    "SeparationResult",
    "SeparationStem",
    "HTDemucsSeparator",
    "MelRoFormerSeparator",
    "DeepFilterNetSeparator",
    "SeparationManager",
]
