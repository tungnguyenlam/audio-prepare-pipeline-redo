"""Backend-independent speaker diarization data structures."""

from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    Speaker,
    SpeakerTurn,
)

__all__ = [
    "DiarizationModelInfo",
    "DiarizationResult",
    "Speaker",
    "SpeakerTurn",
]
