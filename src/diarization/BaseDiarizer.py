"""Backend-independent interface for speaker diarization."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.diarization.schemas import DiarizationResult
from src.utils.AudioClass import Audio


class BaseDiarizer(ABC):
    """Abstract interface implemented by speaker diarization backends."""

    @abstractmethod
    def diarize(self, audio: Audio) -> DiarizationResult:
        """Detect speaker turns in an audio item."""

