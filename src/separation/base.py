"""Base abstract class and definitions for audio source separation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


class SeparationError(RuntimeError):
    """Raised when audio source separation fails."""


@dataclass(frozen=True)
class SeparationStem:
    """Represents a single separated audio stem."""

    name: str  # e.g. "vocals", "accompaniment"
    path: Path
    filename: str
    filesize: int


@dataclass(frozen=True)
class SeparationResult:
    """Results produced by a separation model execution."""

    model: str
    input_file: Path
    output_dir: Path
    stems: Dict[str, Path]


class BaseSeparator(ABC):
    """Abstract Base Class for Separation Models."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    @abstractmethod
    def check_status(self) -> dict[str, Union[bool, str]]:
        """Check whether the model dependencies and weights are available."""
        pass

    @abstractmethod
    def separate(self, input_path: Path, output_dir: Path) -> SeparationResult:
        """Separate input audio into isolated stems (e.g. vocals, accompaniment).

        Args:
            input_path: Path to input audio (.wav, .mp3, etc.)
            output_dir: Path to directory where stems should be saved

        Returns:
            SeparationResult containing mapping of stem names to output file paths.
        """
        pass
