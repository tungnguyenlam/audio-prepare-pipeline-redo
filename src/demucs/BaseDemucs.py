"""Base Demucs separation module.

Defines the interface for demucs (music removal): consumes an ``Audio`` object
and returns a separated ``Audio`` object of the same class.
"""

from __future__ import annotations

import abc
import shutil
from pathlib import Path
from typing import Optional

from src.utils.AudioClass import Audio


class BaseDemucs(abc.ABC):
    """Abstract base class for Demucs separation backends.

    Subclasses implement :meth:`demucs`, which receives an ``Audio`` object and
    returns a demucsed ``Audio`` object (same class) written to ``output_dir``.
    """

    def __init__(
        self,
        model: str = "htdemucs",
        device: str = "cpu",
        two_stems: str = "vocals",
        output_dir: str | Path = ".data/demucsed",
        work_dir: str | Path = "./temp_demucs",
        sample_rate: int = 16000,
        channels: int = 1,
        demucs_bin: Optional[str] = None,
        ffmpeg_bin: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.two_stems = two_stems
        self.output_dir = Path(output_dir)
        self.work_dir = Path(work_dir)
        self.sample_rate = sample_rate
        self.channels = channels
        self.demucs_bin = demucs_bin
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"

    @abc.abstractmethod
    def demucs(self, audio: Audio) -> Audio:
        """Separate an ``Audio`` object and return a demucsed ``Audio`` instance."""