"""Base source-separation interface.

Defines the interface for music/background removal: consumes an ``Audio``
object and returns a separated ``Audio`` object of the same class.
"""

from __future__ import annotations

import abc
import shutil
from pathlib import Path
from typing import Optional

from src.utils.AudioClass import Audio


class BaseSeparator(abc.ABC):
    """Abstract base class for source-separation backends.

    Subclasses implement :meth:`separate`, which receives an ``Audio`` object
    and returns a separated ``Audio`` object written to ``output_dir``.
    """

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        two_stems: str = "vocals",
        output_dir: str | Path = ".data/separated/out",
        work_dir: str | Path = ".data/separated/work",
        sample_rate: int = 16000,
        channels: int = 1,
        ffmpeg_bin: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = str(device)
        self.two_stems = two_stems
        self.output_dir = Path(output_dir)
        self.work_dir = Path(work_dir)
        self.sample_rate = sample_rate
        self.channels = channels
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"

    @abc.abstractmethod
    def separate(self, audio: Audio) -> Audio:
        """Separate an ``Audio`` object and return a cleaned ``Audio`` instance."""

    def close(self) -> None:
        """Release any held resources. Default is a no-op."""
