"""Demucs CLI separation implementation.

Vocal separation through the ``demucs`` CLI, then ffmpeg normalization to the
target sample rate / channel count.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.separation.audio_utils import normalize_wav, probe_wav
from src.separation.BaseSeparator import BaseSeparator
from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio

logger = logging.getLogger(__name__)


class DemucsError(RuntimeError):
    """Raised when Demucs separation cannot run or the output is unusable."""


class HTDemucs(BaseSeparator):
    """Live Demucs backend: separates via the ``demucs`` CLI (vocals by default).

    Usage:
        separator = HTDemucs(output_dir=".data/demucs/out")
        cleaned = separator.separate(audio)
    """

    def __init__(
        self,
        model: str = "htdemucs",
        device: str = "cpu",
        two_stems: str = "vocals",
        output_dir: str | Path = ".data/demucs/out",
        work_dir: str | Path = ".data/demucs/work",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        demucs_bin: Optional[str] = None,
        ffmpeg_bin: Optional[str] = None,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            two_stems=two_stems,
            output_dir=output_dir,
            work_dir=work_dir,
            sample_rate=sample_rate,
            channels=channels,
            ffmpeg_bin=ffmpeg_bin,
        )
        self.demucs_bin = demucs_bin

    def _demucs_prefix(self) -> list[str]:
        """Binary prefix: configured binary, PATH binary, or ``python -m demucs.separate``."""
        if self.demucs_bin:
            return [self.demucs_bin]
        bin_path = shutil.which("demucs")
        if bin_path:
            return [bin_path]
        venv_bin = Path(sys.prefix) / "bin" / "demucs"
        if venv_bin.is_file():
            return [str(venv_bin)]
        return [sys.executable, "-m", "demucs.separate"]

    def _separate_stem(self, audio_path: Path) -> Path:
        """Run Demucs and return the extracted stem WAV path."""
        out_root = self.work_dir / "demucs_out"
        out_root.mkdir(parents=True, exist_ok=True)
        cmd = self._demucs_prefix() + [
            "-n",
            self.model,
            "--two-stems",
            self.two_stems,
            "-d",
            str(self.device),
            "-o",
            str(out_root),
            str(audio_path),
        ]
        logger.info("Running demucs: %s", " ".join(cmd))
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise DemucsError(
                f"demucs failed (exit {completed.returncode}): {detail[:2000] or 'no output'}"
            )

        separated = out_root / self.model / audio_path.stem / f"{self.two_stems}.wav"
        if not separated.is_file():
            matches = list(out_root.rglob(f"{self.two_stems}.wav"))
            if not matches:
                raise DemucsError(
                    f"demucs finished but {self.two_stems}.wav not found under {out_root}"
                )
            separated = matches[0]
        return separated

    def separate(self, audio: Audio) -> Audio:
        """Separate ``audio`` and return a cleaned ``Audio`` object."""
        src_path = Path(audio.path)
        if not src_path.is_file():
            raise DemucsError(f"audio not found: {src_path}")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            separated = self._separate_stem(src_path)
        except FileNotFoundError as exc:  # pragma: no cover
            raise DemucsError(
                "demucs is required for separation; install with: pip install demucs"
            ) from exc

        dest = self.output_dir / f"{src_path.stem}.wav"
        normalize_wav(
            separated,
            dest,
            sample_rate=self.sample_rate,
            channels=self.channels,
            ffmpeg_bin=self.ffmpeg_bin,
        )

        sample_rate, duration_s, channels = probe_wav(dest)
        return audio.with_file(
            dest,
            sample_rate=sample_rate,
            duration_s=duration_s,
            channels=channels,
        )
