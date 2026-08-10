"""Demucs CLI separation implementation.

Extracted from the audio-prepare-pipeline ``DemucsCleaner`` (ADR-0007):
vocal separation through the ``demucs`` CLI, then ffmpeg normalization to the
target sample rate / channel count.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from demucs.BaseDemucs import BaseDemucs
from src.utils.AudioClass import Audio

logger = logging.getLogger(__name__)


class DemucsError(RuntimeError):
    """Raised when Demucs separation cannot run or the output is unusable."""


def probe_wav(path: Path) -> tuple[int, float, int]:
    """Return ``(sample_rate, duration_s, channels)`` for a WAV file."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.getnframes()
        channels = wf.getnchannels()
    duration = frames / float(rate) if rate else 0.0
    return rate, duration, channels


class HTDemucs(BaseDemucs):
    """Live Demucs backend: separates via the ``demucs`` CLI (vocals by default).

    Usage:
        demucs = HTDemucs(output_dir="./data/demucsed")
        separated = demucs.demucs(audio)  # returns a new Audio instance
    """

    def _demucs_prefix(self) -> list[str]:
        """Binary prefix: configured binary or ``python -m demucs``."""
        if self.demucs_bin:
            return [self.demucs_bin]
        return [sys.executable, "-m", "demucs"]

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
            self.device,
            "-o",
            str(out_root),
            str(audio_path),
        ]
        logger.info(f"Running demucs: {' '.join(cmd)}")
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

    def _normalize(self, src: Path, dest: Path) -> None:
        """Convert to the target sample rate / channels with ffmpeg."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() == ".wav":
            try:
                rate, _, ch = probe_wav(src)
                if rate == self.sample_rate and ch == self.channels:
                    if src.resolve() != dest.resolve():
                        shutil.copy2(src, dest)
                    return
            except wave.Error:
                pass

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(src),
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise DemucsError(
                f"ffmpeg convert failed (exit {completed.returncode}): {detail[:2000] or 'no output'}"
            )

    def demucs(self, audio: Audio) -> Audio:
        """Separate ``audio`` and return a demucsed ``Audio`` object."""
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
        self._normalize(separated, dest)

        sample_rate, duration_s, channels = probe_wav(dest)
        return Audio(
            path=dest.resolve(),
            source_id=audio.source_id,
            title=audio.title,
            sample_rate=sample_rate,
            duration_s=duration_s,
            channels=channels,
            format="wav",
        )