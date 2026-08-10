"""Shared audio helpers for separation backends."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path


class AudioConvertError(RuntimeError):
    """Raised when ffmpeg conversion/normalization fails."""


def probe_wav(path: Path) -> tuple[int, float, int]:
    """Return ``(sample_rate, duration_s, channels)`` for a WAV file."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.getnframes()
        channels = wf.getnchannels()
        duration = frames / rate if rate > 0 else 0.0
        return rate, duration, channels


def normalize_wav(
    src: Path,
    dest: Path,
    *,
    sample_rate: int,
    channels: int,
    ffmpeg_bin: str,
) -> None:
    """Convert audio to the pipeline-standard sample rate / channel layout."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".wav":
        try:
            rate, _, ch = probe_wav(src)
            if rate == sample_rate and ch == channels:
                if src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                return
        except wave.Error:
            pass

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise AudioConvertError(
            f"ffmpeg convert failed (exit {completed.returncode}): "
            f"{detail[:2000] or 'no output'}"
        )
