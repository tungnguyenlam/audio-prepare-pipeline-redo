"""Shared audio utility functions for probing, conversion, and normalization."""

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


def _ffmpeg_convert_wav(
    src: Path,
    dest: Path,
    *,
    sample_rate: int,
    channels: int,
    ffmpeg_bin: str,
    codec: str,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
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
        codec,
        str(dest),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise AudioConvertError(
            f"ffmpeg convert failed (exit {completed.returncode}): "
            f"{detail[:2000] or 'no output'}"
        )


def prepare_separator_wav(
    src: Path,
    dest: Path,
    *,
    sample_rate: int,
    channels: int,
    ffmpeg_bin: str,
) -> None:
    """Convert source audio to the layout a separator checkpoint expects.

    BS-RoFormer / Mel-Band RoFormer checkpoints are trained at 44.1 kHz stereo.
    Feeding corpus 16 kHz audio unchanged warps the STFT bands and leaks bass
    into the vocal stem.
    """
    _ffmpeg_convert_wav(
        src,
        dest,
        sample_rate=sample_rate,
        channels=channels,
        ffmpeg_bin=ffmpeg_bin,
        codec="pcm_f32le",
    )


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

    _ffmpeg_convert_wav(
        src,
        dest,
        sample_rate=sample_rate,
        channels=channels,
        ffmpeg_bin=ffmpeg_bin,
        codec="pcm_s16le",
    )
