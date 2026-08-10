"""Audio dataclass representation module."""

from __future__ import annotations

import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _probe_wav(path: Path) -> tuple[int, float, int]:
    """Return ``(sample_rate, duration_s, channels)`` for a WAV file."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.getnframes()
        channels = wf.getnchannels()
    duration = frames / float(rate) if rate else 0.0
    return rate, duration, channels


@dataclass
class Audio:
    """Represents a downloaded and standardized audio file."""

    path: Path
    source_id: str
    title: Optional[str] = None
    sample_rate: Optional[int] = 16000
    duration_s: Optional[float] = None
    channels: Optional[int] = 1
    format: str = "wav"

    def __repr__(self) -> str:
        return (
            f"Audio(source_id={self.source_id!r}, title={self.title!r}, "
            f"path={str(self.path)!r}, sample_rate={self.sample_rate}, "
            f"duration_s={self.duration_s:.2f}s if self.duration_s else None, "
            f"channels={self.channels}, format={self.format!r})"
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        source_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Audio:
        """Load an audio file from disk and return an ``Audio`` instance.

        For WAV files, sample rate, duration, and channel count are probed from
        the file. Other formats keep the class defaults for those fields.

        Args:
            path: Path to an existing audio file.
            source_id: Optional identifier; defaults to the file stem.
            title: Optional display title; defaults to the file stem.

        Returns:
            Audio: Instance pointing at the resolved file path.

        Raises:
            FileNotFoundError: If ``path`` does not exist or is not a file.
        """
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        fmt = file_path.suffix.lstrip(".").lower() or "wav"
        sample_rate: Optional[int] = 16000
        duration_s: Optional[float] = None
        channels: Optional[int] = 1

        if fmt == "wav":
            try:
                sample_rate, duration_s, channels = _probe_wav(file_path)
            except wave.Error:
                pass

        return cls(
            path=file_path,
            source_id=source_id if source_id is not None else file_path.stem,
            title=title if title is not None else file_path.stem,
            sample_rate=sample_rate,
            duration_s=duration_s,
            channels=channels,
            format=fmt,
        )

    def save_to(self, dest: str | Path) -> Audio:
        """Save (copy) the audio file to a destination file path or directory.

        Args:
            dest: Target file path or directory path.

        Returns:
            Audio: This Audio instance with path updated to the destination file location.
        """
        dest_path = Path(dest)
        src_path = Path(self.path)

        if not src_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {src_path}")

        if dest_path.is_dir() or str(dest).endswith(("/", "\\")):
            dest_path.mkdir(parents=True, exist_ok=True)
            target = dest_path / src_path.name
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            target = dest_path

        if src_path.resolve() != target.resolve():
            shutil.copy2(src_path, target)

        self.path = target.resolve()
        return self
