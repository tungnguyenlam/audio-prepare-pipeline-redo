"""Notebook helpers for displaying pipeline ``Audio`` objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from src.utils.AudioClass import Audio


def display_audio(
    audio: Audio,
    dest: Optional[Union[str, Path]] = None,
) -> None:
    """Save ``audio`` to disk, wrap it in IPython's player, and display it.

    Args:
        audio: Pipeline ``Audio`` instance (file-backed).
        dest: Optional destination file or directory. When provided, the audio
            is copied there via ``Audio.save_to`` before playback. When omitted,
            the existing ``audio.path`` is used.
    """
    from IPython.display import Audio as IPythonAudio
    from IPython.display import display

    if dest is not None:
        audio = audio.save_to(dest)

    path = Path(audio.path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    # display() only — do not return the player, or Jupyter renders it twice
    display(IPythonAudio(filename=str(path), rate=audio.sample_rate))
