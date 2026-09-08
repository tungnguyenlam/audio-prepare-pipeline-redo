"""Audio resolution and helper utilities for diarization and verification."""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_audio_path"]


def resolve_audio_path(audio_path: str | Path, repo_root: Path) -> Path:
    """Resolve audio path across absolute, relative, or shared workspace locations.

    Args:
        audio_path: Absolute or relative audio file path.
        repo_root: Root path of the repository for relative path resolution.

    Returns:
        Path pointing to the resolved existing audio file.

    Raises:
        FileNotFoundError: If the file cannot be located.
    """
    p = Path(audio_path)
    if p.is_file():
        return p
    candidate = repo_root / audio_path
    if candidate.is_file():
        return candidate
    if ".data" in str(audio_path):
        rel_data = str(audio_path).split(".data/")[-1]
        candidate = repo_root / ".data" / rel_data
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Audio file not found: {audio_path} (resolved from {repo_root})")
