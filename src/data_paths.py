"""Repository-root runtime data paths and portable path serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / ".data"


def _is_path_key(key: str) -> bool:
    return key == "path" or key == "exported_cuts" or key.endswith(
        ("_path", "_paths", "_filepath", "_filepaths")
    )


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a stored path on the current machine.

    Repository-relative paths are anchored at the repository root. Legacy
    absolute paths copied from another checkout are remapped when their
    ``.data`` suffix exists in this checkout.
    """
    candidate = Path(path).expanduser()
    parts = candidate.parts
    if ".data" in parts:
        data_index = parts.index(".data")
        if candidate.is_absolute():
            try:
                candidate.relative_to(DATA_DIR)
            except ValueError:
                return DATA_DIR.joinpath(*parts[data_index + 1 :]).resolve()
        elif tuple(parts[max(0, data_index - 2) : data_index]) == (
            "src",
            "notebooks",
        ):
            return DATA_DIR.joinpath(*parts[data_index + 1 :]).resolve()
    if not candidate.is_absolute():
        return (REPO_ROOT / candidate).resolve()
    if candidate.exists():
        return candidate.resolve()
    if ".data" in parts:
        suffix = parts[parts.index(".data") + 1 :]
        return (DATA_DIR.joinpath(*suffix)).resolve()
    return candidate.resolve()


def portable_data_path(path: str | Path) -> str:
    """Return a machine-independent repository-relative path when possible."""
    resolved = resolve_data_path(path)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def portable_data_payload(value: Any, key: str = "") -> Any:
    """Recursively make conventional JSON path fields machine-independent."""
    if isinstance(value, dict):
        return {
            child_key: portable_data_payload(child_value, child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [portable_data_payload(item, key) for item in value]
    if isinstance(value, str) and _is_path_key(key):
        return portable_data_path(value)
    return value


def resolve_data_payload(value: Any, key: str = "") -> Any:
    """Recursively resolve conventional JSON path fields for this checkout."""
    if isinstance(value, dict):
        return {
            child_key: resolve_data_payload(child_value, child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [resolve_data_payload(item, key) for item in value]
    if isinstance(value, str) and _is_path_key(key):
        return str(resolve_data_path(value))
    return value
