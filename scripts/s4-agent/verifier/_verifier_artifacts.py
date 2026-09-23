"""Discover saved verifier artifacts and check raw response integrity without inference."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from _common.files import digest, resolve_stored_path


def verifier_json_paths(directory: Path, *, exclude: Path | None = None) -> list[Path]:
    def walk_error(error: OSError) -> None:
        raise error

    paths = []
    for root, dirs, files in os.walk(directory, onerror=walk_error):
        dirs[:] = sorted(
            name for name in dirs
            if name not in {"plot", "plots", "work", "comparisons", "experiments", "__pycache__"}
            and not name.startswith(".") and (Path(root) / name).resolve() != exclude
        )
        paths.extend(Path(root) / name for name in sorted(files)
                     if name.endswith(".json") and not name.startswith(".")
                     and name not in {"analysis.json", "report.json"})
    return paths


def response_complete(destination: Path, artifact: dict[str, Any]) -> bool:
    response = artifact.get("response")
    if not isinstance(response, dict):
        return False
    if response.get("available") is False:
        return False
    path_value = response.get("path")
    expected_sha = response.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha, str):
        return False
    response_path = resolve_stored_path(path_value, base=destination.parent)
    return response_path.is_file() and digest(response_path) == expected_sha
