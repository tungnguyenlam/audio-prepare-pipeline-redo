"""HTML Template Composer with Recursive Partial Inclusions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set

INCLUDE_REGEX = re.compile(
    r'<!--\s*(?:include:|@include|#include|include)\s*["\']?([^"\'\s>]+)["\']?\s*-->',
    re.IGNORECASE,
)


def compose_html(file_path: Path, visited: Set[Path] | None = None) -> str:
    """Compose an HTML document by recursively expanding partial include directives.

    Args:
        file_path: Path to the root HTML or partial file.
        visited: Set of visited file paths to prevent recursion cycles.

    Returns:
        Fully assembled HTML text.
    """
    file_path = file_path.resolve()
    if not file_path.is_file():
        return f"<!-- Include error: file not found {file_path.name} -->"

    if visited is None:
        visited = set()

    if file_path in visited:
        return f"<!-- Include cycle detected: {file_path.name} -->"

    visited.add(file_path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"<!-- Include read error {file_path.name}: {e} -->"

    base_dir = file_path.parent

    def replacer(match: re.Match[str]) -> str:
        rel_path = match.group(1).strip()
        target = (base_dir / rel_path).resolve()
        return compose_html(target, visited=visited.copy())

    return INCLUDE_REGEX.sub(replacer, content)
