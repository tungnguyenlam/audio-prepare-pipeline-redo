"""Shared artifact contract for freeform audio-model experiments.

This module deliberately avoids the name ``_common`` because commands in this
directory also import the top-level ``scripts/_common`` namespace package.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts._common.files import (  # noqa: E402
    batch,
    digest,
    identity,
    positive_int,
    read_json,
    request,
    write_json,
)


def read_prompt(path: Path, label: str = "Prompt") -> str:
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    return path.read_text(encoding="utf-8")


def add_prompt_arguments(
    command: argparse.ArgumentParser, *, top_p: bool = False
) -> None:
    """Add options shared by every free-form generation backend."""
    command.add_argument("--prompt-file", type=Path, required=True)
    command.add_argument("--system-prompt-file", type=Path)
    command.add_argument("--max-tokens", type=positive_int, default=4096)
    command.add_argument("--temperature", type=float, default=0.0)
    if top_p:
        command.add_argument("--top-p", type=float)


def load_prompts(args: Any) -> tuple[str, str | None]:
    prompt = read_prompt(args.prompt_file)
    system_prompt = (
        read_prompt(args.system_prompt_file, "System prompt")
        if args.system_prompt_file is not None
        else None
    )
    return prompt, system_prompt


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_freeform(
    *,
    args: Any,
    pairs: list[tuple[Path, Path]],
    backend: str,
    parameters: dict[str, Any],
    generate: Callable[[Path], dict[str, Any]],
) -> int:
    """Persist unparsed text and the model/provider details returned by generate."""

    def process(source: Path, destination: Path) -> None:
        if destination.suffix.lower() == ".json":
            raise ValueError(
                "Freeform output must not use .json; that suffix is reserved for metadata"
            )
        metadata_path = destination.with_suffix(".json")
        wanted = request(identity(source), "explore_audio_model", parameters, backend)
        if destination.exists() or metadata_path.exists():
            if not args.overwrite and destination.is_file() and metadata_path.is_file():
                old = read_json(metadata_path)
                output = old.get("output", {})
                if all(old.get(key) == value for key, value in wanted.items()) and output.get(
                    "sha256"
                ) == digest(destination):
                    return
            if not args.overwrite:
                raise ValueError(f"Conflicting output: {destination}; use --overwrite")

        result = generate(source)
        text = result.get("text")
        if not isinstance(text, str):
            raise TypeError("Freeform generator must return text as a string")
        response = {key: value for key, value in result.items() if key != "text"}
        write_text(destination, text)
        write_json(
            metadata_path,
            {
                **wanted,
                "response": response,
                "output": {
                    "path": str(destination),
                    "format": "utf-8 text",
                    "bytes": destination.stat().st_size,
                    "sha256": digest(destination),
                },
            },
        )

    return batch(pairs, process)
