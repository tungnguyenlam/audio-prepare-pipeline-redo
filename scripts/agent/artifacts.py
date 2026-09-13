"""Shared artifact contract for free-form agent behavior experiments.

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

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    from _common.files import (  # noqa: E402
        batch,
        digest,
        positive_int,
        read_json,
        request,
        write_json,
    )
except ImportError:
    from scripts._common.files import (  # noqa: E402
        batch,
        digest,
        positive_int,
        read_json,
        request,
        write_json,
    )


def read_prompt(path: Path, label: str = "Prompt") -> str:
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Safely write text to destination via an atomic temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tf:
        tf.write(text)
        tmp_txt = Path(tf.name)
    tmp_txt.replace(path)


def add_prompt_arguments(
    command: argparse.ArgumentParser, *, top_p: bool = False
) -> None:
    """Add options shared by every free-form generation backend."""
    command.add_argument(
        "--prompt-file",
        required=True,
        type=Path,
        help="Path to text prompt file to send to model",
    )
    command.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Optional system instruction prompt file",
    )
    command.add_argument(
        "--max-tokens",
        type=positive_int,
        default=4096,
        help="Maximum number of tokens to generate",
    )
    command.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    if top_p:
        command.add_argument(
            "--top-p",
            type=float,
            default=None,
            help="Nucleus sampling top-p probability threshold",
        )


def load_prompts(args: argparse.Namespace) -> tuple[str, str | None]:
    """Read prompt and optional system prompt from user files."""
    prompt = read_prompt(args.prompt_file, label="Prompt")
    system_prompt = (
        read_prompt(args.system_prompt_file, label="System prompt")
        if args.system_prompt_file
        else None
    )
    return prompt, system_prompt


def run_agent(
    args: argparse.Namespace,
    pairs: list[tuple[Path, Path]],
    parameters: dict[str, Any],
    runner: Callable[[Path], dict[str, Any]],
    logger: Any,
) -> int:
    """Run generation across pairs using the standard batch pipeline."""
    operation = "explore_audio_model"
    model = getattr(args, "model", None) or "hf"

    def process_item(src: Path, dst: Path) -> None:
        res = runner(src)

        write_text(dst, res["text"])

        sidecar_path = dst.with_suffix(".json")
        sidecar_payload = {
            "schema_version": 1,
            "source": {
                "path": str(src.resolve()),
                "sha256": digest(src),
            },
            "operation": operation,
            "model": model,
            "parameters": parameters,
            "response": {
                "latency_s": res.get("latency_s", 0.0),
                "provider_body": res.get("provider_body", {}),
            },
            "output": {
                "path": str(dst.resolve()),
                "format": "utf-8 text",
                "bytes": dst.stat().st_size,
                "sha256": digest(dst),
            },
        }
        write_json(sidecar_path, sidecar_payload)

    return batch(
        pairs,
        process_item,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
