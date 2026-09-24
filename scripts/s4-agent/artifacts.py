"""Shared artifact contract for free-form agent behavior experiments.

This module deliberately avoids the name ``_common`` because commands in this
directory also import the top-level ``scripts/_common`` namespace package.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    from _common.files import (  # noqa: E402
        batch,
        digest,
        persist_path,
        positive_int,
        progress,
        write_json,
    )
except ImportError:
    from scripts._common.files import (  # noqa: E402
        batch,
        digest,
        persist_path,
        positive_int,
        progress,
        write_json,
    )

from _reporting import item_detail, new_run_stats, record_result, report_cost_summary


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
    command: argparse.ArgumentParser,
    *,
    max_tokens: int = 4096,
    sampling: bool = True,
) -> None:
    """Add options shared by every free-form generation backend.

    ``sampling=False`` omits temperature/top-p for providers that ignore them.
    """
    command.add_argument(
        "-pf", "-p",
        "--prompt-file",
        required=True,
        type=Path,
        help="Path to text prompt file to send to model",
    )
    command.add_argument(
        "-spf",
        "--system-prompt-file",
        type=Path,
        help="Optional system instruction prompt file",
    )
    command.add_argument(
        "-mt",
        "--max-tokens",
        type=positive_int,
        default=max_tokens,
        help="Maximum number of tokens to generate",
    )
    if sampling:
        command.add_argument(
            "-t",
            "--temperature",
            type=float,
            default=0.0,
            help="Sampling temperature",
        )
        command.add_argument(
            "-tp",
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
    *,
    args: argparse.Namespace,
    pairs: list[tuple[Path, Path]],
    backend: str,
    parameters: dict[str, Any],
    generate: Callable[[Path], dict[str, Any]],
    cost_summary: Callable[[], Mapping[str, Any]] | None = None,
) -> int:
    """Run generation, persist raw responses, and report per-item/cumulative costs."""
    operation = "explore_audio_model"
    model = (
        parameters.get("model")
        or parameters.get("model_id")
        or getattr(args, "model", None)
        or backend
    )
    stats = new_run_stats()
    stats['attempted'] = len(pairs)
    progress(
        'AGENT_START',
        f'backend={backend}; model={model}; items={len(pairs)}',
    )

    def process_item(src: Path, dst: Path) -> None:
        generation_error = None
        res = None
        try:
            try:
                res = generate(src)
            except Exception as exc:
                # Providers can attach a response even when generation is unusable.
                res = getattr(exc, "generation", None)
                if not isinstance(res, dict):
                    raise
                generation_error = exc
            if not isinstance(res, dict) or not isinstance(res.get('text'), str):
                raise TypeError('Agent generator must return a dict with string text')

            write_text(dst, res["text"])

            sidecar_path = dst.with_suffix(".json")
            sidecar_payload = {
                "schema_version": 1,
                "source": {
                    "path": persist_path(src),
                    "sha256": digest(src),
                },
                "operation": operation,
                "model": model,
                "parameters": parameters,
                "response": {
                    key: value for key, value in res.items() if key != "text"
                },
                "output": {
                    "path": persist_path(dst),
                    "format": "utf-8 text",
                    "bytes": dst.stat().st_size,
                    "sha256": digest(dst),
                },
            }
            if generation_error is not None:
                sidecar_payload["status"] = "fail"
                sidecar_payload["error"] = res["generation_error"]
            write_json(sidecar_path, sidecar_payload)
            if generation_error is not None:
                raise generation_error
            running_total = record_result(stats, res, success=True)
            progress(
                'AGENT_RESULT',
                f'{src.name}: {item_detail(res, text_length=len(res["text"]), running_total_usd=running_total)}',
            )
        except Exception:
            record_result(stats, res, success=False)
            raise

    try:
        result = batch(
            pairs,
            process_item,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
        )
    finally:
        provider = cost_summary() if cost_summary is not None else None
        report_cost_summary(f'agent/{backend}', stats, provider)
    return result
