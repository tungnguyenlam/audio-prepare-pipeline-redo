"""Shared CLI artifact handling for standalone verifier commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from _audio import DEFAULT_ACOUSTIC_PROMPT
from _common.files import ROOT, batch, identity, read_json, request, write_json


def load_prompt(prompt_file: Path | None) -> str:
    """Load an explicit prompt or the repository's default acoustic prompt."""
    if prompt_file is not None:
        if not prompt_file.is_file():
            raise ValueError(f"Prompt file not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8").strip()

    default_file = ROOT / "prompts" / "acoustic_defect.txt"
    if default_file.is_file():
        return default_file.read_text(encoding="utf-8").strip()
    return DEFAULT_ACOUSTIC_PROMPT


def resolved_parameters(parameters: dict[str, Any], verifier: Any) -> dict[str, Any]:
    """Record stable runtime-resolved settings without inspecting credentials."""
    result = dict(parameters)
    missing = object()
    for key in ("model", "model_id", "endpoint", "gguf_variant", "device"):
        value = getattr(verifier, key, missing)
        if value is not missing and isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
    return result


def verdict_processor(
    *,
    args: Any,
    backend: str,
    parameters: dict[str, Any],
    verify: Callable[[Path], dict[str, Any]],
) -> Callable[[Path, Path], None]:
    """Build the common resumable verifier artifact writer."""

    def process(source: Path, destination: Path) -> None:
        wanted = request(identity(source), "verify", parameters, backend)
        if destination.exists() and not args.overwrite:
            old = read_json(destination)
            if all(old.get(key) == value for key, value in wanted.items()) and "verdict" in old:
                return
            raise ValueError(f"Conflicting output: {destination}; use --overwrite")

        verdict = verify(source)
        if not isinstance(verdict, dict) or verdict.get("decision") not in {"pass", "reject"}:
            raise ValueError("Verifier did not return a pass/reject decision")
        write_json(destination, {**wanted, "verdict": verdict})

    return process


def run_verifier(
    *,
    args: Any,
    pairs: list[tuple[Path, Path]],
    backend: str,
    parameters: dict[str, Any],
    verify: Callable[[Path], dict[str, Any]],
) -> int:
    return batch(
        pairs,
        verdict_processor(
            args=args,
            backend=backend,
            parameters=parameters,
            verify=verify,
        ),
        concurrency=getattr(args, 'concurrency', 1),
        batch_size=getattr(args, 'batch_size', 1),
    )
