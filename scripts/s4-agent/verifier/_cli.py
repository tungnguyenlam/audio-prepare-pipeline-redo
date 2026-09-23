"""Shared CLI artifact handling for standalone verifier commands."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from _audio import VerifierResponseError
from _common.files import ROOT, batch, digest, identity, persist_path, progress, read_json, request, resolve_output_dir, resolve_stored_path, write_json
from artifacts import write_text
from _verifier_artifacts import response_complete
from _verdicts import _known_prompts, _validate_verdict
from _reporting import item_detail, new_run_stats, record_result, report_cost_summary


_ERROR_MESSAGES = {
    "invalid_json": "The model response did not contain a valid JSON object.",
    "json_not_object": "The model response was valid JSON but was not an object.",
    "verdict_not_object": "The parsed verifier verdict was not an object.",
    "invalid_decision": "The verdict decision must be either 'pass' or 'reject'.",
    "invalid_speaker_purity": "The verdict contains an unsupported speaker_purity value.",
    "invalid_word_completeness": "The verdict contains an unsupported word_completeness value.",
    "invalid_audio_quality": "The verdict contains an unsupported audio_quality value.",
    "invalid_failure_codes": "The verdict contains invalid failure_codes.",
    "inconsistent_failure_codes": "The failure_codes do not match the reported verifier dimensions.",
    "inconsistent_decision": "The decision conflicts with the reported failures.",
    "missing_transcript": "The model returned 'pass' without a non-empty transcript.",
    "unexpected_transcript": "The model returned a transcript for a rejected clip.",
    "transcript_not_last": "The transcript field must be the final public verdict field.",
    "missing_emotion": "The model returned 'pass' without an emotion description.",
    "invalid_emotion": "The verdict contains an invalid emotion value.",
    "invalid_word_boundary": "The verdict contains an unsupported word-boundary value.",
    "input_not_found": "The input audio file was not found during generation.",
    "input_permission_denied": "The input audio file could not be read due to its permissions.",
    "generation_timeout": "Model generation timed out.",
    "out_of_memory": "Model generation ran out of accelerator or system memory.",
    "device_unavailable": "The requested inference device is unavailable.",
    "dependency_missing": "A required inference dependency is missing.",
    "generation_failed": "Model generation raised an unexpected exception.",
}


def _error_message(code: str) -> str:
    return _ERROR_MESSAGES.get(code, f"Verifier failed with error code '{code}'.")


def _generation_failure(exc: Exception) -> tuple[str, str, str]:
    """Classify an exception without persisting its potentially sensitive text."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__

    code = "generation_failed"
    exception_type = type(exc).__name__
    for item in chain:
        exception_type = type(item).__name__
        if isinstance(item, TimeoutError):
            code = "generation_timeout"
            break
        if isinstance(item, FileNotFoundError):
            code = "input_not_found"
            break
        if isinstance(item, PermissionError):
            code = "input_permission_denied"
            break
        if isinstance(item, ImportError):
            code = "dependency_missing"
            break
        if isinstance(item, MemoryError):
            code = "out_of_memory"
            break

        detail = f"{type(item).__name__}: {item}".lower()
        if "out of memory" in detail or "cuda error: out of memory" in detail:
            code = "out_of_memory"
            break
        if "device" in detail and ("unavailable" in detail or "not found" in detail):
            code = "device_unavailable"
            break
        if "timed out" in detail or "timeout" in detail:
            code = "generation_timeout"
            break

    return code, _error_message(code), exception_type


def load_prompt(prompt_file: Path | None) -> str:
    """Load an explicit prompt or the repository's default acoustic prompt."""
    if prompt_file is not None:
        if not prompt_file.is_file():
            raise ValueError(f"Prompt file not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8").strip()

    default_file = ROOT / "prompts" / "full-tags-prompt.md"
    if not default_file.is_file():
        raise ValueError(f"Default prompt file not found: {default_file}")
    return default_file.read_text(encoding="utf-8").strip()


def resolved_parameters(parameters: dict[str, Any], verifier: Any | None = None) -> dict[str, Any]:
    resolved = {key: value for key, value in parameters.items() if value is not None}
    if verifier is not None:
        model = getattr(verifier, "model", None)
        if model:
            resolved.setdefault("model", model)
        effort = getattr(verifier, "reasoning_effort", None)
        if effort:
            resolved.setdefault("reasoning_effort", effort)
        tokens = getattr(verifier, "max_tokens", None)
        if tokens:
            resolved.setdefault("max_tokens", tokens)
    return resolved


def _write_verdict_artifacts(
    *,
    source: Path,
    destination: Path,
    backend: str,
    parameters: dict[str, Any],
    verdict: dict[str, Any] | None,
    raw_response: str | None,
    error: tuple[str, str, str] | None = None,
    invalid_verdict: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
) -> None:
    text_path = destination.with_suffix(".txt")
    if raw_response is not None:
        write_text(text_path, raw_response)
        response_meta = {
            "path": persist_path(text_path),
            "format": "utf-8 text",
            "kind": "text",
            "bytes": len(raw_response.encode("utf-8")),
            "sha256": digest(text_path),
        }
    else:
        response_meta = {"available": False}

    artifact: dict[str, Any] = request(identity(source), "verify", parameters, backend)
    artifact["status"] = "fail" if error is not None else "success"
    artifact["response"] = response_meta
    if generation is not None:
        artifact["generation"] = {key: value for key, value in generation.items() if key != "text"}
    if error is not None:
        code, message, exception_type = error
        artifact["error"] = {
            "stage": "generation" if generation and generation.get("generation_error") else "verifier",
            "code": code,
            "message": message,
            "exception": exception_type,
        }
        if invalid_verdict is not None:
            artifact["invalid_verdict"] = invalid_verdict
    else:
        artifact["verdict"] = verdict
    write_json(destination, artifact)


def verdict_processor(
    *,
    args: Any,
    backend: str,
    parameters: dict[str, Any],
    verify: Callable[[Path], dict[str, Any]],
    stats: dict[str, Any],
) -> Callable[[Path, Path], None]:
    known_prompts = _known_prompts()
    prompt = parameters.get("prompt")

    def process(source: Path, destination: Path) -> None:
        verdict: dict[str, Any] | None = None
        raw_response: str | None = None
        error: tuple[str, str, str] | None = None
        invalid_verdict: dict[str, Any] | None = None
        generation: dict[str, Any] | None = None

        try:
            verdict = verify(source)
            if isinstance(verdict, dict):
                generation = verdict.pop("_generation", None)
            raw_response = (
                str(verdict.pop("_raw_response"))
                if isinstance(verdict, dict) and "_raw_response" in verdict
                else None
            )
            profile, schema_error = _validate_verdict(verdict, prompt, backend, known_prompts)
            del profile
            if schema_error is not None:
                error = (schema_error, _error_message(schema_error), "VerifierResponseError")
                invalid_verdict = verdict
                verdict = None
        except VerifierResponseError as exc:
            generation = getattr(exc, "generation", None)
            raw_response = exc.raw_response
            error = (exc.code, _error_message(exc.code), type(exc).__name__)
        except Exception as exc:
            generation = getattr(exc, "generation", None)
            raw_error = getattr(exc, "raw_response", None)
            if isinstance(raw_error, str):
                raw_response = raw_error
            error = _generation_failure(exc)
            if isinstance(generation, dict) and generation.get("generation_error"):
                failure = generation["generation_error"]
                error = (failure["code"], failure["message"], type(exc).__name__)

        _write_verdict_artifacts(
            source=source,
            destination=destination,
            backend=backend,
            parameters=parameters,
            verdict=verdict,
            raw_response=raw_response,
            error=error,
            invalid_verdict=invalid_verdict,
            generation=generation,
        )

        decision = verdict.get('decision') if isinstance(verdict, dict) else None
        running_total = record_result(
            stats,
            verdict if verdict is not None else invalid_verdict or generation,
            success=error is None,
            decision=decision,
        )
        progress(
            'VERIFIER_ITEM',
            f'{source.name}: decision={decision or "unknown"}; '
            f'{"error=" + error[0] + "; " if error else ""}'
            f'{item_detail(verdict or generation, running_total_usd=running_total)}',
        )

    return process


def _resolve_verdict_dir(args: Any, pairs: list[tuple[Path, Path]]) -> Path | None:
    if getattr(args, "output_file", None) is not None:
        return args.output_file.resolve().parent
    if getattr(args, "output_dir", None) is not None:
        return args.output_dir.resolve()
    if getattr(args, "input_dir", None) is not None:
        return resolve_output_dir(args, args.input_dir)
    if pairs:
        parents = [dest.resolve().parent for _, dest in pairs]
        try:
            return Path(os.path.commonpath([str(p) for p in parents]))
        except ValueError:
            return parents[0]
    return None


def _run_post_verification_analysis(args: Any, verdict_dir: Path) -> None:
    launcher = ROOT / "scripts" / "s4-agent" / "verifier" / "plot_verifier_analysis.sh"
    if not launcher.is_file():
        return
    has_json = any(
        p.is_file() and p.name not in ("report.json", "analysis.json")
        for p in verdict_dir.rglob("*.json")
        if not any(part in {"work", "plot", "plots", "comparisons", "experiments"} for part in p.parts)
    )
    if not has_json:
        return
    cmd = [
        "bash",
        str(launcher),
        "--input-dir",
        str(verdict_dir),
        "--concurrency",
        str(getattr(args, "concurrency", 1)),
        "--batch-size",
        str(getattr(args, "batch_size", 1)),
    ]
    if getattr(args, "overwrite", False):
        cmd.append("--overwrite")
    try:
        progress("VERIFIER_ANALYZE", f"Running post-verification analysis on {verdict_dir}")
        subprocess.run(cmd, check=True)
    except Exception as exc:
        progress("VERIFIER_ANALYZE_FAIL", f"Post-verification analysis failed: {exc}")


def run_verifier(
    *,
    args: Any,
    pairs: list[tuple[Path, Path]],
    backend: str,
    parameters: dict[str, Any],
    verify: Callable[[Path], dict[str, Any]],
    cost_summary: Callable[[], Mapping[str, Any]] | None = None,
) -> int:
    stats = new_run_stats()
    result = 1
    initial_pairs = list(pairs)
    try:
        pairs = pending_verifier_pairs(
            args=args,
            pairs=pairs,
            backend=backend,
            parameters=parameters,
        )
        stats['attempted'] = len(pairs)
        model = parameters.get('model') or parameters.get('model_id') or backend
        progress(
            'VERIFIER_START',
            f'backend={backend}; model={model}; items={len(pairs)}',
        )
        result = batch(
            pairs,
            verdict_processor(
                args=args,
                backend=backend,
                parameters=parameters,
                verify=verify,
                stats=stats,
            ),
            concurrency=getattr(args, "concurrency", 1),
            batch_size=getattr(args, "batch_size", 1),
        )
        return result
    finally:
        provider = cost_summary() if cost_summary is not None else None
        report_cost_summary(f'verifier/{backend}', stats, provider)
        if not getattr(args, "skip_analysis", False):
            verdict_dir = _resolve_verdict_dir(args, initial_pairs)
            if verdict_dir is not None and verdict_dir.is_dir():
                _run_post_verification_analysis(args, verdict_dir)


def pending_verifier_pairs(
    *,
    args: Any,
    pairs: list[tuple[Path, Path]],
    backend: str,
    parameters: dict[str, Any],
) -> list[tuple[Path, Path]]:
    """Preflight verdict outputs so a paid batch only contains missing artifacts."""
    known_prompts = _known_prompts()
    prompt = parameters.get("prompt")
    pending = []
    for source, destination in pairs:
        wanted = request(identity(source), "verify", parameters, backend)
        response_path = destination.with_suffix(".txt")
        if (destination.exists() or response_path.exists()) and not args.overwrite:
            if getattr(args, "continue_run", False) and not destination.exists():
                # Publication may have stopped between the text and JSON writes.
                # Batch can recover its response; synchronous modes must retry.
                pending.append((source, destination))
                continue
            if not destination.exists():
                raise ValueError(
                    f"Incomplete verifier output pair: {response_path}; use --overwrite"
                )
            old = read_json(destination)
            matches = all(old.get(key) == value for key, value in wanted.items())
            if matches and "verdict" in old:
                _, schema_error = _validate_verdict(
                    old.get("verdict"), prompt, backend, known_prompts
                )
                response_is_complete = (
                    old.get("status") in (None, "success") and response_complete(destination, old)
                )
                if schema_error is None and response_is_complete:
                    continue
            if matches and getattr(args, "continue_run", False):
                # Retry only matching failed/incomplete artifacts. Conflicting
                # source, prompt, or generation settings still require overwrite.
                pending.append((source, destination))
                continue
            if matches and old.get("status") == "fail":
                error = old.get("error") if isinstance(old.get("error"), dict) else {}
                code = str(error.get("code") or "generation_failed")
                message = str(error.get("message") or _error_message(code))
                raise ValueError(
                    f"Cached failed verifier artifact ({code}: {message}): "
                    f"{destination}; use --overwrite to retry"
                )
            raise ValueError(f"Conflicting output: {destination}; use --overwrite")
        pending.append((source, destination))
    return pending
