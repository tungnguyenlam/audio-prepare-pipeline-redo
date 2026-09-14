"""Shared CLI, logging, and evaluation infrastructure for agent verifiers."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.files import ROOT, digest, progress, safe_name, write_json
from artifacts import write_text
from agent.verifier._verdicts import _known_prompts, _validate_verdict


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


def _generation_failure(exc: Exception) -> tuple[str, str, str]:
    """Classify an exception without persisting its potentially sensitive text."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__

    for item in chain:
        if isinstance(item, (TimeoutError,)):
            return "verifier", "generation_timeout", _ERROR_MESSAGES["generation_timeout"]
        if isinstance(item, FileNotFoundError):
            return "input", "input_not_found", _ERROR_MESSAGES["input_not_found"]
        if isinstance(item, PermissionError):
            return "input", "input_permission_denied", _ERROR_MESSAGES["input_permission_denied"]
        if isinstance(item, ImportError):
            return "environment", "dependency_missing", _ERROR_MESSAGES["dependency_missing"]

        text = f"{type(item).__name__}: {item}".lower()
        if "out of memory" in text or "cuda error: out of memory" in text or "oom" in text:
            return "verifier", "out_of_memory", _ERROR_MESSAGES["out_of_memory"]
        if "device" in text and ("unavailable" in text or "not found" in text):
            return "environment", "device_unavailable", _ERROR_MESSAGES["device_unavailable"]
        if "timed out" in text or "timeout" in text:
            return "verifier", "generation_timeout", _ERROR_MESSAGES["generation_timeout"]

    return "verifier", "generation_failed", _ERROR_MESSAGES["generation_failed"]


def _find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    )


def _relative_to(path: Path, start: Path) -> Path:
    try:
        return path.relative_to(start)
    except ValueError:
        return path


def _artifact_paths(audio_path: Path, input_dir: Path, output_dir: Path, model_tag: str) -> tuple[Path, Path]:
    relative = _relative_to(audio_path, input_dir)
    parent = output_dir / relative.parent
    stem = audio_path.stem
    return parent / f"{stem}_{model_tag}.json", parent / f"{stem}_{model_tag}.txt"


def _build_artifact(
    audio_path: Path,
    operation: str,
    model_tag: str,
    parameters: dict[str, Any],
    status: str,
    response_path: Path | None = None,
    response_text: str | None = None,
    verdict: dict[str, Any] | None = None,
    stage: str | None = None,
    code: str | None = None,
    message: str | None = None,
    invalid_verdict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "path": str(audio_path.resolve()),
            "sha256": digest(audio_path),
        },
        "operation": operation,
        "model": model_tag,
        "parameters": parameters,
        "status": status,
    }
    if invalid_verdict is not None:
        artifact["invalid_verdict"] = invalid_verdict
    if stage is not None and code is not None and message is not None:
        artifact["error"] = {
            "stage": stage,
            "code": code,
            "message": message,
        }
    if response_path is not None and response_text is not None:
        artifact["response"] = {
            "path": str(response_path.resolve()),
            "format": "utf-8 text",
            "kind": "text",
            "bytes": len(response_text.encode("utf-8")),
            "sha256": digest(response_path),
        }
    if verdict is not None:
        artifact["verdict"] = verdict
    return artifact


def run_verifier(
    audio_path: Path,
    input_dir: Path,
    output_dir: Path,
    model_tag: str,
    parameters: dict[str, Any],
    verifier_func: Callable[[Path], tuple[dict[str, Any] | None, str, dict[str, Any] | None]],
    force: bool = False,
    known_prompts: dict[str, str] | None = None,
) -> Path:
    json_path, txt_path = _artifact_paths(audio_path, input_dir, output_dir, model_tag)
    if json_path.is_file() and not force:
        return json_path

    known_prompts = _known_prompts() if known_prompts is None else known_prompts
    prompt = parameters.get("prompt")
    backend = model_tag

    try:
        verdict, raw_text, error_info = verifier_func(audio_path)
    except Exception as exc:
        stage, code, message = _generation_failure(exc)
        artifact = _build_artifact(
            audio_path=audio_path,
            operation="verify",
            model_tag=model_tag,
            parameters=parameters,
            status="fail",
            stage=stage,
            code=code,
            message=message,
        )
        write_json(json_path, artifact)
        return json_path

    write_text(txt_path, raw_text)

    if error_info is not None:
        stage = str(error_info.get("stage") or "verifier")
        code = str(error_info.get("code") or "generation_failed")
        message = _ERROR_MESSAGES.get(code, str(error_info.get("message") or "Model generation failed."))
        artifact = _build_artifact(
            audio_path=audio_path,
            operation="verify",
            model_tag=model_tag,
            parameters=parameters,
            status="fail",
            response_path=txt_path,
            response_text=raw_text,
            stage=stage,
            code=code,
            message=message,
            invalid_verdict=verdict if isinstance(verdict, dict) else None,
        )
        write_json(json_path, artifact)
        return json_path

    profile, schema_error = _validate_verdict(verdict, prompt, backend, known_prompts)
    if schema_error is not None:
        message = _ERROR_MESSAGES.get(schema_error, f"Schema validation failed: {schema_error}")
        artifact = _build_artifact(
            audio_path=audio_path,
            operation="verify",
            model_tag=model_tag,
            parameters=parameters,
            status="fail",
            response_path=txt_path,
            response_text=raw_text,
            stage="schema",
            code=schema_error,
            message=message,
            invalid_verdict=verdict if isinstance(verdict, dict) else None,
        )
        write_json(json_path, artifact)
        return json_path

    assert isinstance(verdict, dict)
    artifact = _build_artifact(
        audio_path=audio_path,
        operation="verify",
        model_tag=model_tag,
        parameters=parameters,
        status="success",
        response_path=txt_path,
        response_text=raw_text,
        verdict=verdict,
    )
    write_json(json_path, artifact)
    return json_path
