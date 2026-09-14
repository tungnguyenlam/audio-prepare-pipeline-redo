"""Shared CLI artifact handling for standalone verifier commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from _audio import VerifierResponseError
from _common.files import ROOT, batch, digest, identity, read_json, request, write_json
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

    default_file = ROOT / "prompts" / "acoustic_defect-3.txt"
    if not default_file.is_file():
        raise ValueError(f"Default prompt file not found: {default_file}")
    return default_file.read_text(encoding="utf-8").strip()


def resolved_parameters(parameters: dict[str, Any], verifier: Any) -> dict[str, Any]:
    """Record stable runtime-resolved settings without inspecting credentials."""
    result = dict(parameters)
    missing = object()
    for key in (
        "model",
        "model_id",
        "endpoint",
        "gguf_variant",
        "device",
        "model_class",
        "processor_class",
    ):
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
    known_prompts = _known_prompts()
    prompt = parameters.get("prompt")

    def publish_response(destination: Path, raw_response: str, kind: str) -> dict[str, Any]:
        response_path = destination.with_suffix(".txt")
        write_text(response_path, raw_response)
        return {
            "path": str(response_path),
            "format": "utf-8 text",
            "kind": kind,
            "bytes": response_path.stat().st_size,
            "sha256": digest(response_path),
        }

    def publish_failure(
        destination: Path,
        wanted: dict[str, Any],
        *,
        stage: str,
        code: str,
        raw_response: str | None = None,
        response_kind: str = "text",
        exception_type: str | None = None,
        invalid_verdict: dict[str, Any] | None = None,
    ) -> None:
        error = {
            "stage": stage,
            "code": code,
            "message": _error_message(code),
        }
        if exception_type is not None:
            error["exception_type"] = exception_type
        artifact: dict[str, Any] = {
            **wanted,
            "status": "fail",
            "error": error,
        }
        if invalid_verdict is not None:
            artifact["invalid_verdict"] = invalid_verdict
        if raw_response is not None:
            artifact["response"] = publish_response(
                destination, raw_response, response_kind
            )
        else:
            destination.with_suffix(".txt").unlink(missing_ok=True)
        write_json(destination, artifact)

    def process(source: Path, destination: Path) -> None:
        wanted = request(identity(source), "verify", parameters, backend)
        try:
            verdict = verify(source)
        except VerifierResponseError as exc:
            publish_failure(
                destination,
                wanted,
                stage="parse",
                code=exc.code,
                raw_response=exc.raw_response,
            )
            raise ValueError(f"{exc.code}: {_error_message(exc.code)}") from exc
        except Exception as exc:
            code, message, exception_type = _generation_failure(exc)
            publish_failure(
                destination,
                wanted,
                stage="generation",
                code=code,
                exception_type=exception_type,
            )
            raise ValueError(f"{code}: {message}") from exc

        raw_response = verdict.pop("_raw_response", None) if isinstance(verdict, dict) else None
        response_kind = verdict.pop("_response_kind", "text") if isinstance(verdict, dict) else "text"
        _, schema_error = _validate_verdict(verdict, prompt, backend, known_prompts)
        if schema_error is not None:
            publish_failure(
                destination,
                wanted,
                stage="schema",
                code=schema_error,
                raw_response=raw_response if isinstance(raw_response, str) else None,
                response_kind=str(response_kind),
                invalid_verdict=verdict if isinstance(verdict, dict) else None,
            )
            raise ValueError(f"{schema_error}: {_error_message(schema_error)}")
        response = (
            publish_response(destination, raw_response, str(response_kind))
            if isinstance(raw_response, str)
            else {"available": False}
        )
        write_json(
            destination,
            {**wanted, "status": "success", "response": response, "verdict": verdict},
        )

    return process


def run_verifier(
    *,
    args: Any,
    pairs: list[tuple[Path, Path]],
    backend: str,
    parameters: dict[str, Any],
    verify: Callable[[Path], dict[str, Any]],
) -> int:
    pairs = pending_verifier_pairs(
        args=args,
        pairs=pairs,
        backend=backend,
        parameters=parameters,
    )
    return batch(
        pairs,
        verdict_processor(
            args=args,
            backend=backend,
            parameters=parameters,
            verify=verify,
        ),
        concurrency=getattr(args, "concurrency", 1),
        batch_size=getattr(args, "batch_size", 1),
    )


def _response_complete(destination: Path, artifact: dict[str, Any]) -> bool:
    response = artifact.get("response")
    if not isinstance(response, dict):
        return False
    if response.get("available") is False:
        return True
    path_value = response.get("path")
    expected_sha = response.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha, str):
        return False
    response_path = Path(path_value)
    if not response_path.is_absolute():
        response_path = (destination.parent / response_path).resolve()
    return response_path.is_file() and digest(response_path) == expected_sha


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
                response_complete = old.get("status") is None or (
                    old.get("status") == "success" and _response_complete(destination, old)
                )
                if schema_error is None and response_complete:
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
