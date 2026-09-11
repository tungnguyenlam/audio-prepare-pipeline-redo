"""Shared CLI artifact handling for standalone verifier commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from _audio import DEFAULT_ACOUSTIC_PROMPT, VerifierResponseError
from _common.files import ROOT, batch, digest, identity, read_json, request, write_json
from artifacts import write_text


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
    ) -> None:
        artifact: dict[str, Any] = {
            **wanted,
            "status": "fail",
            "error": {"stage": stage, "code": code},
        }
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
            raise ValueError(exc.code) from exc
        except Exception as exc:
            publish_failure(
                destination,
                wanted,
                stage="generation",
                code="generation_failed",
            )
            raise ValueError("generation_failed") from exc

        raw_response = verdict.pop("_raw_response", None) if isinstance(verdict, dict) else None
        response_kind = verdict.pop("_response_kind", "text") if isinstance(verdict, dict) else "text"
        if not isinstance(verdict, dict) or verdict.get("decision") not in {"pass", "reject"}:
            publish_failure(
                destination,
                wanted,
                stage="schema",
                code="invalid_decision",
                raw_response=raw_response if isinstance(raw_response, str) else None,
                response_kind=str(response_kind),
            )
            raise ValueError("Verifier did not return a pass/reject decision")
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
        concurrency=getattr(args, 'concurrency', 1),
        batch_size=getattr(args, 'batch_size', 1),
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
                if old.get("status") is None or (
                    old.get("status") == "success"
                    and _response_complete(destination, old)
                ):
                    continue
            if matches and old.get("status") == "fail":
                raise ValueError(
                    f"Cached failed verifier artifact: {destination}; use --overwrite to retry"
                )
            raise ValueError(f"Conflicting output: {destination}; use --overwrite")
        pending.append((source, destination))
    return pending
