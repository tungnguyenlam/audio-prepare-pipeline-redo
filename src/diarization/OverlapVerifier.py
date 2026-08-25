"""Direct-audio overlap verification with local Gemma or Gemini."""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.utils.AudioClass import Audio

OVERLAP_PROMPT = (
    "Does this audio contain overlapping speech from two or more speakers "
    "at the same time?"
)
DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS = 128
DEFAULT_UNSLOTH_HOST = "localhost"
DEFAULT_UNSLOTH_PORT = 8888
DEFAULT_UNSLOTH_ENDPOINT = (
    f"http://{DEFAULT_UNSLOTH_HOST}:{DEFAULT_UNSLOTH_PORT}/v1/chat/completions"
)
DEFAULT_GEMMA4_MODEL_ID = "unsloth/gemma-4-12b-it-GGUF"
DEFAULT_GEMINI_MODEL_ID = "gemini-3.1-pro-preview"

_OVERLAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overlap": {"type": "boolean"},
        "reason": {
            "type": "string",
            "description": "A short explanation for the overlap decision.",
        },
    },
    "required": ["overlap", "reason"],
    "additionalProperties": False,
}
_AUDIO_MIME_TYPES = {
    ".aac": "audio/aac",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".flac": "audio/flac",
    ".mp3": "audio/mp3",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".wave": "audio/wav",
}


class OverlapVerificationResult(TypedDict):
    """Normalized answer returned by every overlap verifier."""

    overlap: bool
    reason: str


class OverlapVerifierError(RuntimeError):
    """Raised when overlap verification cannot produce a valid decision."""


class BaseOverlapVerifier(ABC):
    """Backend-independent interface for direct-audio overlap verification."""

    @abstractmethod
    def verify(self, audio: Audio) -> OverlapVerificationResult:
        """Return whether an audio segment contains simultaneous speakers."""


class Gemma4OverlapVerifier(BaseOverlapVerifier):
    """Verify overlap with Gemma 4 12B served by Unsloth Studio."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        prompt: str = OVERLAP_PROMPT,
        max_output_tokens: int = DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS,
    ) -> None:
        """Initialize the Unsloth-backed verifier.

        Args:
            endpoint: Full OpenAI-compatible chat-completions URL. Defaults to
                ``UNSLOTH_ENDPOINT`` or a URL using ``UNSLOTH_HOST`` (default:
                localhost) and ``UNSLOTH_PORT`` (default: 8888).
            model: Loaded Unsloth model ID. Defaults to ``UNSLOTH_MODEL`` or
                the Gemma 4 12B repository ID.
            api_key: Unsloth API key. Defaults to ``UNSLOTH_API_KEY``. It is
                optional for local servers configured without authentication.
            timeout_s: HTTP request timeout in seconds.
            prompt: Instruction sent with every candidate audio segment.
            max_output_tokens: Maximum tokens allowed for the JSON decision.
        """
        self.endpoint = (
            endpoint
            or os.getenv("UNSLOTH_ENDPOINT")
            or _default_unsloth_endpoint()
        )
        self.model = model or os.getenv("UNSLOTH_MODEL") or DEFAULT_GEMMA4_MODEL_ID
        self.api_key = api_key if api_key is not None else os.getenv("UNSLOTH_API_KEY")
        self.timeout_s = _validate_timeout(timeout_s)
        self.prompt = _validate_prompt(prompt)
        self.max_output_tokens = _validate_max_output_tokens(max_output_tokens)

    def verify(self, audio: Audio) -> OverlapVerificationResult:
        """Send the audio segment directly to the local Gemma model."""
        audio_bytes, suffix, _ = _read_audio(audio)
        if suffix not in {".mp3", ".wav", ".wave"}:
            raise OverlapVerifierError(
                "Unsloth input_audio requires a WAV or MP3 segment, got "
                f"{suffix or 'a file without an extension'}"
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                                "format": "mp3" if suffix == ".mp3" else "wav",
                            },
                        },
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "overlap_verification",
                    "strict": True,
                    "schema": _OVERLAP_SCHEMA,
                },
            },
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = _post_json(
            self.endpoint,
            payload,
            headers=headers,
            timeout_s=self.timeout_s,
            backend="Unsloth",
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OverlapVerifierError(
                "Unsloth returned no assistant message content"
            ) from exc
        return _normalize_result(content, backend="Unsloth")


class GeminiOverlapVerifier(BaseOverlapVerifier):
    """Verify overlap with Gemini 3.1 Pro through the Gemini API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        prompt: str = OVERLAP_PROMPT,
        max_output_tokens: int = DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS,
    ) -> None:
        """Initialize the Gemini-backed verifier.

        Args:
            model: Gemini model ID. Defaults to ``GEMINI_MODEL`` or Gemini
                3.1 Pro Preview.
            api_key: Gemini API key. Defaults to ``GEMINI_API_KEY``.
            timeout_s: HTTP request timeout in seconds.
            prompt: Instruction sent with every candidate audio segment.
            max_output_tokens: Maximum tokens allowed for the JSON decision.
        """
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL_ID
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.timeout_s = _validate_timeout(timeout_s)
        self.prompt = _validate_prompt(prompt)
        self.max_output_tokens = _validate_max_output_tokens(max_output_tokens)

    def verify(self, audio: Audio) -> OverlapVerificationResult:
        """Send the audio segment directly to Gemini 3.1 Pro."""
        if not self.api_key:
            raise OverlapVerifierError(
                "Gemini API key is not configured; set GEMINI_API_KEY"
            )

        audio_bytes, _, mime_type = _read_audio(audio)
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(self.model, safe='')}:generateContent"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": self.prompt},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "responseFormat": {
                    "text": {
                        "mimeType": "application/json",
                        "schema": _OVERLAP_SCHEMA,
                    }
                },
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        response = _post_json(
            endpoint,
            payload,
            headers={"x-goog-api-key": self.api_key},
            timeout_s=self.timeout_s,
            backend="Gemini",
        )
        try:
            parts = response["candidates"][0]["content"]["parts"]
            content = "".join(
                str(part["text"])
                for part in parts
                if isinstance(part, dict)
                and "text" in part
                and not part.get("thought", False)
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise OverlapVerifierError(
                "Gemini returned no candidate message content"
            ) from exc
        return _normalize_result(content, backend="Gemini")


def create_overlap_verifier(
    config: Mapping[str, Any] | None = None,
) -> BaseOverlapVerifier:
    """Create the selected verifier from a small flat configuration mapping.

    Args:
        config: Settings containing ``backend`` (``"gemma4"`` or
            ``"gemini"``) and optional ``endpoint``, ``model``, ``api_key``,
            ``timeout_s``, ``prompt``, and ``max_output_tokens`` values.
            ``backend`` falls back to the ``OVERLAP_VERIFIER`` environment
            variable.

    Returns:
        The selected overlap verifier.

    Raises:
        ValueError: If no supported backend is selected.
    """
    settings = dict(config or {})
    backend = str(settings.pop("backend", os.getenv("OVERLAP_VERIFIER", "")))
    backend = backend.strip().lower().replace("_", "-")

    if backend in {"gemma", "gemma4", "gemma-4", "unsloth"}:
        return Gemma4OverlapVerifier(**settings)
    if backend in {"gemini", "gemini-3.1", "gemini-3.1-pro"}:
        if "endpoint" in settings:
            raise ValueError("Gemini overlap verifier does not accept endpoint")
        return GeminiOverlapVerifier(**settings)
    raise ValueError(
        "Select an overlap verifier with backend='gemma4' or backend='gemini'"
    )


def _read_audio(audio: Audio) -> tuple[bytes, str, str]:
    """Read one file-backed Audio and return bytes, suffix, and MIME type."""
    path = Path(audio.path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    suffix = path.suffix.lower()
    mime_type = _AUDIO_MIME_TYPES.get(suffix)
    if mime_type is None:
        raise OverlapVerifierError(f"Unsupported audio format: {suffix or path.name}")
    try:
        return path.read_bytes(), suffix, mime_type
    except OSError as exc:
        raise OverlapVerifierError(f"Could not read audio segment: {path}") from exc


def _post_json(
    endpoint: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
    timeout_s: float,
    backend: str,
) -> dict[str, Any]:
    """POST JSON and return a decoded object with backend-specific errors."""
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OverlapVerifierError(
            f"{backend} request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OverlapVerifierError(f"{backend} request failed: {exc}") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OverlapVerifierError(f"{backend} returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise OverlapVerifierError(f"{backend} returned a non-object response")
    return decoded


def _normalize_result(content: Any, *, backend: str) -> OverlapVerificationResult:
    """Validate a structured model answer and normalize its two fields."""
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.removeprefix("```json").removeprefix("```")
            stripped = stripped.removesuffix("```").strip()
        try:
            content = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise OverlapVerifierError(
                f"{backend} returned an invalid overlap result: {content!r}"
            ) from exc

    if not isinstance(content, dict):
        raise OverlapVerifierError(f"{backend} returned a non-object overlap result")
    overlap = content.get("overlap")
    reason = content.get("reason")
    if not isinstance(overlap, bool):
        raise OverlapVerifierError(f"{backend} result field 'overlap' is not a bool")
    if not isinstance(reason, str) or not reason.strip():
        raise OverlapVerifierError(
            f"{backend} result field 'reason' is not a non-empty string"
        )
    return {"overlap": overlap, "reason": reason.strip()}


def _validate_timeout(timeout_s: float) -> float:
    """Return a positive HTTP timeout."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout_s must be a number")
    timeout = float(timeout_s)
    if timeout <= 0:
        raise ValueError("timeout_s must be greater than zero")
    return timeout


def _validate_prompt(prompt: str) -> str:
    """Return a non-empty verifier instruction."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("prompt must not be empty")
    return normalized


def _validate_max_output_tokens(max_output_tokens: int) -> int:
    """Return a positive structured-response token budget."""
    if isinstance(max_output_tokens, bool) or not isinstance(
        max_output_tokens, int
    ):
        raise TypeError("max_output_tokens must be an integer")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than zero")
    return max_output_tokens


def _default_unsloth_endpoint() -> str:
    """Build the Unsloth endpoint from its configured host and port."""
    host = os.getenv("UNSLOTH_HOST", DEFAULT_UNSLOTH_HOST).strip()
    if not host:
        raise ValueError("UNSLOTH_HOST must not be empty")
    configured_port = os.getenv("UNSLOTH_PORT", str(DEFAULT_UNSLOTH_PORT)).strip()
    try:
        port = int(configured_port)
    except ValueError as exc:
        raise ValueError("UNSLOTH_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("UNSLOTH_PORT must be between 1 and 65535")
    return f"http://{host}:{port}/v1/chat/completions"
