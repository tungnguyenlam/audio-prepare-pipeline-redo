"""Direct-audio speaker-purity and word-boundary verification."""

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

from tqdm.auto import tqdm

from src.utils.AudioClass import Audio

OVERLAP_PROMPT = """Listen to the supplied audio directly. Do not transcribe it.
Evaluate two strict acoustic criteria required for clean speech-synthesis training:

1. SPEAKER PURITY & TAIL INTRUSION:
   - Exactly one human speaker must be audible across the entire audio clip.
   - Reject simultaneous overlap, secondary background voices, or whispers.
   - CRITICAL TAIL INTRUSION CHECK: Scrutinize the final 500 milliseconds of the audio segment with extreme sensitivity. Reject if a secondary speaker begins speaking, whispers, murmurs, laughs, or utters a trailing backchannel (e.g. 'dạ', 'vâng', 'ừ', 'uh-huh', 'yeah') right as or immediately after the main speaker finishes. Even a momentary foreign vocal sound at the end must be rejected (use failure code 'tail_speaker_intrusion' or 'secondary_speaker').
   - Non-speech noise, ambient room reverb, or background music alone does not count as a second speaker.

2. ACOUSTIC WORD COMPLETENESS (NO CLIPPED BOUNDARIES / KHÔNG LẸM CHỮ):
   - Speech must start and end on complete, natural acoustic word boundaries:
     * START: The initial word must have its full phonetic attack/consonant onset. Reject if the audio cuts in abruptly mid-vowel or mid-syllable without its natural phonetic onset ('clipped_word_start').
     * END: The final word must complete its full tonal contour and coda closure (-p, -t, -k, -m, -n, -ng) into natural silence. Reject if the audio cuts off abruptly while vocal fold vibration, tonal contour, or consonant release is still actively in flight ('clipped_word_end').
   - Grammatical fragments are completely acceptable provided every audible word is acoustically complete. Do not infer missing words from grammatical context.

Respond ONLY with a valid JSON object matching this exact schema:
{
  "speaker_purity": "pure" | "impure" | "uncertain",
  "word_completeness": "complete" | "incomplete" | "uncertain",
  "boundary_issue": "none" | "clipped_start" | "clipped_end" | "clipped_both" | "uncertain",
  "failure_codes": [
    "overlapping_speech" | "secondary_speaker" | "tail_speaker_intrusion" | "clipped_word_start" | "clipped_word_end" | "unintelligible_boundary" | "insufficient_evidence"
  ],
  "reason": "<concise acoustic explanation; never transcribe the speech>"
}

Validation rules:
- If speaker_purity is "impure", failure_codes MUST include at least one of: "overlapping_speech", "secondary_speaker", "tail_speaker_intrusion". If "pure", do NOT include these codes.
- If word_completeness is "incomplete", boundary_issue must NOT be "none", and failure_codes MUST include at least one of: "clipped_word_start", "clipped_word_end", "unintelligible_boundary".
- If word_completeness is "complete", boundary_issue MUST be "none" and no clipped codes may be included.
- Return raw JSON only. Do not add markdown backticks (```json), commentary, or text outside the JSON object."""
DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS = 1024
DEFAULT_UNSLOTH_HOST = "localhost"
DEFAULT_UNSLOTH_PORT = 8888
DEFAULT_UNSLOTH_ENDPOINT = (
    f"http://{DEFAULT_UNSLOTH_HOST}:{DEFAULT_UNSLOTH_PORT}/v1/chat/completions"
)
DEFAULT_GEMMA4_MODEL_ID = "unsloth/gemma-4-12b-it-GGUF"
DEFAULT_GEMINI_MODEL_ID = "gemini-3.8-flash"
DEFAULT_GEMINI_FLASH_LITE_MODEL_ID = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_CONCURRENCY = 10
DEFAULT_GEMMA4_CONCURRENCY = 1
UNSLOTH_PROBE_TIMEOUT_S = 5.0

GEMINI_AUDIO_MODELS: tuple[dict[str, Any], ...] = (
    {"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash", "recommended": True},
    {"id": "gemini-3.7-flash", "label": "Gemini 3.7 Flash"},
    {"id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash"},
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
    {"id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash-Lite"},
    {"id": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro Preview"},
    {"id": "gemini-3.1-flash-lite", "label": "Gemini 3.1 Flash-Lite"},
)

# Paid Standard list prices in USD per million tokens. These are deliberately
# versioned in result records because Google prices and introductory offers can
# change independently of this repository.
GEMINI_PRICE_CARD_AS_OF = "2026-09-04"
_GEMINI_STANDARD_PRICES: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3.1-flash-lite": {
        "input": 0.25,
        "audio_input": 0.50,
        "output": 1.50,
    },
}

_FAILURE_CODES = {
    "overlapping_speech",
    "secondary_speaker",
    "tail_speaker_intrusion",
    "clipped_word_start",
    "clipped_word_end",
    "unintelligible_boundary",
    "insufficient_evidence",
}

_OVERLAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "speaker_purity": {
            "type": "string",
            "enum": ["pure", "impure", "uncertain"],
        },
        "word_completeness": {
            "type": "string",
            "enum": ["complete", "incomplete", "uncertain"],
        },
        "boundary_issue": {
            "type": "string",
            "enum": [
                "none",
                "clipped_start",
                "clipped_end",
                "clipped_both",
                "uncertain",
            ],
        },
        "failure_codes": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(_FAILURE_CODES)},
        },
        "reason": {
            "type": "string",
            "description": "A short acoustic explanation; never a transcript.",
        },
    },
    "required": [
        "speaker_purity",
        "word_completeness",
        "boundary_issue",
        "failure_codes",
        "reason",
    ],
    "additionalProperties": False,
}
_GEMINI_OVERLAP_SCHEMA: dict[str, Any] = {
    key: value for key, value in _OVERLAP_SCHEMA.items() if key != "additionalProperties"
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
    """Normalized quality decision returned by every direct-audio verifier."""

    overlap: bool
    speaker_purity: str
    word_completeness: str
    boundary_issue: str
    failure_codes: list[str]
    decision: str
    reason: str
    usage: dict[str, Any] | None
    cost: dict[str, Any] | None


class OverlapVerifierError(RuntimeError):
    """Raised when overlap verification cannot produce a valid decision.

    ``readiness`` is true when the backend itself is unreachable, still
    loading, or has no model — as opposed to a single malformed answer.
    """

    def __init__(self, message: str, *, readiness: bool = False) -> None:
        super().__init__(message)
        self.readiness = bool(readiness)


class BaseOverlapVerifier(ABC):
    """Backend-independent interface for direct-audio overlap verification."""

    concurrency: int = 1

    def check_ready(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Return whether this backend can accept candidate audio."""
        del timeout_s
        return {"ready": True, "message": "Ready.", "models": []}

    @abstractmethod
    def verify(self, audio: Audio) -> OverlapVerificationResult:
        """Return speaker-purity and word-boundary quality for one segment."""

    def verify_batch(
        self,
        audios: list[Audio],
        *,
        max_workers: int | None = None,
    ) -> list[OverlapVerificationResult]:
        """Verify multiple candidate segments concurrently.

        Args:
            audios: Candidate segments to verify.
            max_workers: Concurrency limit. Defaults to ``self.concurrency``
                (e.g., 10 for Gemini, 1 for Gemma).

        Returns:
            List of verification results in the exact order of ``audios``.
        """
        if not audios:
            return []
        effective_workers = (
            max_workers
            if max_workers is not None
            else getattr(self, "concurrency", 1)
        )
        effective_workers = _validate_concurrency(effective_workers)
        if effective_workers <= 1 or len(audios) <= 1:
            return [
                self.verify(audio)
                for audio in tqdm(audios, desc="Verifying segments", unit="segment")
            ]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(effective_workers, len(audios))
        ) as executor:
            return list(
                tqdm(
                    executor.map(self.verify, audios),
                    total=len(audios),
                    desc="Verifying segments",
                    unit="segment",
                )
            )


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
        concurrency: int | None = None,
    ) -> None:
        """Initialize the Unsloth-backed Gemma 4 verifier.

        Args:
            endpoint: Multimodal chat completions endpoint URL.
            model: Model identifier registered on the endpoint. Defaults to
                the Gemma 4 12B repository ID.
            api_key: Unsloth API key. Defaults to ``UNSLOTH_API_KEY``. It is
                optional for local servers configured without authentication.
            timeout_s: HTTP request timeout in seconds.
            prompt: Instruction sent with every candidate audio segment.
            max_output_tokens: Maximum tokens allowed for the JSON decision.
            concurrency: Number of parallel candidate queries allowed when
                calling ``verify_batch()``. Defaults to ``GEMMA4_CONCURRENCY``,
                ``UNSLOTH_CONCURRENCY``, or 1.
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
        if concurrency is None:
            raw_concurrency = (
                os.getenv("GEMMA4_CONCURRENCY")
                or os.getenv("UNSLOTH_CONCURRENCY")
            )
            if raw_concurrency is not None:
                try:
                    concurrency = int(raw_concurrency)
                except ValueError:
                    pass
        if concurrency is None:
            concurrency = DEFAULT_GEMMA4_CONCURRENCY
        self.concurrency = _validate_concurrency(concurrency)

    def check_ready(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Probe Unsloth before sending candidate audio.

        Returns:
            ``ready``, ``message``, and any model IDs advertised at
            ``/v1/models``. Never raises; unreadiness is the result.
        """
        probe_timeout = (
            UNSLOTH_PROBE_TIMEOUT_S if timeout_s is None else _validate_timeout(timeout_s)
        )
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        models_url = _unsloth_models_url(self.endpoint)
        try:
            payload = _get_json(
                models_url,
                headers=headers,
                timeout_s=probe_timeout,
                backend="Unsloth",
            )
        except OverlapVerifierError as exc:
            return {
                "ready": False,
                "message": _unsloth_unready_message(self.endpoint, str(exc)),
                "models": [],
            }

        models = _openai_model_ids(payload)
        if not models:
            return {
                "ready": False,
                "message": (
                    "Unsloth is reachable but has no model loaded at "
                    f"{models_url}. Wait until the Gemma checkpoint finishes "
                    "loading, then retry."
                ),
                "models": [],
            }
        if self.model and self.model not in models:
            return {
                "ready": True,
                "message": (
                    f"Unsloth is up with {', '.join(models)}. Configured "
                    f"model {self.model!r} is not in that list; requests may "
                    "fail if the ID does not match the loaded checkpoint."
                ),
                "models": models,
            }
        return {
            "ready": True,
            "message": f"Unsloth is ready ({', '.join(models)}).",
            "models": models,
        }

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
                    "name": "audio_quality_verification",
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
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise OverlapVerifierError(
                "Unsloth returned no assistant message content. The server "
                "may still be loading the model."
            ) from exc
        if content in (None, "", []):
            raise OverlapVerifierError(
                "Unsloth returned empty assistant content"
                + (
                    f" (finish_reason={finish_reason!r})"
                    if finish_reason
                    else ""
                )
                + ". The model may still be loading or the request was dropped."
            )
        try:
            return _normalize_result(content, backend="Unsloth")
        except OverlapVerifierError as exc:
            if finish_reason in {"length", "max_tokens"}:
                raise OverlapVerifierError(
                    f"Unsloth response was truncated due to max_output_tokens={self.max_output_tokens} "
                    f"(finish_reason={finish_reason!r}): {exc}"
                ) from exc
            raise


class GeminiOverlapVerifier(BaseOverlapVerifier):
    """Verify overlap with an audio-capable model through the Gemini API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        prompt: str = OVERLAP_PROMPT,
        max_output_tokens: int = DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS,
        concurrency: int | None = None,
    ) -> None:
        """Initialize the Gemini-backed verifier.

        Args:
            model: Gemini model ID. Defaults to ``GEMINI_MODEL`` or Gemini
                3.8 Flash.
            api_key: Gemini API key. Defaults to ``GEMINI_API_KEY``.
            timeout_s: HTTP request timeout in seconds.
            prompt: Instruction sent with every candidate audio segment.
            max_output_tokens: Maximum tokens allowed for the JSON decision.
            concurrency: Number of parallel candidate queries allowed when
                calling ``verify_batch()`` or batch tasks. Defaults to
                ``GEMINI_CONCURRENCY`` or 10.
        """
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL_ID
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.timeout_s = _validate_timeout(timeout_s)
        self.prompt = _validate_prompt(prompt)
        self.max_output_tokens = _validate_max_output_tokens(max_output_tokens)
        if concurrency is None:
            env_concurrency = os.getenv("GEMINI_CONCURRENCY")
            if env_concurrency is not None:
                try:
                    concurrency = int(env_concurrency)
                except ValueError:
                    pass
        if concurrency is None:
            concurrency = DEFAULT_GEMINI_CONCURRENCY
        self.concurrency = _validate_concurrency(concurrency)

    def check_ready(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Validate the API key and selected model with Gemini's model API."""
        if not self.api_key:
            return {
                "ready": False,
                "message": "Gemini API key is not configured; set GEMINI_API_KEY.",
                "models": [],
            }
        probe_timeout = (
            UNSLOTH_PROBE_TIMEOUT_S if timeout_s is None else _validate_timeout(timeout_s)
        )
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(self.model, safe='')}"
        )
        try:
            payload = _get_json(
                endpoint,
                headers={"x-goog-api-key": self.api_key},
                timeout_s=probe_timeout,
                backend="Gemini",
            )
        except OverlapVerifierError as exc:
            return {"ready": False, "message": str(exc), "models": []}
        methods = payload.get("supportedGenerationMethods", [])
        if isinstance(methods, list) and methods and "generateContent" not in methods:
            return {
                "ready": False,
                "message": f"{self.model} does not advertise generateContent support.",
                "models": [self.model],
            }
        return {
            "ready": True,
            "message": f"Gemini API key and model {self.model} are ready.",
            "models": [self.model],
        }

    def verify(self, audio: Audio) -> OverlapVerificationResult:
        """Send the audio segment directly to the configured Gemini model."""
        if not self.api_key:
            raise OverlapVerifierError(
                "Gemini API key is not configured; set GEMINI_API_KEY",
                readiness=True,
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
                "responseMimeType": "application/json",
                "responseSchema": _GEMINI_OVERLAP_SCHEMA,
                "responseJsonSchema": _OVERLAP_SCHEMA,
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
            candidates = response.get("candidates") or []
            if not candidates:
                feedback = response.get("promptFeedback", {})
                block_reason = feedback.get("blockReason")
                if block_reason:
                    raise OverlapVerifierError(
                        f"Gemini blocked the candidate audio: {block_reason}"
                    )
                raise OverlapVerifierError("Gemini returned no candidates in response")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")
            parts = candidate.get("content", {}).get("parts", [])
            content = "".join(
                str(part["text"])
                for part in parts
                if isinstance(part, dict)
                and "text" in part
                and not part.get("thought", False)
            )
        except OverlapVerifierError:
            raise
        except (KeyError, IndexError, TypeError) as exc:
            raise OverlapVerifierError(
                "Gemini returned no candidate message content"
            ) from exc

        if not content.strip():
            if finish_reason == "MAX_TOKENS":
                raise OverlapVerifierError(
                    f"Gemini output exceeded max_output_tokens={self.max_output_tokens} "
                    f"(finishReason='MAX_TOKENS'). Increase max_output_tokens."
                )
            if finish_reason == "SAFETY":
                raise OverlapVerifierError(
                    "Gemini candidate was blocked by safety settings (finishReason='SAFETY')"
                )
            if finish_reason:
                raise OverlapVerifierError(
                    f"Gemini returned no candidate message content (finishReason={finish_reason!r})"
                )
            raise OverlapVerifierError("Gemini returned no candidate message content")

        try:
            result = _normalize_result(content, backend="Gemini")
        except OverlapVerifierError as exc:
            if finish_reason == "MAX_TOKENS":
                raise OverlapVerifierError(
                    f"Gemini response was truncated due to max_output_tokens={self.max_output_tokens} "
                    f"(finishReason='MAX_TOKENS'): {exc}"
                ) from exc
            raise
        usage = _normalize_gemini_usage(
            response.get("usageMetadata"), audio_duration_s=audio.duration_s
        )
        result["usage"] = usage
        result["cost"] = _estimate_gemini_cost(self.model, usage)
        return result


def create_overlap_verifier(
    config: Mapping[str, Any] | None = None,
) -> BaseOverlapVerifier:
    """Create the selected verifier from a small flat configuration mapping.

    Args:
        config: Settings containing ``backend`` (``"gemma4"``, ``"gemini"``,
            or ``"gemini-flash-lite"``) and optional ``endpoint``, ``model``,
            ``api_key``, ``timeout_s``, ``prompt``, and ``max_output_tokens``
            values. ``backend`` falls back to the ``OVERLAP_VERIFIER``
            environment variable.

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
    if backend in {"gemini", "gemini-api", "google-gemini", "gemini-3.1", "gemini-3.1-pro"}:
        if "endpoint" in settings:
            raise ValueError("Gemini overlap verifier does not accept endpoint")
        return GeminiOverlapVerifier(**settings)
    if backend in {
        "gemini-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-3.1",
    }:
        if "endpoint" in settings:
            raise ValueError("Gemini overlap verifier does not accept endpoint")
        if not settings.get("model"):
            settings["model"] = DEFAULT_GEMINI_FLASH_LITE_MODEL_ID
        return GeminiOverlapVerifier(**settings)
    raise ValueError(
        "Select an overlap verifier with backend='gemma4', 'gemini', or "
        "'gemini-flash-lite'"
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
    return _send_json_request(request, timeout_s=timeout_s, backend=backend)


def _get_json(
    endpoint: str,
    *,
    headers: Mapping[str, str],
    timeout_s: float,
    backend: str,
) -> dict[str, Any]:
    """GET JSON and return a decoded object with backend-specific errors."""
    request = Request(endpoint, headers=dict(headers), method="GET")
    return _send_json_request(request, timeout_s=timeout_s, backend=backend)


def _send_json_request(
    request: Request,
    *,
    timeout_s: float,
    backend: str,
) -> dict[str, Any]:
    """Execute one JSON HTTP request and fail loudly on transport or API errors."""
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OverlapVerifierError(
            f"{backend} request failed with HTTP {exc.code}: {detail}",
            readiness=_http_code_is_readiness(exc.code),
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OverlapVerifierError(
            f"{backend} is not reachable: {exc}",
            readiness=True,
        ) from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OverlapVerifierError(f"{backend} returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise OverlapVerifierError(f"{backend} returned a non-object response")
    api_error = decoded.get("error")
    if api_error:
        if isinstance(api_error, dict):
            detail = str(api_error.get("message") or api_error)
        else:
            detail = str(api_error)
        raise OverlapVerifierError(
            f"{backend} returned an error: {detail}",
            readiness=_message_is_readiness(detail),
        )
    return decoded


def _normalize_result(content: Any, *, backend: str) -> OverlapVerificationResult:
    """Validate a structured answer and derive the final quality decision."""
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    if isinstance(content, str):
        stripped = content.strip()
        parsed: Any = None
        # 1. Direct JSON parse
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # 2. Extract from markdown code fence ```json ... ``` or ``` ... ```
        if parsed is None and "```" in stripped:
            first_fence = stripped.find("```")
            after_fence = stripped[first_fence + 3 :]
            if after_fence.lower().startswith("json"):
                after_fence = after_fence[4:]
            end_fence = after_fence.find("```")
            fence_block = (
                after_fence[:end_fence].strip()
                if end_fence != -1
                else after_fence.strip()
            )
            try:
                parsed = json.loads(fence_block)
            except json.JSONDecodeError:
                pass

        # 3. Extract outermost JSON object { ... }
        if parsed is None:
            first_brace = stripped.find("{")
            last_brace = stripped.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                brace_block = stripped[first_brace : last_brace + 1]
                try:
                    parsed = json.loads(brace_block)
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            raise OverlapVerifierError(
                f"{backend} returned an invalid audio-quality result: {content!r}"
            )
        content = parsed

    if not isinstance(content, dict):
        raise OverlapVerifierError(f"{backend} returned a non-object audio-quality result")
    speaker_purity = str(content.get("speaker_purity") or "").strip().lower()
    word_completeness = str(content.get("word_completeness") or "").strip().lower()
    boundary_issue = str(content.get("boundary_issue") or "").strip().lower()
    raw_failure_codes = content.get("failure_codes")
    reason = str(content.get("reason") or "").strip()
    if speaker_purity not in {"pure", "impure", "uncertain"}:
        raise OverlapVerifierError(f"{backend} result has invalid 'speaker_purity'")
    if word_completeness not in {"complete", "incomplete", "uncertain"}:
        raise OverlapVerifierError(f"{backend} result has invalid 'word_completeness'")
    if boundary_issue not in {
        "none", "clipped_start", "clipped_end", "clipped_both", "uncertain"
    }:
        raise OverlapVerifierError(f"{backend} result has invalid 'boundary_issue'")
    if not isinstance(raw_failure_codes, list):
        raise OverlapVerifierError(f"{backend} result has invalid 'failure_codes'")

    failure_codes: list[str] = []
    for code in raw_failure_codes:
        normalized_code = str(code).strip().lower() if isinstance(code, str) else ""
        if normalized_code not in _FAILURE_CODES:
            raise OverlapVerifierError(f"{backend} result has invalid failure code: {code!r}")
        failure_codes.append(normalized_code)

    if not reason:
        raise OverlapVerifierError(
            f"{backend} result field 'reason' is not a non-empty string"
        )
    speaker_codes = {
        "overlapping_speech",
        "secondary_speaker",
        "tail_speaker_intrusion",
    }
    clipped_codes = {"clipped_word_start", "clipped_word_end"}
    if speaker_purity == "pure" and speaker_codes.intersection(failure_codes):
        raise OverlapVerifierError(
            f"{backend} result contradicts its speaker-purity decision"
        )
    if speaker_purity == "impure" and not speaker_codes.intersection(failure_codes):
        raise OverlapVerifierError(
            f"{backend} impure result lacks a speaker failure code"
        )
    if word_completeness == "complete" and (
        boundary_issue != "none" or clipped_codes.intersection(failure_codes)
    ):
        raise OverlapVerifierError(
            f"{backend} result contradicts its word-completeness decision"
        )
    if word_completeness == "incomplete" and not (
        clipped_codes.intersection(failure_codes)
        or "unintelligible_boundary" in failure_codes
    ):
        raise OverlapVerifierError(
            f"{backend} incomplete result lacks a boundary failure code"
        )
    if speaker_purity == "pure" and word_completeness == "complete":
        decision = "pass"
    elif speaker_purity == "impure" or word_completeness == "incomplete":
        decision = "reject"
    else:
        decision = "uncertain"
    overlap = "overlapping_speech" in failure_codes
    return {
        "overlap": overlap,
        "speaker_purity": speaker_purity,
        "word_completeness": word_completeness,
        "boundary_issue": boundary_issue,
        "failure_codes": list(dict.fromkeys(failure_codes)),
        "decision": decision,
        "reason": reason,
        "usage": None,
        "cost": None,
    }


def _normalize_gemini_usage(
    value: Any,
    *,
    audio_duration_s: float | None = None,
) -> dict[str, Any]:
    """Normalize Gemini token metadata, retaining modality-level input counts."""
    metadata = value if isinstance(value, dict) else {}
    modalities: dict[str, int] = {}
    for detail in metadata.get("promptTokensDetails", []):
        if not isinstance(detail, dict):
            continue
        modality = str(detail.get("modality", "unknown")).lower()
        count = detail.get("tokenCount", 0)
        if isinstance(count, int) and not isinstance(count, bool):
            modalities[modality] = modalities.get(modality, 0) + count
    prompt_tokens = int(metadata.get("promptTokenCount", 0) or 0)
    audio_tokens = modalities.get("audio", 0)
    text_tokens = modalities.get("text", 0)
    audio_tokens_estimated = False
    if not audio_tokens and audio_duration_s and prompt_tokens:
        audio_tokens = min(prompt_tokens, round(float(audio_duration_s) * 32))
        text_tokens = max(text_tokens, prompt_tokens - audio_tokens)
        audio_tokens_estimated = True
    return {
        "prompt_tokens": prompt_tokens,
        "audio_input_tokens": audio_tokens,
        "audio_input_tokens_estimated": audio_tokens_estimated,
        "text_input_tokens": text_tokens,
        "output_tokens": int(metadata.get("candidatesTokenCount", 0) or 0),
        "thinking_tokens": int(metadata.get("thoughtsTokenCount", 0) or 0),
        "total_tokens": int(metadata.get("totalTokenCount", 0) or 0),
        "service_tier": metadata.get("serviceTier"),
    }


def _estimate_gemini_cost(model: str, usage: Mapping[str, Any]) -> dict[str, Any] | None:
    """Estimate one request at Google's versioned paid Standard list price."""
    rates = _GEMINI_STANDARD_PRICES.get(model)
    if rates is None:
        return None
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    audio_tokens = int(usage.get("audio_input_tokens", 0) or 0)
    text_tokens = int(usage.get("text_input_tokens", 0) or 0)
    if "audio_input" in rates and audio_tokens + text_tokens > 0:
        other_tokens = max(0, prompt_tokens - audio_tokens - text_tokens)
        input_usd = (
            audio_tokens * rates["audio_input"]
            + (text_tokens + other_tokens) * rates["input"]
        ) / 1_000_000
    else:
        input_usd = prompt_tokens * rates["input"] / 1_000_000
    billed_output_tokens = int(usage.get("output_tokens", 0) or 0) + int(
        usage.get("thinking_tokens", 0) or 0
    )
    output_usd = billed_output_tokens * rates["output"] / 1_000_000
    return {
        "input_usd": round(input_usd, 9),
        "output_usd": round(output_usd, 9),
        "total_usd": round(input_usd + output_usd, 9),
        "currency": "USD",
        "pricing_tier": "paid_standard",
        "rate_card_as_of": GEMINI_PRICE_CARD_AS_OF,
        "estimated": True,
    }


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


def _validate_concurrency(concurrency: int) -> int:
    """Return a positive concurrency limit."""
    if isinstance(concurrency, bool) or not isinstance(concurrency, int):
        raise TypeError("concurrency must be an integer")
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")
    return concurrency


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


def _unsloth_models_url(endpoint: str) -> str:
    """Derive the OpenAI-compatible ``/v1/models`` URL from a chat endpoint."""
    url = str(endpoint).rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")] + "/models"
    if url.endswith("/v1"):
        return f"{url}/models"
    return f"{url}/models"


def _openai_model_ids(payload: Mapping[str, Any]) -> list[str]:
    """Extract model IDs from an OpenAI-style ``/v1/models`` body."""
    data = payload.get("data")
    if not isinstance(data, list):
        model = payload.get("id") or payload.get("model")
        return [str(model)] if model else []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
        elif isinstance(item, str) and item.strip():
            ids.append(item.strip())
    return ids


def _http_code_is_readiness(code: int) -> bool:
    """True when the HTTP status means the server is down or still loading."""
    return code in {408, 429, 502, 503, 504} or code >= 500


def _message_is_readiness(message: str) -> bool:
    """True when an API error string means the backend is not ready."""
    text = message.lower()
    markers = (
        "not ready",
        "not loaded",
        "still loading",
        "model is loading",
        "no model",
        "unavailable",
        "overloaded",
        "connection refused",
        "timed out",
        "timeout",
    )
    return any(marker in text for marker in markers)


def _unsloth_unready_message(endpoint: str, detail: str) -> str:
    """Human-readable Unsloth unreadiness text for the workbench."""
    return (
        f"Unsloth is not ready at {endpoint}. {detail.rstrip('.')}."
        " Start Unsloth Studio, wait until Gemma finishes loading, then retry."
    )


def is_overlap_readiness_error(exc: BaseException) -> bool:
    """True when verification failed because the backend is down or loading."""
    if isinstance(exc, OverlapVerifierError) and exc.readiness:
        return True
    return _message_is_readiness(str(exc))
