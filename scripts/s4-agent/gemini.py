from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
import mimetypes
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common.files import (  # noqa: E402
    ROOT,
    digest,
    persist_path,
    positive_int,
    read_json,
    resolve_stored_path,
    safe_name,
    write_json,
)
from _gemini_pricing import (  # noqa: E402
    accumulate_cost,
    accumulate_usage,
    empty_cost_totals,
    empty_usage_totals,
    estimate_gemini_cost,
    estimate_cache_storage_cost,
    normalize_gemini_usage,
)

logger = logging.getLogger("agent.gemini")


def _load_repo_env() -> None:
    """Load environment variables from <repo>/.env without overriding the shell."""
    env_file = ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file if env_file.is_file() else None, override=False)
    except Exception:
        pass
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass


# Must run before SYSTEM_PROMPT_PATH so GEMINI_SYSTEM_PROMPT in .env is honored.
_load_repo_env()

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_USER_PROMPT = "Analyze the attached audio according to the system instructions."
# Portable default: <repo>/prompts/full-tags-prompt.md, overridable per machine.
DEFAULT_SYSTEM_PROMPT_PATH = ROOT / "prompts" / "full-tags-prompt.md"
SYSTEM_PROMPT_PATH = Path(
    os.path.expanduser(os.getenv("GEMINI_SYSTEM_PROMPT") or str(DEFAULT_SYSTEM_PROMPT_PATH))
).resolve()
BATCH_MAX_INLINE_BYTES = 18 * 1024 * 1024
MAX_BACKOFF_S = 60.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
HTTP_HINTS = {
    400: "request rejected; check audio format/size and generation config",
    401: "check GEMINI_API_KEY",
    403: "check GEMINI_API_KEY and its project permissions/billing",
    404: f"model {GEMINI_MODEL!r} is not available to this API key",
    429: "quota or rate limit exhausted",
}
BATCH_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}
MIME_TYPES = {
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


class GeminiHTTPError(RuntimeError):
    """Provider HTTP failure; carries only the numeric status, never the body."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SystemPromptError(RuntimeError):
    """Prompt configuration error.

    Deliberately not a FileNotFoundError: the verifier maps FileNotFoundError to
    'input_not_found' (missing audio), which hid the real cause.
    """


class GeminiResponseError(RuntimeError):
    """An unusable provider answer, with generation evidence for artifact writers."""

    def __init__(self, generation: dict[str, Any]) -> None:
        error = generation["generation_error"]
        super().__init__(error["message"])
        self.generation = generation
        self.raw_response = generation["text"]


def response_error(response: dict[str, Any]) -> dict[str, Any] | None:
    """Check provider completion only; task JSON and acoustic verdicts are separate."""
    feedback = response.get("promptFeedback") or {}
    block = _safe_status_name(feedback.get("blockReason"))
    candidates = response.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    finish = _safe_status_name(candidate.get("finishReason"))
    if block and block != "BLOCK_REASON_UNSPECIFIED":
        code, retryable, detail = "gemini_prompt_blocked", False, f"prompt block {block}"
    elif finish == "MAX_TOKENS":
        code, retryable, detail = "gemini_output_truncated", False, "MAX_TOKENS; review --max-tokens and thinking level"
    elif finish not in {None, "STOP", "FINISH_REASON_UNSPECIFIED"}:
        code, retryable, detail = "gemini_incomplete_response", finish in {
            "OTHER", "MALFORMED_RESPONSE", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL",
        }, f"finish reason {finish}"
    elif not response_text(response).strip():
        code, retryable, detail = "gemini_empty_response", True, "no final-answer text"
    elif finish != "STOP":
        code, retryable, detail = "gemini_incomplete_response", True, "missing final STOP reason"
    else:
        return None
    return {"code": code, "retryable": retryable, "message": f"Gemini returned {detail}."}


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def load_system_prompt(path: Path | None = None) -> str:
    """Load the pipeline's single authorized instruction source."""
    path = Path(path) if path is not None else SYSTEM_PROMPT_PATH
    if not path.is_file():
        raise SystemPromptError(
            f"Required system prompt not found: {path}. "
            f"Put it at {DEFAULT_SYSTEM_PROMPT_PATH} or set GEMINI_SYSTEM_PROMPT."
        )
    system_prompt = path.read_text(encoding="utf-8")
    if not system_prompt.strip():
        raise SystemPromptError(f"System prompt is empty: {path}")
    return system_prompt


def system_prompt_ref() -> Any:
    """Machine-independent prompt reference stored in metadata and manifests."""
    return persist_path(SYSTEM_PROMPT_PATH)


def audio_mime_type(path: Path) -> str:
    return (
        MIME_TYPES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def audio_duration_s(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return None


def _safe_status_name(value: Any) -> str | None:
    """Google status enums (e.g. NOT_FOUND) are safe to log; free text is not."""
    if isinstance(value, str) and value and value.replace("_", "").isalpha() and value.isupper():
        return value
    return None


def http_error_message(operation: str, code: Any, status_name: Any = None) -> str:
    label = f"HTTP {code or 'unknown'}"
    name = _safe_status_name(status_name)
    if name:
        label += f" {name}"
    hint = HTTP_HINTS.get(code) if isinstance(code, int) else None
    return f"Gemini {operation} failed: {label}" + (f"; {hint}" if hint else "")


def response_text(response: dict[str, Any]) -> str:
    """Return only final-answer text, matching the SDK's ``response.text`` behavior."""
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return ""
    content = candidates[0].get("content")
    if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
        return ""
    return "".join(
        part["text"]
        for part in content["parts"]
        if (
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and not part.get("thought", False)
        )
    )


def configure_gemini_paths(args: Any, operation: str) -> Path:
    """Separate default runtime paths by exact Gemini model and reasoning level."""
    old_base = ROOT / ".data" / operation / "gemini"
    variant_base = old_base / safe_name(args.model) / safe_name(args.reasoning_effort)
    if Path(args.work_dir) == old_base / "work":
        args.work_dir = variant_base / "work"
    args._default_base = variant_base
    return variant_base


class GeminiAgent:
    """Raw generation client for Gemini standard, Flex, and asynchronous Batch APIs."""

    def __init__(
        self,
        model: str = GEMINI_MODEL,
        api_key: str | None = None,
        reasoning_effort: str = "medium",
        max_retries: int | None = None,
        max_retry: int | None = None,
        base_backoff_s: float = 2.0,
        max_tokens: int = 65536,
        timeout_s: float | None = None,
        inference_mode: str = "standard",
        cache_prompt: bool = False,
        cache_ttl_s: int | None = None,
        user_prompt: str = DEFAULT_USER_PROMPT,
        max_response_retries: int = 3,
    ) -> None:
        if model != GEMINI_MODEL:
            raise ValueError(
                f"This pipeline is pinned to {GEMINI_MODEL!r}; received {model!r}."
            )
        if inference_mode not in ("standard", "flex", "batch"):
            raise ValueError(f"Unsupported Gemini inference mode: {inference_mode}")
        self.model = model
        if not user_prompt.strip():
            raise ValueError("Gemini requires nonempty --user-prompt text")
        if max_response_retries < 0:
            raise ValueError("max_response_retries must be zero or greater")
        self.user_prompt = user_prompt
        self.max_response_retries = max_response_retries
        self.inference_mode = inference_mode
        self.cache_prompt = cache_prompt
        self.cache_ttl_s = cache_ttl_s or (90000 if inference_mode == "batch" else 3600)
        self._cache_lock = threading.Lock()
        self._cache = None
        self._cache_key = None
        self._cache_expires = 0.0
        # Set when the API refuses to cache this prompt (e.g. below the minimum
        # token count); non-batch modes then send the prompt inline instead.
        self._cache_unavailable = False
        self._pending_storage_usd = 0.0
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable or .env setting is required for GeminiAgent."
            )
        self.reasoning_effort = reasoning_effort
        if max_retry is not None:
            max_retries = max_retry + 1
        self.max_retries = max_retries if max_retries is not None else (12 if inference_mode == "flex" else 5)
        self.base_backoff_s = base_backoff_s
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s if timeout_s is not None else (900.0 if inference_mode == "flex" else 120.0)
        try:
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError(
                "The official Google Gen AI SDK is required. Install it with: "
                "pip install --upgrade google-genai"
            ) from exc
        if inference_mode == "standard" and not hasattr(genai_types, "ThinkingLevel"):
            raise RuntimeError(
                "Installed google-genai is too old (no types.ThinkingLevel). "
                "Upgrade with: pip install --upgrade google-genai"
            )
        self._genai_types = genai_types
        self._genai_errors = genai_errors
        # Previously the SDK client ignored --timeout-s entirely. HttpOptions.timeout is in ms.
        self._genai_client = genai.Client(
            api_key=self.api_key,
            vertexai=False,
            http_options=genai_types.HttpOptions(
                api_version="v1beta",
                timeout=int(self.timeout_s * 1000),
                # Our loop owns the retry budget; SDK retries would multiply it.
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )
        self._lock = threading.Lock()
        self._preflight_lock = threading.Lock()
        self._preflight_ok = False
        self._usage_totals = empty_usage_totals()
        self._cost_totals = empty_cost_totals()
        self._cost_totals["model"] = model
        self._session = httpx.Client(timeout=self.timeout_s)

    def preflight(self) -> None:
        """Verify API key and model access once (free metadata call, no tokens).

        Bad keys or an unavailable model now stop the run with one clear error
        instead of producing N failed artifacts and requests=0.
        """
        with self._preflight_lock:
            if self._preflight_ok:
                return
            self._request_json("GET", f"{API_ROOT}/models/{self.model}", operation="preflight")
            self._preflight_ok = True
            logger.info("Gemini preflight OK: %s is reachable with this API key", self.model)

    def _sdk_thinking_level(self) -> Any:
        """Map the CLI level to the official Google Gen AI SDK enum."""
        levels = {
            "low": self._genai_types.ThinkingLevel.LOW,
            "medium": self._genai_types.ThinkingLevel.MEDIUM,
            "high": self._genai_types.ThinkingLevel.HIGH,
        }
        return levels[self.reasoning_effort.lower()]

    def _sdk_generate_standard(
        self,
        audio_path: Path,
        system_prompt: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Call Gemini through the official SDK with audio and a user instruction."""
        audio_part = self._genai_types.Part.from_bytes(
            data=audio_path.read_bytes(),
            mime_type=audio_mime_type(audio_path),
        )
        contents = self._genai_types.Content(role="user", parts=[
            audio_part, self._genai_types.Part.from_text(text=self.user_prompt),
        ])

        for attempt in range(1, self.max_retries + 1):
            retry_after = None
            # Resolved per attempt so a cache refreshed during retries is picked up.
            cache_name = self._ensure_prompt_cache(system_prompt)
            prompt_config: dict[str, Any] = (
                # The API rejects system_instruction together with cached_content.
                {"cached_content": cache_name}
                if cache_name
                else {"system_instruction": system_prompt}
            )
            config = self._genai_types.GenerateContentConfig(
                **prompt_config,
                max_output_tokens=self.max_tokens,
                thinking_config=self._genai_types.ThinkingConfig(
                    thinking_level=self._sdk_thinking_level(),
                ),
            )
            try:
                response = self._genai_client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                body = response.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                    exclude={"sdk_http_response"},
                )
                http_response = getattr(response, "sdk_http_response", None)
                response_headers = getattr(http_response, "headers", None) or {}
                headers = {
                    key: value
                    for key, value in dict(response_headers).items()
                    if key.lower() in {
                        "content-type",
                        "date",
                        "server",
                        "x-request-id",
                    }
                }
                return body, headers
            except self._genai_errors.APIError as exc:
                status = getattr(exc, "code", None)
                if status not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise RuntimeError(
                        http_error_message("SDK request", status, getattr(exc, "status", None))
                    ) from None
                error_response = getattr(exc, "response", None)
                if error_response is not None:
                    retry_after = error_response.headers.get("retry-after")
            except httpx.TransportError as exc:
                # Previously uncaught: network blips failed the item without retry.
                if attempt == self.max_retries:
                    if isinstance(exc, httpx.TimeoutException):
                        raise TimeoutError(
                            f"Gemini SDK request timed out after {self.timeout_s:g}s"
                        ) from None
                    raise RuntimeError("Gemini SDK connection failed") from None
            delay = self._retry_delay_s(attempt, retry_after)
            logger.warning(
                "Gemini SDK request failed; retry %d/%d in %.2fs",
                attempt,
                self.max_retries - 1,
                delay,
            )
            time.sleep(delay)
        raise RuntimeError("Gemini SDK request exhausted retries")

    def get_cost_summary(self) -> dict[str, Any]:
        """Return a copy of cumulative usage and estimated USD cost for this session."""
        with self._lock:
            usage = dict(self._usage_totals)
            cost = dict(self._cost_totals)
            for key in (
                "uncached_input_usd",
                "cached_input_usd",
                "cache_storage_usd",
                "input_usd",
                "output_usd",
                "total_usd",
            ):
                cost[key] = round(float(cost.get(key, 0.0)), 9)
            return {
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "usage": usage,
                "cost": cost,
            }

    def reset_cost_summary(self) -> None:
        """Clear cumulative usage/cost counters."""
        with self._lock:
            self._usage_totals = empty_usage_totals()
            self._cost_totals = empty_cost_totals()
            self._cost_totals["model"] = self.model
            self._pending_storage_usd = 0.0

    def _backoff_s(self, attempt: int) -> float:
        """Exponential backoff capped so flex 503 storms retry for minutes, not hours."""
        return min(self.base_backoff_s * (2 ** (attempt - 1)), MAX_BACKOFF_S)

    def _retry_delay_s(self, attempt: int, retry_after: str | None) -> float:
        """Respect Google's Retry-After seconds/date when longer than local backoff."""
        delay = self._backoff_s(attempt) + random.uniform(0.1, 0.5)
        if retry_after:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    seconds = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    return delay
            if math.isfinite(seconds):
                delay = max(delay, seconds)
        return delay

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        retry_network_errors: bool = True,
        operation: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if operation is None:
            operation = "prompt cache creation" if url.endswith("/cachedContents") else "request"
        headers = {"x-goog-api-key": self.api_key}
        for attempt in range(1, self.max_retries + 1):
            retry_after = None
            try:
                response = self._session.request(method, url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if not retry_network_errors or attempt == self.max_retries:
                    if isinstance(exc, httpx.TimeoutException):
                        raise TimeoutError(
                            f"Gemini {operation} timed out after {self.timeout_s:g}s"
                        ) from None
                    raise RuntimeError(f"Gemini {operation}: connection failed") from None
            else:
                if response.is_success:
                    return response.json(), {
                        key: value for key, value in response.headers.items()
                        if key in {"content-type", "date", "server", "x-request-id"}
                    }
                if response.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    # Only the numeric code and Google's status enum are surfaced;
                    # provider bodies may contain secrets.
                    status_name = None
                    try:
                        body = response.json()
                        error = body.get("error") if isinstance(body, dict) else None
                        status_name = error.get("status") if isinstance(error, dict) else None
                    except ValueError:
                        pass
                    message = http_error_message(operation, response.status_code, status_name)
                    if operation == "prompt cache creation":
                        message += "; check model caching support and minimum prompt size"
                    raise GeminiHTTPError(message, response.status_code)
                retry_after = response.headers.get("retry-after")
            delay = self._retry_delay_s(attempt, retry_after)
            logger.warning("Gemini %s failed; retry %d/%d in %.2fs", operation, attempt, self.max_retries - 1, delay)
            time.sleep(delay)
        raise RuntimeError(f"Gemini {operation} exhausted retries")

    def _ensure_prompt_cache(self, system_prompt: str) -> str | None:
        """Return a cachedContents name for the system prompt, creating it if needed.

        Shared by all modes: a cache created over REST is referenced by name from
        the SDK (standard) as well as from REST payloads (flex/batch).
        """
        if not self.cache_prompt:
            return None
        key = system_prompt
        with self._cache_lock:
            if self._cache_unavailable:
                return None
            if key == self._cache_key and time.monotonic() < self._cache_expires:
                return self._cache
            body = {
                "model": f"models/{self.model}",
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "ttl": f"{self.cache_ttl_s}s",
            }
            started = time.monotonic()
            try:
                cache, _ = self._request_json(
                    "POST", f"{API_ROOT}/cachedContents", body, retry_network_errors=False
                )
            except GeminiHTTPError as exc:
                if exc.status_code == 400 and self.inference_mode != "batch":
                    self._cache_unavailable = True
                    logger.warning(
                        "Prompt cache refused (%s). Continuing WITHOUT caching: the "
                        "system prompt is sent inline, output is unchanged, cost is higher.",
                        exc,
                    )
                    return None
                raise
            self._cache = cache["name"]
            self._cache_key = key
            reserve_s = 86400 if self.inference_mode == "batch" else min(60, self.cache_ttl_s / 10)
            self._cache_expires = started + self.cache_ttl_s - reserve_s
            tokens = int(cache.get("usageMetadata", {}).get("totalTokenCount", 0))
            storage = estimate_cache_storage_cost(self.model, tokens, self.cache_ttl_s, self.inference_mode)
            with self._lock:
                if storage is None or not tokens:
                    self._cost_totals["unpriced_caches"] += 1
                else:
                    self._cost_totals["cache_storage_usd"] += storage
                    self._cost_totals["total_usd"] += storage
                    self._pending_storage_usd += storage
            logger.info("Created prompt cache %s (%d tokens, TTL %ds)", self._cache, tokens, self.cache_ttl_s)
            return self._cache

    def _apply_prompt_cache(self, payload: dict[str, Any], system_prompt: str) -> None:
        """Cache the fixed system prompt; audio always stays in the request."""
        cache_name = self._ensure_prompt_cache(system_prompt)
        if cache_name:
            payload["cachedContent"] = cache_name
            payload.pop("systemInstruction", None)

    def _request_payload(
        self,
        audio_path: Path,
        *,
        system_prompt: str,
    ) -> tuple[dict[str, Any], float | None]:
        audio_path = Path(audio_path)
        with audio_path.open("rb") as stream:
            audio_b64 = base64.b64encode(stream.read()).decode("ascii")

        payload: dict[str, Any] = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": audio_mime_type(audio_path), "data": audio_b64}},
                    {"text": self.user_prompt},
                ],
            }],
            # Gemini 3.x is tuned for its default sampler. Do not set temperature,
            # topP, or topK here: Google explicitly recommends omitting them.
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": self.reasoning_effort.lower()
            }
        return payload, audio_duration_s(audio_path)

    def _generation_result(
        self,
        response: dict[str, Any],
        *,
        latency_s: float,
        audio_duration_s: float | None,
        pricing_tier: str,
        headers: dict[str, str] | None = None,
        batch_job: str | None = None,
        batch_request_key: str | None = None,
    ) -> dict[str, Any]:
        returned_model = response.get("modelVersion")
        if isinstance(returned_model, str) and returned_model:
            normalized = returned_model.removeprefix("models/")
            if not normalized.startswith(GEMINI_MODEL):
                raise RuntimeError(
                    f"Gemini returned unexpected modelVersion {returned_model!r}; "
                    f"expected {GEMINI_MODEL!r}. No fallback is allowed."
                )

        usage = normalize_gemini_usage(
            response.get("usageMetadata"),
            audio_duration_s=audio_duration_s,
            explicit_text_cache=self.cache_prompt and not self._cache_unavailable,
        )
        if pricing_tier != "paid_batch":
            returned_tier = str(usage.get("service_tier") or "").lower()
            if returned_tier in {"standard", "flex"}:
                pricing_tier = f"paid_{returned_tier}"
        cost = estimate_gemini_cost(
            self.model,
            usage,
            pricing_tier=pricing_tier,
        )
        with self._lock:
            accumulate_usage(self._usage_totals, usage)
            accumulate_cost(self._cost_totals, cost)
            if cost is not None:
                # Assign storage once to an artifact so offline sums include it too.
                cost["cache_storage_usd"] = round(self._pending_storage_usd, 9)
                cost["total_usd"] = round(cost["total_usd"] + self._pending_storage_usd, 9)
                self._pending_storage_usd = 0.0
        result = {
            "text": response_text(response),
            "latency_s": round(latency_s, 3),
            "headers": headers or {},
            "usage": usage,
            "cost": cost,
            "inference_mode": pricing_tier.removeprefix("paid_"),
            "requested_inference_mode": self.inference_mode,
            "cache_prompt": self.cache_prompt,
            "provider_body": response,
            "requested_model": self.model,
            "model_version": returned_model,
        }
        error = response_error(response)
        if error:
            result["generation_error"] = error
        if batch_job is not None:
            result["batch_job"] = batch_job
        if batch_request_key is not None:
            result["batch_request_key"] = batch_request_key
        return result

    def generate(
        self,
        audio_path: Path,
        *,
        system_prompt: str,
    ) -> dict[str, Any]:
        if self.inference_mode == "batch":
            raise ValueError("Batch mode requires generate_batch(), not generate()")
        attempts = []
        started = time.monotonic()
        for attempt in range(self.max_response_retries + 1):
            try:
                result = self._generate_once(audio_path, system_prompt=system_prompt)
            except Exception as exc:
                if attempts:
                    # Preserve earlier paid responses even if the next transport fails.
                    result = dict(attempts[-1])
                    result["generation_error"] = {
                        "code": "gemini_retry_transport_failed", "retryable": False,
                        "message": "Gemini transport failed while retrying an unusable answer.",
                        "exception": type(exc).__name__,
                    }
                    break
                raise
            attempts.append(result)
            error = result.get("generation_error")
            if error is None or not error["retryable"] or attempt == self.max_response_retries:
                break
            delay = self._backoff_s(attempt + 1) + random.uniform(0.1, 0.5)
            logger.warning("%s Response retry %d/%d in %.2fs", error["message"],
                           attempt + 1, self.max_response_retries, delay)
            time.sleep(delay)
        result = dict(result)
        result["latency_s"] = round(time.monotonic() - started, 3)
        if len(attempts) > 1:
            result["attempts"] = attempts
            usage, cost = empty_usage_totals(), empty_cost_totals()
            for generated in attempts:
                accumulate_usage(usage, generated["usage"])
                accumulate_cost(cost, generated["cost"])
            result["usage"] = usage
            result["cost"] = cost if cost["priced_requests"] else None
        if result.get("generation_error"):
            if result["generation_error"]["retryable"]:
                result["generation_error"] = {
                    **result["generation_error"],
                    "retries_exhausted": True,
                    "message": result["generation_error"]["message"]
                    + f" Response retry budget exhausted after {len(attempts)} response(s).",
                }
            raise GeminiResponseError(result)
        return result

    def _generate_once(
        self, audio_path: Path, *, system_prompt: str,
    ) -> dict[str, Any]:
        """One generation with the transport retry budget; no task parsing."""
        audio_path = Path(audio_path)
        started = time.monotonic()
        if self.inference_mode == "standard":
            duration = audio_duration_s(audio_path)
            response, headers = self._sdk_generate_standard(
                audio_path,
                system_prompt,
            )
        else:
            payload, duration = self._request_payload(
                audio_path,
                system_prompt=system_prompt,
            )
            self._apply_prompt_cache(payload, system_prompt)
            payload["serviceTier"] = self.inference_mode
            response, headers = self._request_json(
                "POST",
                f"{API_ROOT}/models/{self.model}:generateContent",
                payload,
            )
        return self._generation_result(
            response,
            latency_s=time.monotonic() - started,
            audio_duration_s=duration,
            pricing_tier=f"paid_{self.inference_mode}",
            headers=headers,
        )

    @staticmethod
    def _batch_state(job: dict[str, Any]) -> str:
        response = job.get("response") if isinstance(job.get("response"), dict) else {}
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        state = response.get("state") or job.get("state") or metadata.get("state")
        if state:
            return str(state)
        if job.get("done") is True:
            return "BATCH_STATE_FAILED" if job.get("error") else "BATCH_STATE_SUCCEEDED"
        return "BATCH_STATE_UNSPECIFIED"

    @staticmethod
    def _batch_resource(job: dict[str, Any]) -> dict[str, Any]:
        response = job.get("response")
        if isinstance(response, dict) and ("dest" in response or "output" in response or "state" in response):
            return response
        return job

    @staticmethod
    def _batch_job_name(created: dict[str, Any]) -> str:
        name = created.get("name")
        metadata = created.get("metadata") if isinstance(created.get("metadata"), dict) else {}
        if not isinstance(name, str) or not name:
            name = metadata.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"Gemini Batch creation returned no job name: {created}")
        return name

    @staticmethod
    def _inline_responses(job: dict[str, Any]) -> list[dict[str, Any]]:
        resource = GeminiAgent._batch_resource(job)
        dest = (
            resource.get("dest")
            or resource.get("output")
            or (job.get("response") if isinstance(job.get("response"), dict) else {})
        )
        responses = None
        if isinstance(dest, dict):
            responses = (
                dest.get("inlinedResponses")
                or dest.get("inlined_responses")
                or dest.get("responses")
            )
        if responses is None:
            responses = (
                resource.get("inlinedResponses")
                or resource.get("inlined_responses")
                or resource.get("responses")
            )
        if isinstance(responses, dict):
            responses = (
                responses.get("inlinedResponses")
                or responses.get("inlined_responses")
                or responses.get("responses")
            )
        if not isinstance(responses, list):
            raise RuntimeError(f"Gemini Batch succeeded without inline responses: {job}")
        return [item for item in responses if isinstance(item, dict)]

    def generate_batch(
        self,
        audio_paths: list[Path],
        *,
        system_prompt: str,
        batch_size: int = 100,
        poll_interval_s: float = 10.0,
        batch_timeout_s: float = 86400.0,
        state_dir: Path,
        reuse_state: bool = True,
    ) -> dict[Path, dict[str, Any] | Exception]:
        """Submit missing audio requests to Gemini Batch and wait for results."""
        if self.inference_mode != "batch":
            raise ValueError("generate_batch() requires inference_mode='batch'")
        if self.cache_prompt and self.cache_ttl_s < 90000:
            raise ValueError("Batch prompt caching requires --cache-ttl-s >= 90000 (25 hours)")
        initial_storage = self._cost_totals["cache_storage_usd"]
        descriptors: list[dict[str, Any]] = []
        for index, source_value in enumerate(audio_paths):
            source = Path(source_value).resolve()
            payload, duration = self._request_payload(
                source,
                system_prompt=system_prompt,
            )
            key_material = {
                "index": index,
                "path": persist_path(source),
                "sha256": digest(source),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "cache_prompt": self.cache_prompt,
                "payload_without_audio": {
                    **payload,
                    "contents": [{"role": "user", "parts": []}],
                },
            }
            request_key = hashlib.sha256(
                json.dumps(key_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:32]
            entry = {"request": payload, "metadata": {"key": request_key}}
            entry_bytes = len(json.dumps(entry, separators=(",", ":")).encode("utf-8"))
            if entry_bytes > BATCH_MAX_INLINE_BYTES:
                raise ValueError(
                    f"Audio request is too large for inline Gemini Batch ({entry_bytes} bytes): {source}. "
                    "Compress/cut the audio or use --inference-mode standard."
                )
            descriptors.append({
                "source": source,
                "duration": duration,
                "key": request_key,
                "entry": entry,
                "bytes": entry_bytes,
            })

        if not descriptors:
            return {}

        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_bytes = 0
        for descriptor in descriptors:
            if current and (
                len(current) >= batch_size
                or current_bytes + int(descriptor["bytes"]) > BATCH_MAX_INLINE_BYTES
            ):
                groups.append(current)
                current = []
                current_bytes = 0
            current.append(descriptor)
            current_bytes += int(descriptor["bytes"])
        if current:
            groups.append(current)

        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        session_material = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "request_keys": [item["key"] for item in descriptors],
        }
        session_id = hashlib.sha256(
            json.dumps(session_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        state_path = state_dir / f"batch_{session_id}.json"
        saved: dict[str, Any] = {}
        if reuse_state:
            candidates = sorted(
                state_dir.glob(f"batch_{session_id}*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if candidates:
                state_path = candidates[0]
                saved = read_json(state_path)
                if saved.get("session") != session_material:
                    raise ValueError(f"Conflicting Batch state: {state_path}; use --overwrite")
        if not saved:
            if state_path.exists():
                state_path = state_dir / f"batch_{session_id}_{time.time_ns()}.json"
            saved = {
                "schema_version": 1,
                "session": session_material,
                "status": "submitting",
                "groups": [],
            }
            write_json(state_path, saved)

        saved_results = saved.get("results")
        if saved.get("status") == "succeeded" and isinstance(saved_results, dict):
            response_items = saved_results
            job_by_key = {
                key: value
                for group in saved.get("groups", [])
                if isinstance(group, dict)
                for key in group.get("request_keys", [])
                for value in [group.get("job_name")]
                if isinstance(key, str) and isinstance(value, str)
            }
            batch_latency = float(saved.get("latency_s", 0.0) or 0.0)
        else:
            saved_groups = saved.get("groups") if isinstance(saved.get("groups"), list) else []
            for group_index, group in enumerate(groups):
                keys = [str(item["key"]) for item in group]
                existing = saved_groups[group_index] if group_index < len(saved_groups) else None
                if (
                    not isinstance(existing, dict)
                    or existing.get("request_keys") != keys
                    or not existing.get("job_name")
                    or existing.get("state")
                    in BATCH_TERMINAL_STATES
                    - {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}
                ):
                    previous_storage = self._cost_totals["cache_storage_usd"]
                    for item in group:
                        self._apply_prompt_cache(item["entry"]["request"], system_prompt)
                    saved["cache_storage_usd"] = saved.get("cache_storage_usd", 0.0) + (
                        self._cost_totals["cache_storage_usd"] - previous_storage
                    )
                    write_json(state_path, saved)
                    body = {
                        "batch": {
                            "display_name": (
                                f"{safe_name(self.model)}-"
                                f"{safe_name(self.reasoning_effort)}-"
                                f"{session_id}-{group_index + 1}"
                            ),
                            "input_config": {
                                "requests": {"requests": [item["entry"] for item in group]}
                            },
                        }
                    }
                    created, _ = self._request_json(
                        "POST",
                        f"{API_ROOT}/models/{self.model}:batchGenerateContent",
                        body,
                        retry_network_errors=False,
                        operation="batch submission",
                    )
                    existing = {
                        "request_keys": keys,
                        "job_name": self._batch_job_name(created),
                        "state": self._batch_state(created),
                    }
                    if group_index < len(saved_groups):
                        saved_groups[group_index] = existing
                    else:
                        saved_groups.append(existing)
                    saved["groups"] = saved_groups
                    write_json(state_path, saved)
                    logger.info(
                        "Submitted Gemini Batch job %s with %d request(s).",
                        existing["job_name"],
                        len(group),
                    )

            started = time.monotonic()
            final_jobs: dict[str, dict[str, Any]] = {}
            while len(final_jobs) < len(saved_groups):
                for group in saved_groups:
                    job_name = str(group["job_name"])
                    if job_name in final_jobs:
                        continue
                    job_url = (
                        job_name
                        if job_name.startswith("http")
                        else f"https://generativelanguage.googleapis.com/{job_name}"
                        if job_name.startswith("v1beta/")
                        else f"{API_ROOT}/{job_name.lstrip('/')}"
                    )
                    job, _ = self._request_json("GET", job_url, operation="batch polling")
                    state = self._batch_state(job)
                    group["state"] = state
                    logger.info("Gemini Batch job %s: %s", job_name, state)
                    if state in BATCH_TERMINAL_STATES:
                        final_jobs[job_name] = job
                saved["status"] = "polling"
                saved["groups"] = saved_groups
                write_json(state_path, saved)
                if len(final_jobs) == len(saved_groups):
                    break
                if time.monotonic() - started >= batch_timeout_s:
                    raise TimeoutError(
                        f"Gemini Batch did not finish within {batch_timeout_s:g}s; "
                        f"resume with the same command plus --continue. State: {state_path}"
                    )
                time.sleep(min(poll_interval_s, 60.0))

            batch_latency = time.monotonic() - started
            response_items: dict[str, Any] = {}
            job_by_key: dict[str, str] = {}
            failed_jobs = []
            for group in saved_groups:
                job_name = str(group["job_name"])
                job = final_jobs[job_name]
                state = self._batch_state(job)
                keys = [str(key) for key in group.get("request_keys", [])]
                if state not in ("JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"):
                    failed_jobs.append(job_name)
                    error = self._batch_resource(job).get("error") or job.get("error") or {"state": state}
                    for key in keys:
                        response_items[key] = {"error": error}
                        job_by_key[key] = job_name
                    continue
                inline_responses = self._inline_responses(job)
                for index, item in enumerate(inline_responses):
                    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                    key = metadata.get("key") or metadata.get("custom_id") or metadata.get("customId")
                    if not isinstance(key, str) or key not in keys:
                        key = keys[index] if index < len(keys) else None
                    if key is not None:
                        response_items[key] = item
                        job_by_key[key] = job_name
                for key in keys:
                    if key not in response_items:
                        response_items[key] = {"error": {"message": "missing inline response"}}
                        job_by_key[key] = job_name

            saved.update({
                "status": "failed" if failed_jobs else "succeeded",
                "latency_s": round(batch_latency, 3),
                "results": response_items,
                "failed_jobs": failed_jobs,
            })
            write_json(state_path, saved)

        # Batch artifacts include the job's storage estimate even after a restart.
        with self._lock:
            storage = float(saved.get("cache_storage_usd", 0.0))
            new_storage = self._cost_totals["cache_storage_usd"] - initial_storage
            self._cost_totals["total_usd"] += storage - new_storage
            self._cost_totals["cache_storage_usd"] += storage - new_storage
            self._pending_storage_usd += storage - new_storage
        results: dict[Path, dict[str, Any] | Exception] = {}
        descriptor_by_key = {str(item["key"]): item for item in descriptors}
        for key, descriptor in descriptor_by_key.items():
            item = response_items.get(key)
            if not isinstance(item, dict):
                results[descriptor["source"]] = RuntimeError(
                    f"Gemini Batch returned no result for request {key}."
                )
                continue
            error = item.get("error")
            response = item.get("response")
            if response is None and "candidates" in item:
                response = item
            if error or not isinstance(response, dict):
                results[descriptor["source"]] = RuntimeError(
                    f"Gemini Batch request {key} failed: {json.dumps(error or item, ensure_ascii=False)}"
                )
                continue
            generated = self._generation_result(
                response,
                latency_s=batch_latency,
                audio_duration_s=descriptor["duration"],
                pricing_tier="paid_batch",
                batch_job=job_by_key.get(key),
                batch_request_key=key,
            )
            results[descriptor["source"]] = (
                GeminiResponseError(generated) if generated.get("generation_error") else generated
            )
        return results


def add_gemini_arguments(command: argparse.ArgumentParser) -> None:
    """Identical provider controls for raw generation and verification."""
    command.add_argument(
        "--user-prompt", default=DEFAULT_USER_PROMPT,
        help="Nonempty user text accompanying audio; keep the rubric in the system prompt",
    )
    command.add_argument(
        "--max-response-retries", type=nonnegative_int, default=3,
        help="Additional Standard/Flex generations for empty or transient incomplete answers "
             "(default: 3); separate from HTTP retries; no Batch resubmission",
    )
    command.add_argument(
        "--continue", dest="continue_run", action="store_true",
        help="Skip completed outputs (even with other prompt/settings) and rerun missing, failed, or incomplete ones; Batch also reconnects to saved jobs",
    )
    command.add_argument(
        "-m", "--model",
        choices=(GEMINI_MODEL,),
        default=GEMINI_MODEL,
        help=f"Gemini model name (pinned to {GEMINI_MODEL})",
    )
    command.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="medium",
        help="Reasoning effort level for models supporting thinking",
    )
    command.add_argument(
        "--timeout-s",
        type=positive_float,
        help="HTTP request timeout in seconds (default: 120; 900 for flex)",
    )
    command.add_argument(
        "--max-retry",
        type=nonnegative_int,
        help="Additional retries per transient failed request; 0 disables retries "
             "(default: 4; 11 for flex). Verifier rejections are never retried",
    )
    command.add_argument(
        "--inference-mode",
        choices=("batch", "flex", "standard"),
        default="standard",
        help=(
            "Gemini provider mode; standard uses synchronous generateContent. "
            "Batch and flex are opt-in cost/latency modes"
        ),
    )
    command.add_argument(
        "-b", "-bs", "--batch-size",
        type=positive_int,
        default=10,
        help="Maximum requests per Gemini Batch job (also capped below 20 MB)",
    )
    command.add_argument(
        "--batch-poll-interval-s",
        type=positive_float,
        default=10.0,
        help="Seconds between Batch status polls",
    )
    command.add_argument(
        "--batch-timeout-s",
        type=positive_float,
        default=86400.0,
        help="Maximum seconds to wait for Batch completion",
    )
    command.add_argument("--cache-prompt", action="store_true", help="Explicitly cache the system prompt in any inference mode (default: false)")
    command.add_argument(
        "--cache-ttl-s",
        type=positive_int,
        help="Cache lifetime in seconds (default: 3600; 90000 for batch)",
    )


def gemini_parameters(args: Any) -> dict[str, Any]:
    if args.continue_run and args.overwrite:
        raise ValueError("--continue and --overwrite cannot be combined")
    if args.timeout_s is None:
        args.timeout_s = 900.0 if args.inference_mode == "flex" else 120.0
    if args.max_retry is not None:
        args.max_retries = args.max_retry + 1
    else:
        args.max_retries = 12 if args.inference_mode == "flex" else 5
    if args.cache_ttl_s is None:
        args.cache_ttl_s = 90000 if args.inference_mode == "batch" else 3600
    if args.cache_prompt and args.inference_mode == "batch" and args.cache_ttl_s < 90000:
        raise ValueError("Batch prompt caching requires --cache-ttl-s >= 90000 (25 hours)")
    return {key: getattr(args, key) for key in (
        "model", "reasoning_effort", "max_tokens", "timeout_s", "max_retries",
        "inference_mode", "cache_prompt", "cache_ttl_s",
        "user_prompt", "max_response_retries",
    )}


def pending_agent_pairs(
    args: Any,
    pairs: list[tuple[Path, Path]],
    parameters: dict[str, Any],
) -> list[tuple[Path, Path]]:
    """Check raw Gemini pairs before any provider submission."""
    pending = []
    kept_other_settings = 0
    for source, destination in pairs:
        metadata = destination.with_suffix(".json")
        if destination == metadata:
            raise ValueError("Raw response --output-file cannot have a .json suffix")
        if not args.overwrite and (destination.exists() or metadata.exists()):
            if not metadata.exists() and args.continue_run:
                pending.append((source, destination))
                continue
            try:
                old = read_json(metadata)
            except (OSError, ValueError) as exc:
                if args.continue_run:
                    pending.append((source, destination))
                    continue
                raise ValueError(f"Unreadable output metadata: {metadata}; use --overwrite") from exc
            wanted = {
                "source": {"path": persist_path(source), "sha256": digest(source)},
                "operation": "explore_audio_model",
                "model": parameters["model"],
                "parameters": parameters,
            }
            old = old if isinstance(old, dict) else {}
            matches = all(old.get(key) == value for key, value in wanted.items())
            if not matches and not args.continue_run:
                raise ValueError(f"Conflicting output: {metadata}; use --continue or --overwrite")
            output = old.get("output") or {}
            if (old.get("status") != "fail" and destination.is_file()
                    and destination.read_text(encoding="utf-8").strip()
                    and output.get("sha256") == digest(destination)):
                # --continue keeps complete outputs made with other settings;
                # delete an output to regenerate it with the current ones.
                kept_other_settings += not matches
                continue
            if not args.continue_run:
                raise ValueError(f"Incomplete output pair: {destination}; use --continue or --overwrite")
        pending.append((source, destination))
    if kept_other_settings:
        logger.info("--continue kept %d complete output(s) made with other prompt/settings", kept_other_settings)
    return pending


def generation_callback(
    client: GeminiAgent,
    args: Any,
    pairs: list[tuple[Path, Path]],
    system_prompt: str,
    *,
    all_pairs: list[tuple[Path, Path]],
) -> Callable[[Path], dict[str, Any]]:
    """Select the provider API once; both commands consume raw results."""
    if pairs:
        # Shared by the agent and the verifier: a bad key/model now aborts the
        # run once, before any per-item failure artifacts are written.
        client.preflight()
    if client.inference_mode != "batch":
        return lambda source: client.generate(source, system_prompt=system_prompt)
    # Record the original submission subset separately from the full input identity.
    # Completed artifacts may shrink `pairs` on a later invocation.
    root = args.input_file if args.input_file is not None else args.input_dir
    signature = {
        "root": persist_path(root),
        "inputs": [
            {"source": persist_path(source), "sha256": digest(source), "output": persist_path(destination)}
            for source, destination in all_pairs
        ],
        "parameters": gemini_parameters(args),
        "system_prompt": system_prompt,
        "system_prompt_path": system_prompt_ref(),
        "batch_size": args.batch_size,
    }
    scope = {"root": signature["root"], "operation": args._operation}
    run_id = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()[:20]
    manifest_path = args.work_dir / "batch_jobs" / f"run_{run_id}.json"
    sources = [source for source, _ in pairs]
    manifest = read_json(manifest_path) if args.continue_run and manifest_path.exists() else {}
    saved = [resolve_stored_path(value) for value in manifest.get("sources", [])]
    if manifest.get("signature") == signature and all(source in saved for source in sources):
        # Resubmitting the saved subset reproduces its request keys, reconnecting saved jobs.
        sources = saved
    elif pairs:
        if manifest:
            logger.info("--continue: settings or missing outputs differ from the saved Batch run; "
                        "submitting the %d missing output(s) as a new Batch run", len(pairs))
        write_json(manifest_path, {
            "signature": signature,
            "sources": [persist_path(source) for source in sources],
        })
    if not pairs:
        return lambda source: {}  # No generation callback will be consumed.
    try:
        results = client.generate_batch(
            sources, system_prompt=system_prompt,
            batch_size=args.batch_size, poll_interval_s=args.batch_poll_interval_s,
            batch_timeout_s=args.batch_timeout_s, state_dir=args.work_dir / "batch_jobs",
            reuse_state=args.continue_run,
        )
    except BaseException:
        from _reporting import new_run_stats, report_cost_summary

        report_cost_summary("gemini/batch", new_run_stats(), client.get_cost_summary())
        raise

    def generate(source: Path) -> dict[str, Any]:
        value = results[source.resolve()]
        if isinstance(value, Exception):
            raise value
        return value

    return generate


def main() -> int:
    from artifacts import run_agent
    from _common.files import destinations, parser

    command = parser(
        "Explore Gemini audio understanding without imposing a response schema.",
        "s4-agent",
        "gemini",
    )
    command.add_argument(
        "-mt",
        "--max-tokens",
        type=positive_int,
        default=65536,
        help="Maximum output tokens",
    )
    add_gemini_arguments(command)
    args = command.parse_args()
    configure_gemini_paths(args, "s4-agent")

    system_prompt = load_system_prompt()
    client = GeminiAgent(**gemini_parameters(args))
    parameters = {
        **gemini_parameters(args),
        "system_prompt": system_prompt,
        "system_prompt_path": system_prompt_ref(),
    }
    pairs = destinations(args, "_gemini", ".txt")
    all_pairs = pairs
    pairs = pending_agent_pairs(args, pairs, parameters)
    generate = generation_callback(
        client,
        args,
        pairs,
        system_prompt,
        all_pairs=all_pairs,
    )

    result = run_agent(
        args=args,
        pairs=pairs,
        backend="gemini",
        parameters=parameters,
        generate=generate,
        cost_summary=client.get_cost_summary,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
