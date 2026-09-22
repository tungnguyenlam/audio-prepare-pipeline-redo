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
from pathlib import Path
from collections.abc import Callable
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common.files import ROOT, digest, persist_path, safe_name, write_json, positive_int, read_json, resolve_stored_path  # noqa: E402
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

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-3.8-flash"
BATCH_MAX_INLINE_BYTES = 18 * 1024 * 1024
MAX_BACKOFF_S = 60.0
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


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def audio_mime_type(path: Path) -> str:
    return (
        MIME_TYPES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


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


def _load_repo_env() -> None:
    """Load environment variables from .env in the repository root if available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    env_file = Path(__file__).resolve().parents[2] / ".env"
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


_load_repo_env()


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
        base_backoff_s: float = 2.0,
        max_tokens: int = 65536,
        timeout_s: float | None = None,
        inference_mode: str = "standard",
        cache_prompt: bool = False,
        cache_ttl_s: int | None = None,
    ) -> None:
        if model != GEMINI_MODEL:
            raise ValueError(
                f"This pipeline is pinned to {GEMINI_MODEL!r}; received {model!r}."
            )
        if inference_mode not in ("standard", "flex", "batch"):
            raise ValueError(f"Unsupported Gemini inference mode: {inference_mode}")
        self.model = model
        self.inference_mode = inference_mode
        self.cache_prompt = cache_prompt
        self.cache_ttl_s = cache_ttl_s or (90000 if inference_mode == "batch" else 3600)
        self._cache_lock = threading.Lock()
        self._cache = None
        self._cache_key = None
        self._cache_expires = 0.0
        self._pending_storage_usd = 0.0
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable or .env setting is required for GeminiAgent."
            )
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries if max_retries is not None else (12 if inference_mode == "flex" else 5)
        self.base_backoff_s = base_backoff_s
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s if timeout_s is not None else (900.0 if inference_mode == "flex" else 120.0)
        self._lock = threading.Lock()
        self._usage_totals = empty_usage_totals()
        self._cost_totals = empty_cost_totals()
        self._cost_totals["model"] = model
        import httpx

        self._session = httpx.Client(timeout=self.timeout_s)

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

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        retry_network_errors: bool = True,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        import httpx

        headers = {"x-goog-api-key": self.api_key}
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.request(method, url, json=payload, headers=headers)
            except httpx.TransportError:
                if not retry_network_errors or attempt == self.max_retries:
                    raise RuntimeError("Gemini connection failed") from None
            else:
                if response.is_success:
                    return response.json(), {
                        key: value for key, value in response.headers.items()
                        if key in {"content-type", "date", "server", "x-request-id"}
                    }
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    # Do not echo provider bodies or transport exceptions: they may contain secrets.
                    operation = "prompt cache creation" if url.endswith("/cachedContents") else "request"
                    hint = "; check model caching support and minimum prompt size" if operation == "prompt cache creation" else ""
                    raise RuntimeError(f"Gemini {operation}: HTTP {response.status_code}{hint}")
            delay = self._backoff_s(attempt) + random.uniform(0.1, 0.5)
            logger.warning("Gemini request failed; retry %d/%d in %.2fs", attempt, self.max_retries, delay)
            time.sleep(delay)
        raise RuntimeError("Gemini API call exhausted retries")

    def _apply_prompt_cache(self, payload: dict[str, Any], prompt: str, system_prompt: str | None) -> None:
        """Reuse one text-only cache per client; audio always stays in the request."""
        if not self.cache_prompt:
            return
        key = (prompt, system_prompt)
        with self._cache_lock:
            if key != self._cache_key or time.monotonic() >= self._cache_expires:
                body = {
                    "model": f"models/{self.model}",
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "ttl": f"{self.cache_ttl_s}s",
                }
                if system_prompt is not None:
                    body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
                started = time.monotonic()
                cache, _ = self._request_json("POST", f"{API_ROOT}/cachedContents", body, retry_network_errors=False)
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
            payload["cachedContent"] = self._cache
        payload["contents"][0]["parts"] = [
            part for part in payload["contents"][0]["parts"] if "text" not in part
        ]
        payload.pop("systemInstruction", None)

    def _request_payload(
        self,
        audio_path: Path,
        prompt: str,
        *,
        system_prompt: str | None,
    ) -> tuple[dict[str, Any], float | None]:
        audio_path = Path(audio_path)
        with audio_path.open("rb") as stream:
            audio_b64 = base64.b64encode(stream.read()).decode("ascii")
        audio_duration_s: float | None = None
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
            if info.samplerate and info.frames:
                audio_duration_s = float(info.frames) / float(info.samplerate)
        except Exception:
            pass

        payload: dict[str, Any] = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": audio_mime_type(audio_path), "data": audio_b64}},
                ],
            }],
            # Gemini 3.x is tuned for its default sampler. Do not set temperature,
            # topP, or topK here: Google explicitly recommends omitting them.
            "generationConfig": {
                "candidateCount": 1,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system_prompt is not None:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": self.reasoning_effort.upper()
            }
        return payload, audio_duration_s

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
            explicit_text_cache=self.cache_prompt,
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
        if batch_job is not None:
            result["batch_job"] = batch_job
        if batch_request_key is not None:
            result["batch_request_key"] = batch_request_key
        return result

    def generate(
        self,
        audio_path: Path,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if self.inference_mode == "batch":
            raise ValueError("Batch mode requires generate_batch(), not generate()")
        payload, duration = self._request_payload(
            audio_path,
            prompt,
            system_prompt=system_prompt,
        )
        self._apply_prompt_cache(payload, prompt, system_prompt)
        # AI Studio's normal Run action uses standard generateContent and does
        # not add a service-tier override. Omit it for request parity.
        if self.inference_mode != "standard":
            payload["serviceTier"] = self.inference_mode
        started = time.monotonic()
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
        prompt: str,
        *,
        system_prompt: str | None = None,
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
                prompt,
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
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
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
                    or existing.get("state") in BATCH_TERMINAL_STATES - {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}
                ):
                    previous_storage = self._cost_totals["cache_storage_usd"]
                    for item in group:
                        self._apply_prompt_cache(item["entry"]["request"], prompt, system_prompt)
                    saved["cache_storage_usd"] = saved.get("cache_storage_usd", 0.0) + (
                        self._cost_totals["cache_storage_usd"] - previous_storage
                    )
                    write_json(state_path, saved)
                    body = {
                        "batch": {
                            "display_name": f"{safe_name(self.model)}-{safe_name(self.reasoning_effort)}-{session_id}-{group_index + 1}",
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
                    job, _ = self._request_json("GET", job_url)
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
                        f"Gemini Batch did not finish within {batch_timeout_s:g}s; resume with the same command plus --continue. State: {state_path}"
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
            results[descriptor["source"]] = self._generation_result(
                response,
                latency_s=batch_latency,
                audio_duration_s=descriptor["duration"],
                pricing_tier="paid_batch",
                batch_job=job_by_key.get(key),
                batch_request_key=key,
            )
        return results


def add_gemini_arguments(command: argparse.ArgumentParser) -> None:
    """Identical provider controls for raw generation and verification."""
    command.add_argument(
        "--continue", dest="continue_run", action="store_true",
        help="Skip matching completed outputs and retry unfinished work; Batch also reconnects to saved jobs",
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
        "--max-retries",
        type=positive_int,
        help="Maximum retry attempts per request (default: 5; 12 for flex, which returns 503 when capacity is short)",
    )
    command.add_argument(
        "--inference-mode",
        choices=("batch", "flex", "standard"),
        default="standard",
        help=(
            "Gemini provider mode; standard matches a normal Google AI Studio Run. "
            "Batch and flex are opt-in cost/latency modes"
        ),
    )
    command.add_argument(
        "-b", "-bs", "--batch-size",
        type=positive_int,
        default=10,
        help="Maximum requests per Gemini Batch job (also capped below 20 MB)",
    )
    command.add_argument("--batch-poll-interval-s", type=positive_float, default=10.0, help="Seconds between Batch status polls")
    command.add_argument("--batch-timeout-s", type=positive_float, default=86400.0, help="Maximum seconds to wait for Batch completion")
    command.add_argument("--cache-prompt", action="store_true", help="Explicitly cache prompt text (default: false)")
    command.add_argument("--cache-ttl-s", type=positive_int, help="Cache lifetime in seconds (default: 3600; 90000 for batch)")


def gemini_parameters(args: Any) -> dict[str, Any]:
    if args.continue_run and args.overwrite:
        raise ValueError("--continue and --overwrite cannot be combined")
    if args.timeout_s is None:
        args.timeout_s = 900.0 if args.inference_mode == "flex" else 120.0
    if args.max_retries is None:
        args.max_retries = 12 if args.inference_mode == "flex" else 5
    if args.cache_ttl_s is None:
        args.cache_ttl_s = 90000 if args.inference_mode == "batch" else 3600
    if args.cache_prompt and args.inference_mode == "batch" and args.cache_ttl_s < 90000:
        raise ValueError("Batch prompt caching requires --cache-ttl-s >= 90000 (25 hours)")
    return {key: getattr(args, key) for key in (
        "model", "reasoning_effort", "max_tokens", "timeout_s", "max_retries",
        "inference_mode", "cache_prompt", "cache_ttl_s",
    )}


def pending_agent_pairs(args: Any, pairs: list[tuple[Path, Path]], parameters: dict[str, Any]) -> list[tuple[Path, Path]]:
    """Check raw Gemini pairs before any provider submission."""
    pending = []
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
                raise ValueError(f"Unreadable output metadata: {metadata}; use --overwrite") from exc
            wanted = {
                "source": {"path": persist_path(source), "sha256": digest(source)},
                "operation": "explore_audio_model",
                "model": parameters["model"],
                "parameters": parameters,
            }
            if not isinstance(old, dict) or any(old.get(key) != value for key, value in wanted.items()):
                raise ValueError(f"Conflicting output: {metadata}; use --overwrite")
            output = old.get("output") or {}
            if destination.is_file() and output.get("sha256") == digest(destination):
                continue
            if not args.continue_run:
                raise ValueError(f"Incomplete output pair: {destination}; use --continue or --overwrite")
        pending.append((source, destination))
    return pending


def generation_callback(
    client: GeminiAgent,
    args: Any,
    pairs: list[tuple[Path, Path]],
    prompt: str,
    system_prompt: str | None = None,
    *,
    all_pairs: list[tuple[Path, Path]],
) -> Callable[[Path], dict[str, Any]]:
    """Select the provider API once; both commands consume raw results."""
    if client.inference_mode != "batch":
        return lambda source: client.generate(source, prompt, system_prompt=system_prompt)
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
        "prompt": prompt,
        "system_prompt": system_prompt,
        "batch_size": args.batch_size,
    }
    scope = {"root": signature["root"], "operation": args._operation}
    run_id = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()[:20]
    manifest_path = args.work_dir / "batch_jobs" / f"run_{run_id}.json"
    sources = [source for source, _ in pairs]
    if args.continue_run and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("signature") != signature:
            raise ValueError(f"Batch inputs/settings changed: {manifest_path}; use --overwrite")
        sources = [resolve_stored_path(value) for value in manifest["sources"]]
        if any(source not in sources for source, _ in pairs):
            raise ValueError("Missing outputs outside the saved Batch submission; rerun without --continue")
    elif pairs:
        write_json(manifest_path, {
            "signature": signature,
            "sources": [persist_path(source) for source in sources],
        })
    if not pairs:
        return lambda source: {}  # No generation callback will be consumed.
    try:
        results = client.generate_batch(
            sources, prompt, system_prompt=system_prompt,
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
    from artifacts import add_prompt_arguments, load_prompts, run_agent
    from _common.files import destinations, parser

    command = parser(
        "Explore Gemini audio understanding without imposing a response schema.",
        "s4-agent",
        "gemini",
    )
    add_prompt_arguments(command, max_tokens=65536, sampling=False)
    add_gemini_arguments(command)
    args = command.parse_args()
    configure_gemini_paths(args, "s4-agent")

    prompt, system_prompt = load_prompts(args)
    client = GeminiAgent(**gemini_parameters(args))
    parameters = {**gemini_parameters(args), "prompt": prompt, "system_prompt": system_prompt}
    pairs = destinations(args, "_gemini", ".txt")
    all_pairs = pairs
    pairs = pending_agent_pairs(args, pairs, parameters)
    generate = generation_callback(client, args, pairs, prompt, system_prompt, all_pairs=all_pairs)

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