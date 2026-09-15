from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import mimetypes
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common.files import ROOT, digest, safe_name, write_json  # noqa: E402
from _gemini_pricing import (  # noqa: E402
    accumulate_cost,
    accumulate_usage,
    empty_cost_totals,
    empty_usage_totals,
    estimate_gemini_cost,
    normalize_gemini_usage,
)

logger = logging.getLogger("agent.gemini")

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
BATCH_MAX_INLINE_BYTES = 18 * 1024 * 1024
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
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def audio_mime_type(path: Path) -> str:
    return (
        MIME_TYPES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def response_text(response: dict[str, Any]) -> str:
    """Concatenate the first candidate's text parts without altering them."""
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return ""
    content = candidates[0].get("content")
    if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
        return ""
    return "".join(
        part["text"]
        for part in content["parts"]
        if isinstance(part, dict) and isinstance(part.get("text"), str)
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
    """Raw generation client for Gemini standard and asynchronous Batch APIs."""

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: str | None = None,
        reasoning_effort: str = "medium",
        max_retries: int = 5,
        base_backoff_s: float = 2.0,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        timeout_s: float = 120.0,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable or .env setting is required for GeminiAgent."
            )
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._usage_totals = empty_usage_totals()
        self._cost_totals = empty_cost_totals()
        self._cost_totals["model"] = model
        self._session = None
        try:
            import requests

            self._session = requests.Session()
        except ImportError:
            self._session = None
        logger.info(
            "Initialized GeminiAgent with model '%s' (reasoning_effort=%s, max_tokens=%d, temp=%.2f, session_transport=%s).",
            model,
            reasoning_effort,
            max_tokens,
            temperature,
            "requests" if self._session is not None else "urllib",
        )

    def get_cost_summary(self) -> dict[str, Any]:
        """Return a copy of cumulative usage and estimated USD cost for this session."""
        with self._lock:
            usage = dict(self._usage_totals)
            cost = dict(self._cost_totals)
            for key in (
                "uncached_input_usd",
                "cached_input_usd",
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

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        retry_network_errors: bool = True,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        for attempt in range(1, self.max_retries + 1):
            if self._session is not None:
                try:
                    resp = self._session.request(
                        method,
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_s,
                    )
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                        sleep_s = self.base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0.1, 1.0)
                        logger.warning(
                            "Gemini HTTP %d (%s). Retrying in %.2fs (attempt %d/%d)...",
                            resp.status_code,
                            resp.reason,
                            sleep_s,
                            attempt,
                            self.max_retries,
                        )
                        time.sleep(sleep_s)
                        continue
                    if not resp.ok:
                        raise RuntimeError(f"Gemini API HTTP {resp.status_code} error: {resp.text}")
                    return resp.json(), {
                        key: value
                        for key, value in resp.headers.items()
                        if key.lower() in {"content-type", "date", "server", "x-request-id"}
                    }
                except Exception as exc:
                    if isinstance(exc, RuntimeError):
                        raise
                    err_type = type(exc).__name__
                    if "InvalidSchema" in err_type and "SOCKS" in str(exc):
                        raise RuntimeError(
                            f"SOCKS proxy is configured but PySocks is missing: {exc}. Install with: uv pip install pysocks"
                        ) from exc
                    if retry_network_errors and attempt < self.max_retries:
                        sleep_s = self.base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                        logger.warning(
                            "Gemini network timeout/error: %s. Retrying in %.2fs (attempt %d/%d)...",
                            exc,
                            sleep_s,
                            attempt,
                            self.max_retries,
                        )
                        time.sleep(sleep_s)
                        continue
                    raise RuntimeError(f"Gemini API connection error: {exc}") from exc

            req_body = json.dumps(payload).encode("utf-8") if payload is not None else None
            req = urllib.request.Request(
                url,
                data=req_body,
                headers=headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8")), {
                        key: value
                        for key, value in resp.headers.items()
                        if key.lower() in {"content-type", "date", "server", "x-request-id"}
                    }
            except urllib.error.HTTPError as exc:
                err_msg = exc.read().decode("utf-8", errors="replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    sleep_s = self.base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0.1, 1.0)
                    logger.warning(
                        "Gemini HTTP %d (%s). Retrying in %.2fs (attempt %d/%d)...",
                        exc.code,
                        exc.reason,
                        sleep_s,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Gemini API HTTP {exc.code} error: {err_msg}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if retry_network_errors and attempt < self.max_retries:
                    sleep_s = self.base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                    logger.warning(
                        "Gemini network timeout/error: %s. Retrying in %.2fs (attempt %d/%d)...",
                        exc,
                        sleep_s,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Gemini API connection error: {exc}") from exc
        raise RuntimeError("Gemini API call exhausted retries.")

    def _request_payload(
        self,
        audio_path: Path,
        prompt: str,
        *,
        system_prompt: str | None,
        json_response: bool,
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
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system_prompt is not None:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if json_response:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if self.top_p is not None:
            payload["generationConfig"]["topP"] = self.top_p
        if self.top_k is not None:
            payload["generationConfig"]["topK"] = self.top_k
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
        usage = normalize_gemini_usage(
            response.get("usageMetadata"),
            audio_duration_s=audio_duration_s,
        )
        cost = estimate_gemini_cost(
            self.model,
            usage,
            pricing_tier=pricing_tier,
        )
        with self._lock:
            accumulate_usage(self._usage_totals, usage)
            accumulate_cost(self._cost_totals, cost)
        result = {
            "text": response_text(response),
            "latency_s": round(latency_s, 3),
            "headers": headers or {},
            "usage": usage,
            "cost": cost,
            "inference_mode": "batch" if pricing_tier == "paid_batch" else "standard",
            "provider_body": response,
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
        json_response: bool = False,
    ) -> dict[str, Any]:
        payload, duration = self._request_payload(
            audio_path,
            prompt,
            system_prompt=system_prompt,
            json_response=json_response,
        )
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
            pricing_tier="paid_standard",
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
        json_response: bool = False,
        batch_size: int = 100,
        poll_interval_s: float = 10.0,
        batch_timeout_s: float = 86400.0,
        state_dir: Path,
        reuse_state: bool = True,
    ) -> dict[Path, dict[str, Any] | Exception]:
        """Submit missing audio requests to Gemini Batch and wait for results."""
        descriptors: list[dict[str, Any]] = []
        for index, source_value in enumerate(audio_paths):
            source = Path(source_value).resolve()
            payload, duration = self._request_payload(
                source,
                prompt,
                system_prompt=system_prompt,
                json_response=json_response,
            )
            key_material = {
                "index": index,
                "path": str(source),
                "sha256": digest(source),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
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
        if reuse_state and state_path.is_file():
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                saved = {}
        if saved.get("session") != session_material or saved.get("status") == "failed":
            if state_path.exists():
                state_path = state_dir / f"batch_{session_id}_{int(time.time())}.json"
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
                if not isinstance(existing, dict) or existing.get("request_keys") != keys or not existing.get("job_name"):
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
                        f"Gemini Batch did not finish within {batch_timeout_s:g}s; resume with the same command. State: {state_path}"
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


def main() -> int:
    from artifacts import add_prompt_arguments, load_prompts, run_agent
    from _common.files import destinations, parser, positive_int

    command = parser(
        "Explore Gemini audio understanding without imposing a response schema.",
        "agent",
        "gemini",
    )
    add_prompt_arguments(command, top_p=True)
    command.add_argument("--model", default="gemini-3.8-flash", help="Gemini model name")
    command.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="medium",
        help="Reasoning effort level for models supporting thinking",
    )
    command.add_argument("--top-k", type=positive_int, help="Top-k sampling parameter")
    command.add_argument("--timeout-s", type=positive_float, default=120.0, help="HTTP request timeout in seconds")
    command.add_argument("--max-retries", type=positive_int, default=5, help="Maximum retry attempts per request")
    command.add_argument(
        "--inference-mode",
        choices=("batch", "standard"),
        default="batch",
        help="Gemini provider mode; batch is asynchronous and billed at Batch rates",
    )
    command.add_argument(
        "--batch-size",
        type=positive_int,
        default=10,
        help="Maximum requests per Gemini Batch job (also capped below 20 MB)",
    )
    command.add_argument("--batch-poll-interval-s", type=positive_float, default=10.0, help="Seconds between Batch status polls")
    command.add_argument("--batch-timeout-s", type=positive_float, default=86400.0, help="Maximum seconds to wait for Batch completion")
    args = command.parse_args()
    configure_gemini_paths(args, "agent")

    prompt, system_prompt = load_prompts(args)
    parameters = {
        "model": args.model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "inference_mode": args.inference_mode,
    }
    client = GeminiAgent(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )
    pairs = destinations(args, "_gemini", ".txt")
    if args.inference_mode == "batch":
        generated = client.generate_batch(
            [source for source, _ in pairs],
            prompt,
            system_prompt=system_prompt,
            batch_size=args.batch_size,
            poll_interval_s=args.batch_poll_interval_s,
            batch_timeout_s=args.batch_timeout_s,
            state_dir=args.work_dir / "batch_jobs",
            reuse_state=not args.overwrite,
        )

        def generate(source: Path) -> dict[str, Any]:
            value = generated[source.resolve()]
            if isinstance(value, Exception):
                raise value
            return value
    else:
        generate = lambda source: client.generate(source, prompt, system_prompt=system_prompt)

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
