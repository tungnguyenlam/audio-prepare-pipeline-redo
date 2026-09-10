from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import base64
import concurrent.futures
import contextlib
import json
import logging
import mimetypes
import os
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from _gemini_pricing import (
    accumulate_cost,
    accumulate_usage,
    empty_cost_totals,
    empty_usage_totals,
    estimate_gemini_cost,
    normalize_gemini_usage,
)
from _audio import (
    extract_json_payload,
)

logger = logging.getLogger("verifier.gemini")

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


def audio_mime_type(path: Path) -> str:
    return MIME_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


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


class GeminiVerifier:
    """Verifier querying Google Gemini multimodal audio API.

    Each successful ``verify`` call attaches ``_usage`` and ``_cost`` to the
    returned verdict and accumulates session totals (thread-safe) for automatic
    cost logging. Thinking tokens are billed as output when present.
    """

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
        del kwargs  # accept factory extras without failing
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable or .env setting is required for GeminiVerifier."
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
            "Initialized GeminiVerifier with model '%s' (reasoning_effort=%s, max_tokens=%d, temp=%.2f, session_transport=%s).",
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
            for key in ("input_usd", "output_usd", "total_usd"):
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

    def generate(
        self,
        audio_path: Path,
        prompt: str,
        *,
        system_prompt: str | None = None,
        json_response: bool = False,
    ) -> dict[str, Any]:
        audio_path = Path(audio_path)
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        audio_duration_s: float | None = None
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
            if info.samplerate and info.frames:
                audio_duration_s = float(info.frames) / float(info.samplerate)
        except Exception:
            audio_duration_s = None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": audio_mime_type(audio_path), "data": audio_b64}},
                    ]
                }
            ],
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

        t0 = time.time()
        res = None
        response_headers: dict[str, str] = {}
        for attempt in range(1, self.max_retries + 1):
            if self._session is not None:
                try:
                    resp = self._session.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=self.timeout_s,
                    )
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                        sleep_s = (self.base_backoff_s * (2 ** (attempt - 1))) + random.uniform(0.1, 1.0)
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
                    res = resp.json()
                    response_headers = {
                        key: value
                        for key, value in resp.headers.items()
                        if key.lower() in {"content-type", "date", "server", "x-request-id"}
                    }
                    break
                except Exception as exc:
                    if isinstance(exc, RuntimeError):
                        raise
                    err_type = type(exc).__name__
                    if "InvalidSchema" in err_type and "SOCKS" in str(exc):
                        raise RuntimeError(
                            f"SOCKS proxy is configured but PySocks is missing: {exc}. "
                            "Install with: uv pip install pysocks"
                        ) from exc
                    if attempt < self.max_retries:
                        sleep_s = (self.base_backoff_s * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
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
            else:
                # Fallback to urllib.request
                # Always create a new Request per attempt to prevent urllib from mutating
                # req.type into 'socks5h' across retries.
                req_body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=req_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        response_headers = {
                            key: value
                            for key, value in resp.headers.items()
                            if key.lower() in {"content-type", "date", "server", "x-request-id"}
                        }
                        break
                except urllib.error.HTTPError as exc:
                    err_msg = exc.read().decode("utf-8", errors="replace")
                    if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                        sleep_s = (self.base_backoff_s * (2 ** (attempt - 1))) + random.uniform(0.1, 1.0)
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
                    if attempt < self.max_retries:
                        sleep_s = (self.base_backoff_s * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
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

        latency = round(time.time() - t0, 3)
        if not res:
            raise RuntimeError("Gemini API call failed to return a response.")

        usage = normalize_gemini_usage(
            res.get("usageMetadata"),
            audio_duration_s=audio_duration_s,
        )
        cost = estimate_gemini_cost(self.model, usage)
        with self._lock:
            accumulate_usage(self._usage_totals, usage)
            accumulate_cost(self._cost_totals, cost)
        return {
            "text": response_text(res),
            "latency_s": latency,
            "headers": response_headers,
            "usage": usage,
            "cost": cost,
            "provider_body": res,
        }

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        generated = self.generate(audio_path, prompt, json_response=True)
        raw_text = generated["text"].strip()
        if not raw_text:
            raise RuntimeError(f"No text returned from Gemini API: {generated['provider_body']}")
        parsed = extract_json_payload(raw_text)
        parsed["_latency_s"] = generated["latency_s"]
        parsed["_usage"] = generated["usage"]
        parsed["_cost"] = generated["cost"]
        parsed["_model"] = self.model
        parsed["_reasoning_effort"] = self.reasoning_effort
        parsed["_engine"] = "gemini"
        provider_body = generated["provider_body"]
        if provider_body.get("modelVersion"):
            parsed["_model_version"] = provider_body["modelVersion"]

        with self._lock:
            running = round(float(self._cost_totals.get("total_usd", 0.0)), 6)
        cost = generated["cost"]
        if cost:
            logger.info(
                "Gemini %s -> %s | tokens p/o/t=%d/%d/%d | cost=$%.6f | session=$%.6f",
                Path(audio_path).name,
                parsed.get("decision", "?"),
                generated["usage"].get("prompt_tokens", 0),
                generated["usage"].get("output_tokens", 0),
                generated["usage"].get("thinking_tokens", 0),
                float(cost["total_usd"]),
                running,
            )
        else:
            logger.info(
                "Gemini %s -> %s | tokens p/o/t=%d/%d/%d | unpriced",
                Path(audio_path).name,
                parsed.get("decision", "?"),
                generated["usage"].get("prompt_tokens", 0),
                generated["usage"].get("output_tokens", 0),
                generated["usage"].get("thinking_tokens", 0),
            )
        return parsed


def batch_concurrent(
    pairs: list[tuple[Path, Path]],
    process,
    concurrency: int = 1,
) -> int:
    from _common.files import batch, progress

    if concurrency <= 1 or len(pairs) <= 1:
        return batch(pairs, process)

    total = len(pairs)
    failed = 0
    completed = 0
    lock = threading.Lock()
    t_start = time.perf_counter()
    progress("BATCH", f"Starting concurrent batch processing of {total} item(s) (concurrency={concurrency})")

    def task(item_idx: int, src: Path, dest: Path) -> tuple[bool, Path, str]:
        item_start = time.perf_counter()
        with lock:
            progress("ITEM_START", f"{src.name} -> {dest.name}", current=item_idx, total=total)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                process(src, dest)
            elapsed = time.perf_counter() - item_start
            return True, dest, f"{src.name} ({elapsed:.2f}s)"
        except Exception as exc:
            elapsed = time.perf_counter() - item_start
            return False, dest, f"{src.name}: {exc} ({elapsed:.2f}s)"

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(task, idx, src, dest) for idx, (src, dest) in enumerate(pairs, 1)]
        for future in concurrent.futures.as_completed(futures):
            success, dest, msg = future.result()
            with lock:
                completed += 1
                if success:
                    progress("ITEM_DONE", msg, current=completed, total=total)
                    print(dest, flush=True)
                else:
                    failed += 1
                    progress("ITEM_FAIL", msg, current=completed, total=total)

    total_elapsed = time.perf_counter() - t_start
    progress("BATCH_COMPLETE", f"{total - failed} succeeded; {failed} failed", elapsed_s=total_elapsed)
    return int(failed > 0)


def main() -> int:
    from _cli import load_prompt, resolved_parameters, verdict_processor
    from _common.files import batch, destinations, parser, positive_int

    p = parser('Verify audio with gemini; writes verdicts without filtering audio.', 'verify', 'gemini')
    p.add_argument('--prompt-file', type=Path, help='Path to prompt text file')
    p.add_argument('--model', type=str, default='gemini-3.8-flash', help='Gemini model (e.g. gemini-3.8-flash, gemini-3.5-flash-lite)')
    p.add_argument('--reasoning-effort', type=str, default='medium', choices=('low', 'medium', 'high', 'none'), help='Reasoning effort level for models supporting thinking')
    p.add_argument('--max-tokens', type=positive_int, default=2048, help='Maximum output tokens')
    p.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    p.add_argument('--top-p', type=float, default=None, help='Nucleus sampling top_p (optional)')
    p.add_argument('--top-k', type=positive_int, default=None, help='Top-k sampling (optional)')
    p.add_argument('--timeout-s', type=float, default=120.0, help='Timeout in seconds per API request')
    p.add_argument('--max-retries', type=positive_int, default=5, help='Maximum retry attempts per request')
    args = p.parse_args()

    pairs = destinations(args, '_gemini', '.json')
    init_kwargs = {
        'model': args.model,
        'reasoning_effort': args.reasoning_effort,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'timeout_s': args.timeout_s,
        'max_retries': args.max_retries,
    }

    prompt = load_prompt(args.prompt_file)

    with contextlib.redirect_stdout(sys.stderr):
        verifier = GeminiVerifier(**init_kwargs)

    parameters = {
        'model': verifier.model,
        'reasoning_effort': verifier.reasoning_effort,
        'prompt': prompt,
        'max_tokens': verifier.max_tokens,
        'temperature': verifier.temperature,
    }
    if verifier.top_p is not None:
        parameters['top_p'] = verifier.top_p
    if verifier.top_k is not None:
        parameters['top_k'] = verifier.top_k
    parameters = resolved_parameters(parameters, verifier)
    process = verdict_processor(
        args=args,
        backend='gemini',
        parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt),
    )

    res = batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)
    cost_summary = verifier.get_cost_summary()
    logger.info(
        "Session summary: %d input tokens, %d output tokens, %d think tokens | Total cost: $%.6f",
        cost_summary["usage"].get("prompt_tokens", 0),
        cost_summary["usage"].get("output_tokens", 0),
        cost_summary["usage"].get("thinking_tokens", 0),
        cost_summary["cost"].get("total_usd", 0.0),
    )
    return res


if __name__ == '__main__':
    raise SystemExit(main())
