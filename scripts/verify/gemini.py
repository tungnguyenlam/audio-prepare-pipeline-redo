from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import base64
import json
import logging
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
    DEFAULT_ACOUSTIC_PROMPT,
    extract_json_payload,
)

logger = logging.getLogger("verifier.gemini")


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
        **kwargs: Any,
    ) -> None:
        del kwargs  # accept factory extras without failing
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable or parameter is required for GeminiVerifier."
            )
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self._lock = threading.Lock()
        self._usage_totals = empty_usage_totals()
        self._cost_totals = empty_cost_totals()
        self._cost_totals["model"] = model
        logger.info(
            "Initialized GeminiVerifier with model '%s' (reasoning_effort=%s).",
            model,
            reasoning_effort,
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

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
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
                    "parts": [
                        {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0,
                "maxOutputTokens": 2048,
            },
        }
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": self.reasoning_effort.upper()
            }

        req_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()
        res = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    break
            except urllib.error.HTTPError as exc:
                err_msg = exc.read().decode("utf-8", errors="replace")
                if exc.code in (429, 503) and attempt < self.max_retries:
                    sleep_s = (self.base_backoff_s * (2 ** (attempt - 1))) + random.uniform(0.1, 1.0)
                    logger.warning(
                        "Gemini HTTP %d (%s). Retrying in %.2fs (attempt %d/%d)...",
                        exc.code, exc.reason, sleep_s, attempt, self.max_retries
                    )
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Gemini API HTTP {exc.code} error: {err_msg}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    sleep_s = (self.base_backoff_s * (2 ** (attempt - 1))) + random.uniform(0.1, 0.5)
                    logger.warning(
                        "Gemini network timeout/error: %s. Retrying in %.2fs (attempt %d/%d)...",
                        exc, sleep_s, attempt, self.max_retries
                    )
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Gemini API connection error: {exc}") from exc

        latency = round(time.time() - t0, 3)
        if not res:
            raise RuntimeError("Gemini API call failed to return a response.")

        candidates = res.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates returned from Gemini API: {res}")

        raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
        parsed = extract_json_payload(raw_text)
        parsed["_latency_s"] = latency

        usage = normalize_gemini_usage(
            res.get("usageMetadata"),
            audio_duration_s=audio_duration_s,
        )
        cost = estimate_gemini_cost(self.model, usage)
        parsed["_usage"] = usage
        parsed["_cost"] = cost
        parsed["_model"] = self.model
        parsed["_reasoning_effort"] = self.reasoning_effort
        parsed["_engine"] = "gemini"
        if res.get("modelVersion"):
            parsed["_model_version"] = res["modelVersion"]

        with self._lock:
            accumulate_usage(self._usage_totals, usage)
            accumulate_cost(self._cost_totals, cost)
            running = round(float(self._cost_totals.get("total_usd", 0.0)), 6)

        cost_usd = float(cost["total_usd"]) if cost else None
        if cost_usd is None:
            logger.info(
                "Gemini %s usage prompt=%d out=%d think=%d running=$%.6f",
                audio_path.name,
                usage.get("prompt_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("thinking_tokens", 0),
                running,
            )
        else:
            logger.info(
                "Gemini %s -> %s | tokens p/o/t=%d/%d/%d | cost=$%.6f | session=$%.6f",
                audio_path.name,
                parsed.get("decision", "?"),
                usage.get("prompt_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("thinking_tokens", 0),
                cost_usd,
                running,
            )
        return parsed


def main() -> int:
    import argparse
    import contextlib
    from _common.files import batch, destinations, identity, parser, read_json, request, write_json

    p = parser('Verify audio with gemini; writes verdicts without filtering audio.', 'verify', 'gemini')
    p.add_argument('--prompt-file', type=Path, help='Path to prompt text file')
    p.add_argument('--model', type=str, default='gemini-3.8-flash', help='Gemini model (e.g. gemini-3.8-flash, gemini-3.5-flash-lite)')
    p.add_argument('--reasoning-effort', type=str, default='medium', choices=('low', 'medium', 'high', 'none'))
    args = p.parse_args()

    pairs = destinations(args, '_gemini', '.json')
    parameters = {key: getattr(args, key) for key in ('model', 'reasoning_effort')}

    if args.prompt_file and args.prompt_file.is_file():
        prompt = args.prompt_file.read_text(encoding='utf-8').strip()
    elif Path('prompts/acoustic_defect.txt').is_file():
        prompt = Path('prompts/acoustic_defect.txt').read_text(encoding='utf-8').strip()
    else:
        prompt = DEFAULT_ACOUSTIC_PROMPT

    with contextlib.redirect_stdout(sys.stderr):
        verifier = GeminiVerifier(**parameters)
    parameters['prompt'] = prompt

    for key in ('model', 'model_id', 'endpoint', 'gguf_variant', 'device'):
        if hasattr(verifier, key) and isinstance(getattr(verifier, key), (str, int, float, bool, type(None))):
            parameters[key] = getattr(verifier, key)

    def process(src: Path, dest: Path) -> None:
        wanted = request(identity(src), 'verify', parameters, 'gemini')
        if dest.exists() and not args.overwrite:
            old = read_json(dest)
            if all(old.get(k) == v for k, v in wanted.items()) and 'verdict' in old:
                return
            raise ValueError(f'Conflicting output: {dest}; use --overwrite')
        verdict = verifier.verify(src, prompt)
        if not isinstance(verdict, dict) or verdict.get('decision') not in {'pass', 'reject'}:
            raise ValueError('Verifier did not return a pass/reject decision')
        write_json(dest, {**wanted, 'verdict': verdict})

    return batch(pairs, process)


if __name__ == '__main__':
    raise SystemExit(main())
