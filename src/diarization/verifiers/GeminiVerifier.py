from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from src.diarization.gemini_pricing import (
    accumulate_cost,
    accumulate_usage,
    empty_cost_totals,
    empty_usage_totals,
    estimate_gemini_cost,
    normalize_gemini_usage,
)
from src.diarization.verifiers.BaseVerifier import (
    BaseVerifier,
    extract_json_payload,
)

logger = logging.getLogger("verifier.gemini")


class GeminiVerifier(BaseVerifier):
    """Verifier querying Google Gemini multimodal audio API.

    Each successful ``verify`` call attaches ``_usage`` and ``_cost`` to the
    returned verdict and accumulates session totals (thread-safe) for automatic
    cost logging. Thinking tokens are billed as output when present.
    """

    supports_concurrency: bool = True

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: str | None = None,
        reasoning_effort: str = "medium",
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

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        latency = round(time.time() - t0, 3)

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
        if res.get("modelVersion"):
            parsed["_model_version"] = res["modelVersion"]

        with self._lock:
            accumulate_usage(self._usage_totals, usage)
            accumulate_cost(self._cost_totals, cost)
            running = round(float(self._cost_totals.get("total_usd", 0.0)), 6)

        cost_usd = float(cost["total_usd"]) if cost else None
        if cost_usd is None:
            logger.info(
                "Gemini %s usage prompt=%d out=%d think=%d (no price card) running=$%.6f",
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
