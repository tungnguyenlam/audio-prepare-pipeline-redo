from __future__ import annotations

import base64
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

from src.diarization.verifiers.BaseVerifier import (
    BaseVerifier,
    extract_json_payload,
)

logger = logging.getLogger("verifier.endpoint")


class EndpointVerifier(BaseVerifier):
    """Verifier querying OpenAI / vLLM compatible multimodal audio endpoints."""

    supports_concurrency: bool = True

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1/chat/completions",
        model: str = "default",
        **kwargs: Any,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        logger.info("Initialized EndpointVerifier targeting '%s' (model='%s').", endpoint, model)

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        t0 = time.time()
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        latency = round(time.time() - t0, 3)

        raw_text = res["choices"][0]["message"]["content"]
        parsed = extract_json_payload(raw_text)
        parsed["_latency_s"] = latency
        return parsed
