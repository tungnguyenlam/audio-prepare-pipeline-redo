from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.diarization.verifiers.BaseVerifier import extract_json_payload
from src.diarization.verifiers.EndpointVerifier import (
    EndpointVerifier,
    send_http_request,
)

logger = logging.getLogger("verifier.unsloth")

DEFAULT_UNSLOTH_HOST = "localhost"
DEFAULT_UNSLOTH_PORT = 8888
DEFAULT_UNSLOTH_MODEL = "unsloth/gemma-4-12b-it-GGUF"


def _derive_models_url(endpoint: str) -> str:
    """Derive the OpenAI-compatible /v1/models URL from a chat completions endpoint."""
    url = endpoint.rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")] + "/models"
    if url.endswith("/v1"):
        return f"{url}/models"
    return f"{url}/models"


class UnslothVerifier(EndpointVerifier):
    """Specialized verifier for Unsloth Studio / Unsloth inference servers.

    Inherits from EndpointVerifier and provides Unsloth-optimized capabilities:
    - Default endpoint auto-configured from UNSLOTH_ENDPOINT, or
      http://{UNSLOTH_HOST:localhost}:{UNSLOTH_PORT:8888}/v1/chat/completions.
    - Default model auto-configured from UNSLOTH_MODEL, probed from /v1/models,
      or 'unsloth/gemma-4-12b-it-GGUF'.
    - Authentication via UNSLOTH_API_KEY (falling back to OPENAI_API_KEY).
    - Multi-payload format support with automatic fallback:
        1. Standard OpenAI multimodal (`type: input_audio`)
        2. Unsloth top-level `audio_base64`
    - Thinking/reasoning model support (e.g. MOSS-Audio-8B-Thinking, Gemma-4).
    - Proxy-safe transport for local connections (preventing socks5h issues).
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        auto_probe_model: bool = True,
        payload_mode: str = "hybrid",
        **kwargs: Any,
    ) -> None:
        # 1. Resolve endpoint
        resolved_endpoint = endpoint
        if not resolved_endpoint or resolved_endpoint in {"default", "http://localhost:8000/v1/chat/completions"}:
            env_ep = os.getenv("UNSLOTH_ENDPOINT")
            if env_ep:
                resolved_endpoint = env_ep
            else:
                host = os.getenv("UNSLOTH_HOST", DEFAULT_UNSLOTH_HOST).strip() or DEFAULT_UNSLOTH_HOST
                port = os.getenv("UNSLOTH_PORT", str(DEFAULT_UNSLOTH_PORT)).strip() or str(DEFAULT_UNSLOTH_PORT)
                resolved_endpoint = f"http://{host}:{port}/v1/chat/completions"

        # 2. Resolve API key
        resolved_api_key = api_key or os.getenv("UNSLOTH_API_KEY") or os.getenv("OPENAI_API_KEY")

        # 3. Resolve model
        resolved_model = model
        if not resolved_model or resolved_model in {"default", "google/gemma-4-E2B-it"}:
            env_model = os.getenv("UNSLOTH_MODEL")
            if env_model:
                resolved_model = env_model
            elif auto_probe_model:
                probed = self._probe_loaded_model(resolved_endpoint, resolved_api_key)
                if probed:
                    resolved_model = probed
            if not resolved_model or resolved_model == "default":
                resolved_model = DEFAULT_UNSLOTH_MODEL

        super().__init__(
            endpoint=resolved_endpoint,
            model=resolved_model,
            api_key=resolved_api_key,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.payload_mode = payload_mode
        logger.info(
            "Initialized UnslothVerifier targeting '%s' (model='%s', api_key_configured=%s, mode=%s).",
            self.endpoint,
            self.model,
            bool(self.api_key),
            self.payload_mode,
        )

    @classmethod
    def _probe_loaded_model(cls, endpoint: str, api_key: str | None = None, timeout_s: float = 3.0) -> str | None:
        """Probe /v1/models on the Unsloth endpoint to discover loaded model."""
        models_url = _derive_models_url(endpoint)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            res = send_http_request(models_url, method="GET", headers=headers, timeout_s=timeout_s)
            data = res.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and first.get("id"):
                    return str(first["id"])
            if res.get("id"):
                return str(res["id"])
        except Exception:
            pass
        return None

    def check_ready(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """Check if Unsloth endpoint is reachable and responsive."""
        models_url = _derive_models_url(self.endpoint)
        headers = self._get_headers()
        try:
            res = send_http_request(models_url, method="GET", headers=headers, timeout_s=timeout_s)
            models: list[str] = []
            data = res.get("data")
            if isinstance(data, list):
                for m in data:
                    if isinstance(m, dict) and m.get("id"):
                        models.append(str(m["id"]))
            elif res.get("id"):
                models.append(str(res["id"]))
            return {
                "ready": True,
                "endpoint": self.endpoint,
                "models": models,
                "active_model": self.model,
                "message": f"Unsloth endpoint is ready ({len(models)} model(s) detected).",
            }
        except Exception as exc:
            return {
                "ready": False,
                "endpoint": self.endpoint,
                "models": [],
                "active_model": self.model,
                "message": f"Unsloth endpoint unreachable at {self.endpoint}: {exc}",
            }

    def _build_payload(self, audio_b64: str, prompt: str, mode: str) -> dict[str, Any]:
        """Build request payload according to the configured mode."""
        if mode == "audio_base64":
            return {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "audio_base64": audio_b64,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        elif mode == "standard":
            return {
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
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        else:  # hybrid
            return {
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
                "audio_base64": audio_b64,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        t0 = time.time()
        payload = self._build_payload(audio_b64, prompt, mode=self.payload_mode)

        try:
            res = self._post_json(payload)
        except Exception as exc:
            err_str = str(exc).lower()
            # If server rejected root audio_base64, try standard format
            if self.payload_mode == "hybrid" and (
                "extra fields" in err_str
                or "audio_base64" in err_str
                or "unknown field" in err_str
                or "400" in err_str
                or "422" in err_str
            ):
                logger.debug("Server rejected root audio_base64, falling back to standard format.")
                self.payload_mode = "standard"
                payload = self._build_payload(audio_b64, prompt, mode="standard")
                res = self._post_json(payload)
            # If server rejected content array, try audio_base64 format
            elif (
                "input_audio" in err_str
                or "content" in err_str
                or "str type expected" in err_str
            ) and self.payload_mode != "audio_base64":
                logger.debug("Server rejected content array, falling back to audio_base64 format.")
                self.payload_mode = "audio_base64"
                payload = self._build_payload(audio_b64, prompt, mode="audio_base64")
                res = self._post_json(payload)
            else:
                raise

        latency = round(time.time() - t0, 3)

        choices = res.get("choices", [])
        if not choices:
            raise RuntimeError(f"Unsloth endpoint returned no choices: {res}")

        msg = choices[0].get("message", {})
        raw_content = msg.get("content")
        reasoning = msg.get("reasoning_content") or ""

        target_text = ""
        if isinstance(raw_content, str) and raw_content.strip():
            target_text = raw_content.strip()
        elif isinstance(reasoning, str) and reasoning.strip():
            target_text = reasoning.strip()
        elif isinstance(raw_content, list):
            parts = [p.get("text", "") for p in raw_content if isinstance(p, dict)]
            target_text = "".join(parts).strip()

        if not target_text:
            raise RuntimeError(f"Unsloth endpoint returned empty content and reasoning: {msg}")

        parsed = extract_json_payload(target_text)
        parsed["_latency_s"] = latency
        if reasoning:
            parsed["_reasoning"] = reasoning
        return parsed
