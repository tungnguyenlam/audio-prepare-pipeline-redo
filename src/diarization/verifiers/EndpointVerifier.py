from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any

from src.diarization.verifiers.BaseVerifier import (
    BaseVerifier,
    extract_json_payload,
)

logger = logging.getLogger("verifier.endpoint")


def should_bypass_proxy(url: str) -> bool:
    """Check if the given URL host should bypass HTTP/SOCKS proxies.

    Handles localhost, 127.0.0.1, 0.0.0.0, ::1, and NO_PROXY environment variables,
    even when ports are included in the host URL.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
            return True

        no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
        if no_proxy:
            if no_proxy.strip() == "*":
                return True
            for item in no_proxy.split(","):
                item = item.strip().lower()
                if not item:
                    continue
                if ":" in item:
                    item = item.split(":")[0]
                if host == item or host.endswith("." + item):
                    return True
    except Exception:
        pass
    return False


def send_http_request(
    url: str,
    method: str = "POST",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Send an HTTP request with automatic proxy-bypass for local connections.

    Uses `requests` when available, with a fallback to `urllib.request`.
    Prevents '<urlopen error unknown url type: socks5h>' errors when local endpoints
    are queried from environments with ALL_PROXY / SOCKS proxies configured.
    """
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    bypass = should_bypass_proxy(url)

    # 1. Try requests library
    try:
        import requests

        session = requests.Session()
        if bypass:
            session.trust_env = False
            session.proxies = {"http": None, "https": None}
        if method.upper() == "POST":
            resp = session.post(url, json=payload, headers=req_headers, timeout=timeout_s)
        else:
            resp = session.get(url, headers=req_headers, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        pass
    except Exception as exc:
        import requests

        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            err_msg = exc.response.text
            raise RuntimeError(f"HTTP {exc.response.status_code} from {url}: {err_msg}") from exc
        raise

    # 2. Fallback to urllib.request
    import urllib.error
    import urllib.request

    data_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method.upper())

    if bypass:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        opener = urllib.request.build_opener()

    try:
        with opener.open(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc.reason}") from exc


class EndpointVerifier(BaseVerifier):
    """Verifier querying OpenAI / vLLM compatible multimodal audio endpoints."""

    supports_concurrency: bool = True

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1/chat/completions",
        model: str = "default",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("UNSLOTH_API_KEY")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("Initialized EndpointVerifier targeting '%s' (model='%s').", endpoint, model)

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, payload: dict[str, Any], endpoint: str | None = None) -> dict[str, Any]:
        target = endpoint or self.endpoint
        return send_http_request(
            url=target,
            method="POST",
            payload=payload,
            headers=self._get_headers(),
            timeout_s=self.timeout_s,
        )

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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        t0 = time.time()
        res = self._post_json(payload)
        latency = round(time.time() - t0, 3)

        choices = res.get("choices", [])
        if not choices:
            raise RuntimeError(f"Endpoint returned no choices: {res}")

        msg = choices[0].get("message", {})
        raw_content = msg.get("content")
        reasoning = msg.get("reasoning_content") or ""

        # Handle thinking / reasoning models (e.g. MOSS-Audio-8B-Thinking)
        target_text = ""
        if isinstance(raw_content, str) and raw_content.strip():
            target_text = raw_content.strip()
        elif isinstance(reasoning, str) and reasoning.strip():
            target_text = reasoning.strip()
        elif isinstance(raw_content, list):
            # Content may be a list of text parts
            parts = [p.get("text", "") for p in raw_content if isinstance(p, dict)]
            target_text = "".join(parts).strip()

        if not target_text:
            raise RuntimeError(f"Endpoint returned empty content and reasoning: {msg}")

        parsed = extract_json_payload(target_text)
        parsed["_latency_s"] = latency
        if reasoning:
            parsed["_reasoning"] = reasoning
        return parsed
