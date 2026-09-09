from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import base64
import json
import logging
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any

from _audio import (
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


class EndpointVerifier:
    """Verifier querying OpenAI / vLLM compatible multimodal audio endpoints."""


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

    def generate(
        self,
        audio_path: Path,
        prompt: str,
        *,
        system_prompt: str | None = None,
        audio_position: str = "before",
    ) -> dict[str, Any]:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        audio_part = {
            "type": "input_audio",
            "input_audio": {
                "data": audio_b64,
                "format": audio_path.suffix.lstrip(".").lower() or "wav",
            },
        }
        prompt_part = {"type": "text", "text": prompt}
        content = (
            [audio_part, prompt_part]
            if audio_position == "before"
            else [prompt_part, audio_part]
        )
        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": messages,
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

        text = raw_content if isinstance(raw_content, str) else ""
        if isinstance(raw_content, list):
            text = "".join(
                part.get("text", "")
                for part in raw_content
                if isinstance(part, dict) and isinstance(part.get("text", ""), str)
            )
        if not text and isinstance(reasoning, str):
            text = reasoning
        return {
            "text": text,
            "reasoning": reasoning,
            "latency_s": latency,
            "provider_body": res,
        }

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        generated = self.generate(audio_path, prompt)
        text = generated["text"]
        reasoning = generated["reasoning"]
        target_text = text.strip() if isinstance(text, str) else ""
        if not target_text and isinstance(reasoning, str):
            target_text = reasoning.strip()

        if not target_text:
            raise RuntimeError(
                f"Endpoint returned empty content and reasoning: {generated['provider_body']}"
            )

        parsed = extract_json_payload(target_text)
        parsed["_latency_s"] = generated["latency_s"]
        if reasoning:
            parsed["_reasoning"] = reasoning
        return parsed


def main() -> int:
    import argparse
    import contextlib
    from _audio import DEFAULT_ACOUSTIC_PROMPT
    from _common.files import batch, destinations, identity, parser, read_json, request, write_json
    p = parser('Verify audio with endpoint; writes verdicts without filtering audio.', 'verify', 'endpoint')
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--endpoint', type=str, default='http://localhost:8000/v1/chat/completions')
    p.add_argument('--model', type=str, default='default')
    p.add_argument('--timeout-s', type=float, default=120.0)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=1024)
    args = p.parse_args()
    pairs = destinations(args, '_endpoint', '.json')
    parameters = {key: getattr(args, key) for key in ('endpoint', 'model', 'timeout_s', 'temperature', 'max_tokens')}
    prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else DEFAULT_ACOUSTIC_PROMPT
    with contextlib.redirect_stdout(sys.stderr):
        verifier = EndpointVerifier(**parameters)
    parameters['prompt'] = prompt
    # Record environment-derived model and endpoint values, never API credentials.
    for key in ('model', 'model_id', 'endpoint', 'gguf_variant', 'device'):
        if hasattr(verifier, key) and isinstance(getattr(verifier, key), (str, int, float, bool, type(None))):
            parameters[key] = getattr(verifier, key)
    def process(src, dest):
        wanted = request(identity(src), 'verify', parameters, 'endpoint')
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
