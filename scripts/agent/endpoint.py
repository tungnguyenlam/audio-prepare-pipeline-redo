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
from typing import Any

logger = logging.getLogger("agent.endpoint")


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


class EndpointAgent:
    """Raw generation client for OpenAI-compatible multimodal audio endpoints."""

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
        logger.info("Initialized EndpointAgent targeting '%s' (model='%s').", endpoint, model)

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
        content = [prompt_part, audio_part]
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


def main() -> int:
    from artifacts import add_prompt_arguments, load_prompts, run_agent
    from _common.files import destinations, parser

    command = parser(
        "Explore an OpenAI-compatible audio model without parsing its response.",
        "agent",
        "endpoint",
    )
    add_prompt_arguments(command)
    command.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1/chat/completions",
        help="OpenAI-compatible chat completions endpoint URL",
    )
    command.add_argument("--model", default="default", help="Model name to request from endpoint")
    command.add_argument("--timeout-s", type=float, default=120.0, help="Request timeout in seconds")
    args = command.parse_args()

    prompt, system_prompt = load_prompts(args)
    parameters = {
        "endpoint": args.endpoint,
        "model": args.model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    client = EndpointAgent(
        endpoint=args.endpoint,
        model=args.model,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("UNSLOTH_API_KEY"),
        timeout_s=args.timeout_s,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return run_agent(
        args=args,
        pairs=destinations(args, "_endpoint", ".txt"),
        backend="endpoint",
        parameters=parameters,
        generate=lambda source: client.generate(
            source,
            prompt,
            system_prompt=system_prompt,
        ),
    )


if __name__ == '__main__':
    raise SystemExit(main())
