from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

from _audio import extract_json_payload
from endpoint import (
    EndpointVerifier,
    send_http_request,
)

logger = logging.getLogger("verifier.unsloth")

DEFAULT_UNSLOTH_HOST = "localhost"
DEFAULT_UNSLOTH_PORT = 8888
DEFAULT_UNSLOTH_MODEL = "unsloth/gemma-4-12b-it-GGUF"


def _derive_unsloth_url(endpoint: str, path: str) -> str:
    """Derive an endpoint URL (e.g. /v1/status, /v1/load, /v1/models) from chat endpoint."""
    url = endpoint.rstrip("/")
    if url.endswith("/chat/completions"):
        base = url[: -len("/chat/completions")]
    elif url.endswith("/v1"):
        base = url
    else:
        base = f"{url}/v1"
    clean_path = path.lstrip("/")
    if clean_path.startswith("v1/"):
        clean_path = clean_path[3:]
    return f"{base}/{clean_path}"


class UnslothVerifier(EndpointVerifier):
    """Specialized verifier for Unsloth Studio / Unsloth inference servers.

    Inherits from EndpointVerifier and provides Unsloth-optimized capabilities:
    - Default endpoint auto-configured from UNSLOTH_ENDPOINT, or
      http://{UNSLOTH_HOST:localhost}:{UNSLOTH_PORT:8888}/v1/chat/completions.
    - Default model auto-configured from UNSLOTH_MODEL, probed from /v1/models,
      or 'unsloth/gemma-4-12b-it-GGUF'.
    - GGUF variant support (e.g. Q8_0, UD-Q6_K_XL, Q4_K_M, BF16) passed in completions,
      queried via /v1/status, and switchable via load_model().
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
        gguf_variant: str | None = None,
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

        # 3. Resolve GGUF variant
        self.gguf_variant = gguf_variant or os.getenv("UNSLOTH_GGUF_VARIANT") or os.getenv("GGUF_VARIANT")

        # 4. Resolve model
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
            "Initialized UnslothVerifier targeting '%s' (model='%s', variant=%s, api_key_configured=%s, mode=%s).",
            self.endpoint,
            self.model,
            self.gguf_variant,
            bool(self.api_key),
            self.payload_mode,
        )

    @classmethod
    def _probe_loaded_model(cls, endpoint: str, api_key: str | None = None, timeout_s: float = 3.0) -> str | None:
        """Probe /v1/status or /v1/models on the Unsloth endpoint to discover loaded model."""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Try /v1/status first
        try:
            status_url = _derive_unsloth_url(endpoint, "status")
            sdata = send_http_request(status_url, method="GET", headers=headers, timeout_s=timeout_s)
            if isinstance(sdata, dict) and sdata.get("active_model"):
                return str(sdata["active_model"])
        except Exception:
            pass

        # Fallback to /v1/models
        try:
            models_url = _derive_unsloth_url(endpoint, "models")
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
        """Check if Unsloth endpoint is reachable, responsive, and report status/variant."""
        headers = self._get_headers()
        status_url = _derive_unsloth_url(self.endpoint, "status")
        models_url = _derive_unsloth_url(self.endpoint, "models")

        active_model = self.model
        active_variant = self.gguf_variant
        server_status = "ready"
        models: list[str] = []

        # 1. Try querying /v1/status
        try:
            sdata = send_http_request(status_url, method="GET", headers=headers, timeout_s=timeout_s)
            if isinstance(sdata, dict):
                active_model = sdata.get("active_model") or self.model
                active_variant = sdata.get("gguf_variant") or self.gguf_variant
                server_status = sdata.get("status", "ready")
                loaded = sdata.get("loaded", [])
                if isinstance(loaded, list):
                    models.extend(str(m) for m in loaded if m)
        except Exception:
            pass

        # 2. Query /v1/models
        try:
            res = send_http_request(models_url, method="GET", headers=headers, timeout_s=timeout_s)
            data = res.get("data")
            if isinstance(data, list):
                for m in data:
                    if isinstance(m, dict) and m.get("id"):
                        mid = str(m["id"])
                        if mid not in models:
                            models.append(mid)
            elif res.get("id"):
                mid = str(res["id"])
                if mid not in models:
                    models.append(mid)
        except Exception as exc:
            if not models and server_status != "loaded":
                return {
                    "ready": False,
                    "endpoint": self.endpoint,
                    "models": [],
                    "active_model": self.model,
                    "gguf_variant": self.gguf_variant,
                    "message": f"Unsloth endpoint unreachable at {self.endpoint}: {exc}",
                }

        return {
            "ready": True,
            "endpoint": self.endpoint,
            "models": models,
            "active_model": active_model,
            "gguf_variant": active_variant,
            "status": server_status,
            "message": f"Unsloth endpoint is ready (active={active_model}, variant={active_variant}).",
        }

    def load_model(
        self,
        model_path: str | None = None,
        gguf_variant: str | None = None,
        force_cancel_active: bool = True,
        timeout_s: float = 60.0,
        poll_interval_s: float = 2.0,
    ) -> bool:
        """Request Unsloth Studio to load a specific model and/or GGUF variant."""
        target_model = model_path or self.model
        target_variant = gguf_variant or self.gguf_variant

        load_url = _derive_unsloth_url(self.endpoint, "load")
        payload: dict[str, Any] = {
            "model_path": target_model,
            "force_cancel_active": force_cancel_active,
        }
        if target_variant:
            payload["gguf_variant"] = target_variant

        logger.info(
            "Requesting Unsloth to load '%s' (variant=%s)...",
            target_model,
            target_variant,
        )
        try:
            res = send_http_request(
                load_url,
                method="POST",
                payload=payload,
                headers=self._get_headers(),
                timeout_s=15.0,
            )
            logger.info("Unsloth /v1/load response: %s", res.get("status") if isinstance(res, dict) else res)
        except Exception as exc:
            logger.warning("Unsloth /v1/load request failed: %s", exc)
            return False

        # Poll status
        status_url = _derive_unsloth_url(self.endpoint, "status")
        t_end = time.time() + timeout_s
        attempt = 0
        while time.time() < t_end:
            attempt += 1
            time.sleep(poll_interval_s)
            try:
                sdata = send_http_request(
                    status_url,
                    method="GET",
                    headers=self._get_headers(),
                    timeout_s=5.0,
                )
                active = sdata.get("active_model")
                loaded = sdata.get("loaded", [])
                variant = sdata.get("gguf_variant")
                status = sdata.get("status")
                logger.debug(
                    "Status poll [%d]: status=%s active=%s variant=%s",
                    attempt,
                    status,
                    active,
                    variant,
                )
                if active == target_model or target_model in loaded or status == "loaded":
                    if not target_variant or variant == target_variant:
                        logger.info(
                            "Model %s (variant=%s) is successfully loaded and ready for inference!",
                            target_model,
                            target_variant,
                        )
                        self.model = target_model
                        if target_variant:
                            self.gguf_variant = target_variant
                        return True
            except Exception as exc:
                logger.debug("Status check poll error: %s", exc)

        logger.warning("Timed out waiting for Unsloth to load %s (variant=%s)", target_model, target_variant)
        return False

    def _build_payload(self, audio_b64: str, prompt: str, mode: str) -> dict[str, Any]:
        """Build request payload according to configured mode and GGUF variant."""
        if mode == "audio_base64":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "audio_base64": audio_b64,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        elif mode == "standard":
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
        else:  # hybrid
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
                "audio_base64": audio_b64,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

        if self.gguf_variant:
            payload["gguf_variant"] = self.gguf_variant

        return payload

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        t0 = time.time()
        payload = self._build_payload(audio_b64, prompt, mode=self.payload_mode)

        try:
            res = self._post_json(payload)
        except Exception as exc:
            err_str = str(exc).lower()
            # If server rejected gguf_variant in completion payload
            if "gguf_variant" in err_str and "gguf_variant" in payload:
                logger.debug("Server rejected payload gguf_variant, retrying without it.")
                payload.pop("gguf_variant", None)
                res = self._post_json(payload)
            # If server rejected root audio_base64, try standard format
            elif self.payload_mode == "hybrid" and (
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
        parsed["_engine"] = "unsloth"
        if reasoning:
            parsed["_reasoning"] = reasoning
        if self.gguf_variant:
            parsed["_gguf_variant"] = self.gguf_variant
        return parsed


def main() -> int:
    import argparse
    import contextlib
    from _audio import DEFAULT_ACOUSTIC_PROMPT
    from _common.files import batch, destinations, identity, parser, read_json, request, write_json
    p = parser('Verify audio with unsloth; writes verdicts without filtering audio.', 'verify', 'unsloth')
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--endpoint', type=str, default=None)
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--gguf-variant', type=str, default=None)
    p.add_argument('--timeout-s', type=float, default=120.0)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--auto-probe-model', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--payload-mode', type=str, default='hybrid')
    args = p.parse_args()
    pairs = destinations(args, '_unsloth', '.json')
    parameters = {key: getattr(args, key) for key in ('endpoint', 'model', 'gguf_variant', 'timeout_s', 'temperature', 'max_tokens', 'auto_probe_model', 'payload_mode')}
    if args.prompt_file and args.prompt_file.is_file():
        prompt = args.prompt_file.read_text(encoding='utf-8').strip()
    elif Path('prompts/acoustic_defect.txt').is_file():
        prompt = Path('prompts/acoustic_defect.txt').read_text(encoding='utf-8').strip()
    else:
        prompt = DEFAULT_ACOUSTIC_PROMPT
    with contextlib.redirect_stdout(sys.stderr):
        verifier = UnslothVerifier(**parameters)
    parameters['prompt'] = prompt
    # Record environment-derived model and endpoint values, never API credentials.
    for key in ('model', 'model_id', 'endpoint', 'gguf_variant', 'device'):
        if hasattr(verifier, key) and isinstance(getattr(verifier, key), (str, int, float, bool, type(None))):
            parameters[key] = getattr(verifier, key)
    def process(src, dest):
        wanted = request(identity(src), 'verify', parameters, 'unsloth')
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
