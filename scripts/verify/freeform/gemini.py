"""Send audio and a user-owned prompt to Gemini without constraining its reply."""

from __future__ import annotations

import base64
import mimetypes
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

VERIFY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VERIFY_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from _common.files import (  # noqa: E402
    destinations,
    parser,
    positive_int,
)
from _gemini_pricing import estimate_gemini_cost, normalize_gemini_usage  # noqa: E402
from freeform._common import read_prompt, run_freeform  # noqa: E402


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
    return (
        MIME_TYPES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def response_text(response: dict[str, Any]) -> str:
    """Return the first candidate's text verbatim; the sidecar retains every candidate."""
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return ""
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    return "".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


class FreeformGemini:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        reasoning_effort: str,
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        timeout_s: float,
        max_retries: int,
    ) -> None:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("The freeform Gemini command requires requests") from exc

        self.session = requests.Session()
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def generate(
        self,
        audio_path: Path,
        prompt: str,
        *,
        system_prompt: str | None,
        audio_position: str,
    ) -> tuple[dict[str, Any], dict[str, str], float]:
        audio_part = {
            "inlineData": {
                "mimeType": audio_mime_type(audio_path),
                "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            }
        }
        prompt_part = {"text": prompt}
        parts = (
            [audio_part, prompt_part]
            if audio_position == "before"
            else [prompt_part, audio_part]
        )
        generation_config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
        }
        if self.top_p is not None:
            generation_config["topP"] = self.top_p
        if self.top_k is not None:
            generation_config["topK"] = self.top_k
        if self.reasoning_effort != "none":
            generation_config["thinkingConfig"] = {"thinkingLevel": self.reasoning_effort.upper()}

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        if system_prompt is not None:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        started = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                http_response = self.session.post(url, json=payload, timeout=self.timeout_s)
                if http_response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1) + random.uniform(0.1, 0.5))
                    continue
                if not http_response.ok:
                    raise RuntimeError(
                        f"Gemini API HTTP {http_response.status_code}: {http_response.text}"
                    )
                body = http_response.json()
                if not isinstance(body, dict):
                    raise RuntimeError("Gemini returned a non-object response")
                headers = {
                    key: value
                    for key, value in http_response.headers.items()
                    if key.lower() in {"content-type", "date", "server", "x-request-id"}
                }
                return body, headers, round(time.perf_counter() - started, 3)
            except RuntimeError:
                raise
            except Exception as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Gemini API connection error: {exc}") from exc
                time.sleep(2 ** (attempt - 1) + random.uniform(0.1, 0.5))
        raise RuntimeError("Gemini API call failed")


def main() -> int:
    command = parser(
        "Explore Gemini audio understanding without imposing a response schema.",
        "verify",
        "freeform-gemini",
    )
    command.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Prompt sent verbatim after reading UTF-8",
    )
    command.add_argument(
        "--system-prompt-file", type=Path, help="Optional Gemini system instruction"
    )
    command.add_argument("--model", default="gemini-3.8-flash")
    command.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="medium",
    )
    command.add_argument("--audio-position", choices=("before", "after"), default="before")
    command.add_argument("--max-tokens", type=positive_int, default=4096)
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--top-p", type=float)
    command.add_argument("--top-k", type=positive_int)
    command.add_argument("--timeout-s", type=float, default=120.0)
    command.add_argument("--max-retries", type=positive_int, default=5)
    args = command.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY (the shell launcher also reads it from .env)")
    prompt = read_prompt(args.prompt_file)
    system_prompt = (
        read_prompt(args.system_prompt_file, "System prompt")
        if args.system_prompt_file is not None
        else None
    )
    parameters = {
        "model": args.model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "reasoning_effort": args.reasoning_effort,
        "audio_position": args.audio_position,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    client = FreeformGemini(
        model=args.model,
        api_key=api_key,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )

    pairs = destinations(args, "_gemini", ".txt")

    def generate(source: Path) -> dict[str, Any]:
        provider_response, response_headers, latency_s = client.generate(
            source,
            prompt,
            system_prompt=system_prompt,
            audio_position=args.audio_position,
        )
        text = response_text(provider_response)
        duration_s = None
        try:
            import soundfile as sf

            duration_s = float(sf.info(str(source)).duration)
        except Exception:
            pass
        usage = normalize_gemini_usage(
            provider_response.get("usageMetadata"), audio_duration_s=duration_s
        )
        return {
            "text": text,
            "latency_s": latency_s,
            "headers": response_headers,
            "usage": usage,
            "cost": estimate_gemini_cost(args.model, usage),
            "provider_body": provider_response,
        }

    return run_freeform(
        args=args,
        pairs=pairs,
        backend="freeform-gemini",
        parameters=parameters,
        generate=generate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
