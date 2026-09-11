"""Harden OpenAI-compatible raw generation into an acoustic verifier verdict."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(AGENT_DIR))

from _audio import extract_json_payload  # noqa: E402
from _cli import load_prompt, resolved_parameters, run_verifier  # noqa: E402
from _common.files import destinations, parser  # noqa: E402
from endpoint import EndpointAgent, send_http_request  # noqa: E402,F401


class EndpointVerifier(EndpointAgent):
    """OpenAI-compatible generation parsed as a pass/reject verdict."""

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
        usage = generated["provider_body"].get("usage")
        if isinstance(usage, dict):
            parsed["_usage"] = usage
        return parsed


def main() -> int:
    command = parser(
        "Verify audio with an OpenAI-compatible endpoint; writes verdicts without filtering audio.",
        "agent/verifier",
        "endpoint",
    )
    command.add_argument("--prompt-file", type=Path, help="Optional prompt file; defaults to acoustic defect prompt")
    command.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions", help="Chat completions endpoint URL")
    command.add_argument("--model", default="default", help="Model name to request")
    command.add_argument("--timeout-s", type=float, default=120.0, help="Request timeout in seconds")
    command.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    command.add_argument("--max-tokens", type=int, default=1024, help="Maximum generated tokens")
    args = command.parse_args()

    pairs = destinations(args, "_endpoint", ".json")
    parameters = {
        key: getattr(args, key)
        for key in ("endpoint", "model", "timeout_s", "temperature", "max_tokens")
    }
    prompt = load_prompt(args.prompt_file)
    with contextlib.redirect_stdout(sys.stderr):
        verifier = EndpointVerifier(**parameters)
    parameters = resolved_parameters({**parameters, "prompt": prompt}, verifier)
    return run_verifier(
        args=args,
        pairs=pairs,
        backend="endpoint",
        parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt),
    )


if __name__ == "__main__":
    raise SystemExit(main())
