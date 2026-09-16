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

from _audio import parse_verifier_response  # noqa: E402
from _cli import load_prompt, resolved_parameters, run_verifier  # noqa: E402
from _common.files import destinations, parser  # noqa: E402
from endpoint import EndpointAgent, send_http_request  # noqa: E402,F401


class EndpointVerifier(EndpointAgent):
    """OpenAI-compatible generation parsed as a pass/reject verdict."""

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        generated = self.generate(audio_path, prompt)
        text = generated["text"]
        reasoning = generated["reasoning"]
        target_text = text if isinstance(text, str) else ""
        if not target_text.strip() and isinstance(reasoning, str):
            target_text = reasoning

        parsed = parse_verifier_response(target_text)
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
        "s4-agent/verifier",
        "endpoint",
    )
    command.add_argument("-pf", "-p", "--prompt-file", type=Path, help="Prompt file (default: prompts/acoustic_defect-3.txt)")
    command.add_argument("-ep", "--endpoint", default="http://localhost:8000/v1/chat/completions", help="Chat completions endpoint URL")
    command.add_argument("-m", "--model", default="default", help="Model name to request")
    command.add_argument("--timeout-s", type=float, default=120.0, help="Request timeout in seconds")
    command.add_argument("-t", "--temperature", type=float, default=0.0, help="Sampling temperature")
    command.add_argument("-mt", "--max-tokens", type=int, default=1024, help="Maximum generated tokens")
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
