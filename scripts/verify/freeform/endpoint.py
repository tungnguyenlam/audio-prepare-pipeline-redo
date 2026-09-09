"""Freeform audio generation through an OpenAI-compatible endpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VERIFY_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from _common.files import destinations, parser, positive_int  # noqa: E402
from endpoint import EndpointVerifier  # noqa: E402
from freeform._common import read_prompt, run_freeform  # noqa: E402


def main() -> int:
    command = parser(
        "Explore an OpenAI-compatible audio model without parsing its response.",
        "verify",
        "freeform-endpoint",
    )
    command.add_argument("--prompt-file", type=Path, required=True)
    command.add_argument("--system-prompt-file", type=Path)
    command.add_argument(
        "--endpoint", default="http://localhost:8000/v1/chat/completions"
    )
    command.add_argument("--model", default="default")
    command.add_argument("--audio-position", choices=("before", "after"), default="before")
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--max-tokens", type=positive_int, default=4096)
    command.add_argument("--timeout-s", type=float, default=120.0)
    args = command.parse_args()

    prompt = read_prompt(args.prompt_file)
    system_prompt = (
        read_prompt(args.system_prompt_file, "System prompt")
        if args.system_prompt_file is not None
        else None
    )
    parameters = {
        "endpoint": args.endpoint,
        "model": args.model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "audio_position": args.audio_position,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    client = EndpointVerifier(
        endpoint=args.endpoint,
        model=args.model,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("UNSLOTH_API_KEY"),
        timeout_s=args.timeout_s,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    pairs = destinations(args, "_endpoint", ".txt")

    return run_freeform(
        args=args,
        pairs=pairs,
        backend="freeform-endpoint",
        parameters=parameters,
        generate=lambda source: client.generate(
            source,
            prompt,
            system_prompt=system_prompt,
            audio_position=args.audio_position,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
