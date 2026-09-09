"""Freeform audio generation through an OpenAI-compatible endpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERIFY_DIR))

from scripts._common.files import destinations, parser  # noqa: E402
from endpoint import EndpointVerifier  # noqa: E402
from scripts.verify.freeform.artifacts import (  # noqa: E402
    add_prompt_arguments,
    load_prompts,
    run_freeform,
)


def main() -> int:
    command = parser(
        "Explore an OpenAI-compatible audio model without parsing its response.",
        "verify",
        "freeform-endpoint",
    )
    add_prompt_arguments(command)
    command.add_argument(
        "--endpoint", default="http://localhost:8000/v1/chat/completions"
    )
    command.add_argument("--model", default="default")
    command.add_argument("--timeout-s", type=float, default=120.0)
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
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
