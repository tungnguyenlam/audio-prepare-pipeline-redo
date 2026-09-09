"""Send audio and a user-owned prompt to Gemini without constraining its reply."""

from __future__ import annotations

import sys
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VERIFY_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from _common.files import destinations, parser, positive_int  # noqa: E402
from freeform.artifacts import (  # noqa: E402
    add_prompt_arguments,
    load_prompts,
    run_freeform,
)
from gemini import GeminiVerifier  # noqa: E402


def main() -> int:
    command = parser(
        "Explore Gemini audio understanding without imposing a response schema.",
        "verify",
        "freeform-gemini",
    )
    add_prompt_arguments(command, top_p=True)
    command.add_argument("--model", default="gemini-3.8-flash")
    command.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="medium",
    )
    command.add_argument("--top-k", type=positive_int)
    command.add_argument("--timeout-s", type=float, default=120.0)
    command.add_argument("--max-retries", type=positive_int, default=5)
    args = command.parse_args()

    prompt, system_prompt = load_prompts(args)
    parameters = {
        "model": args.model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    client = GeminiVerifier(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )

    return run_freeform(
        args=args,
        pairs=destinations(args, "_gemini", ".txt"),
        backend="freeform-gemini",
        parameters=parameters,
        generate=lambda source: client.generate(
            source,
            prompt,
            system_prompt=system_prompt,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
