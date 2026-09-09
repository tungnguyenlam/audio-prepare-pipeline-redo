"""Freeform generation with a local Hugging Face audio model."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VERIFY_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from _common.files import destinations, parser  # noqa: E402
from freeform.artifacts import add_prompt_arguments, load_prompts, run_freeform  # noqa: E402
from hf import DefaultHFVerifier  # noqa: E402


def main() -> int:
    command = parser(
        "Explore a local Hugging Face audio model without parsing its response.",
        "verify",
        "freeform-hf",
    )
    add_prompt_arguments(command, top_p=True)
    command.add_argument("--model-id", default="google/gemma-4-E2B-it")
    command.add_argument("--device", default="auto")
    command.add_argument("--adapter-path")
    command.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    command.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    command.add_argument("--load-in-4bit", action="store_true")
    command.add_argument("--load-in-8bit", action="store_true")
    args = command.parse_args()

    prompt, system_prompt = load_prompts(args)
    parameters = {
        "model_id": args.model_id,
        "device": args.device,
        "adapter_path": args.adapter_path,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": args.torch_dtype,
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    with contextlib.redirect_stdout(sys.stderr):
        model = DefaultHFVerifier(
            model_id=args.model_id,
            device=args.device,
            adapter_path=args.adapter_path,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
    pairs = destinations(args, "_hf", ".txt")

    return run_freeform(
        args=args,
        pairs=pairs,
        backend="freeform-hf",
        parameters=parameters,
        generate=lambda source: model.generate(
            source,
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
