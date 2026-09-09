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

from _common.files import destinations, parser, positive_int  # noqa: E402
from freeform.artifacts import read_prompt, run_freeform  # noqa: E402
from hf import DefaultHFVerifier  # noqa: E402


def main() -> int:
    command = parser(
        "Explore a local Hugging Face audio model without parsing its response.",
        "verify",
        "freeform-hf",
    )
    command.add_argument("--prompt-file", type=Path, required=True)
    command.add_argument("--system-prompt-file", type=Path)
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
    command.add_argument("--audio-position", choices=("before", "after"), default="before")
    command.add_argument("--max-tokens", type=positive_int, default=4096)
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--top-p", type=float)
    args = command.parse_args()

    prompt = read_prompt(args.prompt_file)
    system_prompt = (
        read_prompt(args.system_prompt_file, "System prompt")
        if args.system_prompt_file is not None
        else None
    )
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
        "audio_position": args.audio_position,
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
            audio_position=args.audio_position,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
