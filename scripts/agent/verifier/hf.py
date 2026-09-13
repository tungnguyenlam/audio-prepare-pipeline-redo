"""Harden Hugging Face raw generation into an acoustic verifier verdict."""

from __future__ import annotations

import argparse
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
from hf import HFAgent  # noqa: E402


class DefaultHFVerifier(HFAgent):
    """Hugging Face generation parsed as a pass/reject verdict."""

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        generated = self.generate(audio_path, prompt)
        parsed = parse_verifier_response(generated["text"])
        parsed["_latency_s"] = generated["latency_s"]
        parsed["_engine"] = "huggingface"
        parsed["_model"] = self.model_id
        return parsed


def main() -> int:
    command = parser(
        "Verify audio with a Hugging Face model; writes verdicts without filtering audio.",
        "agent/verifier",
        "hf",
    )
    command.add_argument("--prompt-file", type=Path, help="Prompt file (default: prompts/acoustic_defect-3.txt)")
    command.add_argument("--model-id", default="google/gemma-4-E2B-it", help="Hugging Face model repository ID")
    command.add_argument("--device", default="auto", help='Inference device ("auto", "cpu", "cuda", or "hip")')
    command.add_argument("--adapter-path", help="Optional LoRA adapter checkpoint directory")
    command.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow executing custom code from the model repository",
    )
    command.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="PyTorch weights dtype",
    )
    command.add_argument("--load-in-4bit", action="store_true", help="Load in 4-bit NF4 with bitsandbytes")
    command.add_argument("--load-in-8bit", action="store_true", help="Load in 8-bit with bitsandbytes")
    args = command.parse_args()

    pairs = destinations(args, "_hf", ".json")
    parameters = {
        key: getattr(args, key)
        for key in (
            "model_id",
            "device",
            "adapter_path",
            "trust_remote_code",
            "torch_dtype",
            "load_in_4bit",
            "load_in_8bit",
        )
    }
    prompt = load_prompt(args.prompt_file)
    with contextlib.redirect_stdout(sys.stderr):
        verifier = DefaultHFVerifier(**parameters)
    parameters = resolved_parameters({**parameters, "prompt": prompt}, verifier)
    return run_verifier(
        args=args,
        pairs=pairs,
        backend="hf",
        parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt),
    )


if __name__ == "__main__":
    raise SystemExit(main())
