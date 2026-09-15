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
from _common.files import destinations, parser, positive_int  # noqa: E402
from hf import HFAgent  # noqa: E402


class DefaultHFVerifier(HFAgent):
    """Hugging Face generation parsed as a pass/reject verdict."""

    _PASS_REPAIR = """

IMPORTANT PASS-RESPONSE REPAIR:
Do not close a pass JSON object before its final "transcript" field. A pass is
incomplete until you add a non-empty transcript using the words actually heard in
the audio, and only then close the JSON object. Return the entire JSON verdict
again, not just the missing field.
""".strip()

    def verify(
        self, audio_path: Path, prompt: str, max_new_tokens: int
    ) -> dict[str, Any]:
        generated = self.generate(
            audio_path,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        latency_s = float(generated["latency_s"])
        usage = generated.get("usage")
        parsed = parse_verifier_response(generated["text"])
        missing_pass_fields = []
        if parsed.get("decision") == "pass":
            for field in ("emotion", "transcript"):
                value = parsed.get(field)
                if not isinstance(value, str) or not value.strip():
                    missing_pass_fields.append(field)
        if missing_pass_fields:
            generated = self.generate(
                audio_path,
                f"{prompt}\n\n{self._PASS_REPAIR}",
                max_new_tokens=max_new_tokens,
            )
            latency_s += float(generated["latency_s"])
            retry_usage = generated.get("usage")
            if isinstance(usage, dict) and isinstance(retry_usage, dict):
                usage = {
                    **usage,
                    "output_tokens": int(usage.get("output_tokens", 0) or 0)
                    + int(retry_usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0)
                    + int(retry_usage.get("total_tokens", 0) or 0),
                }
            parsed = parse_verifier_response(generated["text"])
            parsed["_schema_retry"] = {
                "count": 1,
                "reason": "missing_" + "_and_".join(missing_pass_fields),
            }
        parsed["_latency_s"] = round(latency_s, 3)
        if isinstance(usage, dict):
            parsed["_usage"] = usage
        parsed["_engine"] = "huggingface"
        parsed["_model"] = self.model_id
        parsed["_model_class"] = self.model_class
        parsed["_processor_class"] = self.processor_class
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
    command.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=1024,
        help="Maximum number of tokens generated for the verdict and transcript",
    )
    command.add_argument("--load-in-4bit", action="store_true", help="Load in 4-bit NF4 with bitsandbytes")
    command.add_argument("--load-in-8bit", action="store_true", help="Load in 8-bit with bitsandbytes")
    args = command.parse_args()

    pairs = destinations(args, "_hf", ".json")
    init_parameters = {
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
        verifier = DefaultHFVerifier(**init_parameters)
    parameters = resolved_parameters(
        {**init_parameters, "max_new_tokens": args.max_new_tokens, "prompt": prompt},
        verifier,
    )
    return run_verifier(
        args=args,
        pairs=pairs,
        backend="hf",
        parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt, args.max_new_tokens),
    )


if __name__ == "__main__":
    raise SystemExit(main())
