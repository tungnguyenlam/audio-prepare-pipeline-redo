"""Harden Gemini raw generation into an acoustic verifier verdict."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(AGENT_DIR))

from _audio import parse_verifier_response  # noqa: E402
from _cli import (  # noqa: E402
    load_prompt,
    pending_verifier_pairs,
    resolved_parameters,
    run_verifier,
)
from _common.files import destinations, parser, positive_int  # noqa: E402
from gemini import (  # noqa: E402
    GEMINI_MODEL,
    GeminiAgent,
    add_gemini_arguments,
    configure_gemini_paths,
    gemini_parameters,
    generation_callback,
)


class GeminiVerifier(GeminiAgent):
    """Gemini generation constrained and parsed as a pass/reject verdict."""

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        generated = self.generate(audio_path, prompt)
        return self.parse_generated(audio_path, generated)

    def parse_generated(
        self,
        audio_path: Path,
        generated: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse a standard or Batch generation through the same verdict path."""
        raw_text = generated["text"]
        parsed = parse_verifier_response(raw_text)
        parsed["_latency_s"] = generated["latency_s"]
        parsed["_usage"] = generated["usage"]
        parsed["_cost"] = generated["cost"]
        parsed["_model"] = self.model
        if self.model != GEMINI_MODEL:
            raise RuntimeError(
                f"Verifier must use {GEMINI_MODEL!r}, not {self.model!r}."
            )
        parsed["_reasoning_effort"] = self.reasoning_effort
        parsed["_inference_mode"] = generated.get("inference_mode", "standard")
        parsed["_requested_inference_mode"] = self.inference_mode
        parsed["_cache_prompt"] = self.cache_prompt
        parsed["_engine"] = "gemini"
        if generated.get("batch_job"):
            parsed["_batch_job"] = generated["batch_job"]
        if generated.get("batch_request_key"):
            parsed["_batch_request_key"] = generated["batch_request_key"]
        provider_body = generated["provider_body"]
        if generated.get("model_version"):
            parsed["_model_version"] = generated["model_version"]
        if provider_body.get("responseId"):
            parsed["_response_id"] = provider_body["responseId"]

        return parsed


def main() -> int:
    command = parser(
        "Verify audio with Gemini; writes verdicts without filtering audio.",
        "s4-agent/verifier",
        "gemini",
    )
    command.add_argument("-pf", "-p", "--prompt-file", type=Path, help="Prompt text file (default: prompts/full-tags-prompt.md)")
    command.add_argument("-mt", "--max-tokens", type=positive_int, default=65536, help="Maximum output tokens")
    add_gemini_arguments(command)
    args = command.parse_args()
    configure_gemini_paths(args, "s4-agent/verifier")

    pairs = destinations(args, "_gemini", ".json")
    prompt = load_prompt(args.prompt_file)
    init_parameters = gemini_parameters(args)
    verifier = GeminiVerifier(**init_parameters)

    parameters = resolved_parameters({**init_parameters, "prompt": prompt}, verifier)
    all_pairs = pairs
    pairs = pending_verifier_pairs(
        args=args,
        pairs=pairs,
        backend="gemini",
        parameters=parameters,
    )
    generate = generation_callback(verifier, args, pairs, prompt, all_pairs=all_pairs)

    return run_verifier(
        args=args,
        pairs=all_pairs,
        backend="gemini",
        parameters=parameters,
        verify=lambda source: verifier.parse_generated(source, generate(source)),
        cost_summary=verifier.get_cost_summary,
    )


if __name__ == "__main__":
    raise SystemExit(main())