"""Harden Gemini raw generation into an acoustic verifier verdict."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from collections.abc import Callable

logger = logging.getLogger("verifier.gemini")

AGENT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(AGENT_DIR))

from _audio import VerifierResponseError, parse_verifier_response  # noqa: E402
from _cli import (  # noqa: E402
    load_prompt,
    pending_verifier_pairs,
    resolved_parameters,
    run_verifier,
)
from _common.files import destinations, parser, positive_int  # noqa: E402
from gemini import (  # noqa: E402
    GeminiAgent,
    add_gemini_arguments,
    configure_gemini_paths,
    gemini_parameters,
    generation_callback,
)


class GeminiVerifier(GeminiAgent):
    """Gemini generation constrained and parsed as a pass/reject verdict."""

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        max_retry = max(0, self.max_retries - 1)
        for attempt in range(1, max_retry + 2):
            generated = self.generate(audio_path, system_prompt=prompt)
            try:
                return self.parse_generated(audio_path, generated)
            except VerifierResponseError as exc:
                if attempt > max_retry:
                    raise
                raw = str(generated.get("text", ""))
                cleaned = raw.replace("\r", " ").replace("\n", " ").strip()
                snippet = f" | output: {cleaned[:200]!r}" if cleaned else " | output was empty"
                logger.warning(
                    "%s: Model failed to return a valid JSON object (%s)%s -> retrying (%d/%d)...",
                    audio_path.name,
                    exc.code,
                    snippet,
                    attempt,
                    max_retry,
                )

    def parse_generated(
        self,
        audio_path: Path,
        generated: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse a standard or Batch generation through the same verdict path."""
        raw_text = generated["text"]
        try:
            parsed = parse_verifier_response(raw_text)
        except VerifierResponseError as exc:
            exc.generation = generated
            raise
        parsed["_latency_s"] = generated["latency_s"]
        parsed["_usage"] = generated["usage"]
        parsed["_cost"] = generated["cost"]
        parsed["_model"] = self.model
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

    parameters = resolved_parameters({**init_parameters, "prompt": prompt, "prompt_role": "system"}, verifier)
    all_pairs = pairs
    pairs = pending_verifier_pairs(
        args=args,
        pairs=pairs,
        backend="gemini",
        parameters=parameters,
    )
    run_batch = None
    if args.inference_mode == "batch":
        ready: dict[Path, dict[str, Any] | Exception] = {}
        destinations_by_source = {source.resolve(): destination for source, destination in pairs}

        def generate(source: Path) -> dict[str, Any]:
            value = ready.pop(source.resolve())
            if isinstance(value, Exception):
                raise value
            return value

        def run_batch(publish: Callable[[Path, Path], None]) -> None:
            def on_result(source: Path, value: dict[str, Any] | Exception) -> None:
                ready[source] = value
                publish(source, destinations_by_source[source])

            generation_callback(verifier, args, pairs, prompt,
                                all_pairs=all_pairs, on_result=on_result)
    else:
        generate = generation_callback(verifier, args, pairs, prompt, all_pairs=all_pairs)

    return run_verifier(
        args=args,
        pairs=all_pairs,
        backend="gemini",
        parameters=parameters,
        verify=lambda source: verifier.parse_generated(source, generate(source)),
        cost_summary=verifier.get_cost_summary,
        run_batch=run_batch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
