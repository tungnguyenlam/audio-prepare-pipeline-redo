"""Harden Gemini raw generation into an acoustic verifier verdict."""

from __future__ import annotations

import contextlib
import logging
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
    verdict_processor,
)
from _common.files import batch, destinations, parser, positive_int  # noqa: E402
from gemini import GeminiAgent, configure_gemini_paths, positive_float  # noqa: E402

logger = logging.getLogger("agent.verifier.gemini")


class GeminiVerifier(GeminiAgent):
    """Gemini generation constrained and parsed as a pass/reject verdict."""

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        generated = self.generate(audio_path, prompt, json_response=True)
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
        parsed["_reasoning_effort"] = self.reasoning_effort
        parsed["_inference_mode"] = generated.get("inference_mode", "standard")
        parsed["_engine"] = "gemini"
        if generated.get("batch_job"):
            parsed["_batch_job"] = generated["batch_job"]
        if generated.get("batch_request_key"):
            parsed["_batch_request_key"] = generated["batch_request_key"]
        provider_body = generated["provider_body"]
        if provider_body.get("modelVersion"):
            parsed["_model_version"] = provider_body["modelVersion"]
        if provider_body.get("responseId"):
            parsed["_response_id"] = provider_body["responseId"]

        with self._lock:
            running = round(float(self._cost_totals.get("total_usd", 0.0)), 6)
        cost = generated["cost"]
        if cost:
            logger.info(
                "Gemini %s -> %s | tokens p/o/t=%d/%d/%d | cost=$%.6f | session=$%.6f",
                audio_path.name,
                parsed.get("decision", "?"),
                generated["usage"].get("prompt_tokens", 0),
                generated["usage"].get("output_tokens", 0),
                generated["usage"].get("thinking_tokens", 0),
                float(cost["total_usd"]),
                running,
            )
        else:
            logger.info(
                "Gemini %s -> %s | tokens p/o/t=%d/%d/%d | unpriced",
                audio_path.name,
                parsed.get("decision", "?"),
                generated["usage"].get("prompt_tokens", 0),
                generated["usage"].get("output_tokens", 0),
                generated["usage"].get("thinking_tokens", 0),
            )
        return parsed


def main() -> int:
    command = parser(
        "Verify audio with Gemini; writes verdicts without filtering audio.",
        "agent/verifier",
        "gemini",
    )
    command.add_argument("--prompt-file", type=Path, help="Path to prompt text file")
    command.add_argument("--model", default="gemini-3.8-flash", help="Gemini model name")
    command.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "none"),
        default="medium",
        help="Reasoning effort level for models supporting thinking",
    )
    command.add_argument("--max-tokens", type=positive_int, default=2048, help="Maximum output tokens")
    command.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    command.add_argument("--top-p", type=float, help="Nucleus sampling top-p probability threshold")
    command.add_argument("--top-k", type=positive_int, help="Top-k sampling parameter")
    command.add_argument("--timeout-s", type=positive_float, default=120.0, help="HTTP request timeout in seconds")
    command.add_argument("--max-retries", type=positive_int, default=5, help="Maximum retry attempts per request")
    command.add_argument(
        "--inference-mode",
        choices=("batch", "standard"),
        default="batch",
        help="Gemini provider mode; batch is asynchronous and billed at Batch rates",
    )
    command.add_argument(
        "--batch-size",
        type=positive_int,
        default=10,
        help="Maximum requests per Gemini Batch job (also capped below 20 MB)",
    )
    command.add_argument("--batch-poll-interval-s", type=positive_float, default=10.0, help="Seconds between Batch status polls")
    command.add_argument("--batch-timeout-s", type=positive_float, default=86400.0, help="Maximum seconds to wait for Batch completion")
    args = command.parse_args()
    configure_gemini_paths(args, "agent/verifier")

    pairs = destinations(args, "_gemini", ".json")
    prompt = load_prompt(args.prompt_file)
    init_parameters = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "timeout_s": args.timeout_s,
        "max_retries": args.max_retries,
        "inference_mode": args.inference_mode,
    }
    with contextlib.redirect_stdout(sys.stderr):
        verifier = GeminiVerifier(**init_parameters)

    parameters = resolved_parameters({**init_parameters, "prompt": prompt}, verifier)
    pairs = pending_verifier_pairs(
        args=args,
        pairs=pairs,
        backend="gemini",
        parameters=parameters,
    )
    if args.inference_mode == "batch":
        generated = verifier.generate_batch(
            [source for source, _ in pairs],
            prompt,
            json_response=True,
            batch_size=args.batch_size,
            poll_interval_s=args.batch_poll_interval_s,
            batch_timeout_s=args.batch_timeout_s,
            state_dir=args.work_dir / "batch_jobs",
            reuse_state=not args.overwrite,
        )

        def verify(source: Path) -> dict[str, Any]:
            value = generated[source.resolve()]
            if isinstance(value, Exception):
                raise value
            return verifier.parse_generated(source, value)
    else:
        verify = lambda source: verifier.verify(source, prompt)

    process = verdict_processor(
        args=args,
        backend="gemini",
        parameters=parameters,
        verify=verify,
    )
    result = batch(
        pairs,
        process,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
    cost_summary = verifier.get_cost_summary()
    logger.info(
        "Session summary: %d input tokens (%d cached), %d output tokens, %d think tokens | %s cost: $%.6f",
        cost_summary["usage"].get("prompt_tokens", 0),
        cost_summary["usage"].get("cached_input_tokens", 0),
        cost_summary["usage"].get("output_tokens", 0),
        cost_summary["usage"].get("thinking_tokens", 0),
        cost_summary["cost"].get("pricing_tier") or args.inference_mode,
        cost_summary["cost"].get("total_usd", 0.0),
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
