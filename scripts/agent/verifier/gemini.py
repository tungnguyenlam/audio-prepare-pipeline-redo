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
from _cli import load_prompt, resolved_parameters, verdict_processor  # noqa: E402
from _common.files import batch, destinations, parser, positive_int  # noqa: E402
from gemini import GeminiAgent  # noqa: E402

logger = logging.getLogger("agent.verifier.gemini")


class GeminiVerifier(GeminiAgent):
    """Gemini generation constrained and parsed as a pass/reject verdict."""

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        generated = self.generate(audio_path, prompt, json_response=True)
        raw_text = generated["text"]
        parsed = parse_verifier_response(raw_text)
        parsed["_latency_s"] = generated["latency_s"]
        parsed["_usage"] = generated["usage"]
        parsed["_cost"] = generated["cost"]
        parsed["_model"] = self.model
        parsed["_reasoning_effort"] = self.reasoning_effort
        parsed["_engine"] = "gemini"
        provider_body = generated["provider_body"]
        if provider_body.get("modelVersion"):
            parsed["_model_version"] = provider_body["modelVersion"]

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
    command.add_argument("--timeout-s", type=float, default=120.0, help="Request timeout in seconds")
    command.add_argument("--max-retries", type=positive_int, default=5, help="Maximum retry attempts per request")
    args = command.parse_args()

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
    }
    with contextlib.redirect_stdout(sys.stderr):
        verifier = GeminiVerifier(**init_parameters)

    parameters = resolved_parameters({**init_parameters, "prompt": prompt}, verifier)
    process = verdict_processor(
        args=args,
        backend="gemini",
        parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt),
    )
    result = batch(
        pairs,
        process,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
    cost_summary = verifier.get_cost_summary()
    logger.info(
        "Session summary: %d input tokens, %d output tokens, %d think tokens | Total cost: $%.6f",
        cost_summary["usage"].get("prompt_tokens", 0),
        cost_summary["usage"].get("output_tokens", 0),
        cost_summary["usage"].get("thinking_tokens", 0),
        cost_summary["cost"].get("total_usd", 0.0),
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
