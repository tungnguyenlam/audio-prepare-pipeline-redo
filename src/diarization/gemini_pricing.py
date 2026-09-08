"""Shared Gemini usage normalization and paid-Standard cost estimates."""

from __future__ import annotations

from typing import Any, Mapping

# Paid Standard list prices in USD per million tokens. Versioned because Google
# prices and introductory offers can change independently of this repository.
GEMINI_PRICE_CARD_AS_OF = "2026-09-04"
GEMINI_STANDARD_PRICES: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3.1-flash-lite": {
        "input": 0.25,
        "audio_input": 0.50,
        "output": 1.50,
    },
}


def normalize_gemini_usage(
    value: Mapping[str, Any] | None,
    *,
    audio_duration_s: float | None = None,
) -> dict[str, Any]:
    """Normalize Gemini token metadata, retaining modality-level input counts."""
    metadata = value if isinstance(value, dict) else {}
    modalities: dict[str, int] = {}
    for detail in metadata.get("promptTokensDetails", []):
        if not isinstance(detail, dict):
            continue
        modality = str(detail.get("modality", "unknown")).lower()
        count = detail.get("tokenCount", 0)
        if isinstance(count, int) and not isinstance(count, bool):
            modalities[modality] = modalities.get(modality, 0) + count
    prompt_tokens = int(metadata.get("promptTokenCount", 0) or 0)
    audio_tokens = modalities.get("audio", 0)
    text_tokens = modalities.get("text", 0)
    audio_tokens_estimated = False
    if not audio_tokens and audio_duration_s and prompt_tokens:
        audio_tokens = min(prompt_tokens, round(float(audio_duration_s) * 32))
        text_tokens = max(text_tokens, prompt_tokens - audio_tokens)
        audio_tokens_estimated = True
    return {
        "prompt_tokens": prompt_tokens,
        "audio_input_tokens": audio_tokens,
        "audio_input_tokens_estimated": audio_tokens_estimated,
        "text_input_tokens": text_tokens,
        "output_tokens": int(metadata.get("candidatesTokenCount", 0) or 0),
        "thinking_tokens": int(metadata.get("thoughtsTokenCount", 0) or 0),
        "total_tokens": int(metadata.get("totalTokenCount", 0) or 0),
        "service_tier": metadata.get("serviceTier"),
    }


def estimate_gemini_cost(model: str, usage: Mapping[str, Any]) -> dict[str, Any] | None:
    """Estimate one request at Google's versioned paid Standard list price."""
    rates = GEMINI_STANDARD_PRICES.get(model)
    if rates is None:
        return None
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    audio_tokens = int(usage.get("audio_input_tokens", 0) or 0)
    text_tokens = int(usage.get("text_input_tokens", 0) or 0)
    if "audio_input" in rates and audio_tokens + text_tokens > 0:
        other_tokens = max(0, prompt_tokens - audio_tokens - text_tokens)
        input_usd = (
            audio_tokens * rates["audio_input"]
            + (text_tokens + other_tokens) * rates["input"]
        ) / 1_000_000
    else:
        input_usd = prompt_tokens * rates["input"] / 1_000_000
    billed_output_tokens = int(usage.get("output_tokens", 0) or 0) + int(
        usage.get("thinking_tokens", 0) or 0
    )
    output_usd = billed_output_tokens * rates["output"] / 1_000_000
    return {
        "input_usd": round(input_usd, 9),
        "output_usd": round(output_usd, 9),
        "total_usd": round(input_usd + output_usd, 9),
        "currency": "USD",
        "pricing_tier": "paid_standard",
        "rate_card_as_of": GEMINI_PRICE_CARD_AS_OF,
        "estimated": True,
        "model": model,
    }


def empty_usage_totals() -> dict[str, Any]:
    """Zeroed cumulative usage counters for a GeminiVerifier session."""
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "audio_input_tokens": 0,
        "text_input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
    }


def empty_cost_totals() -> dict[str, Any]:
    """Zeroed cumulative cost counters for a GeminiVerifier session."""
    return {
        "input_usd": 0.0,
        "output_usd": 0.0,
        "total_usd": 0.0,
        "currency": "USD",
        "pricing_tier": "paid_standard",
        "rate_card_as_of": GEMINI_PRICE_CARD_AS_OF,
        "estimated": True,
        "priced_requests": 0,
        "unpriced_requests": 0,
    }


def accumulate_usage(totals: dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Add one normalized usage record into cumulative totals (in place)."""
    totals["requests"] = int(totals.get("requests", 0)) + 1
    for key in (
        "prompt_tokens",
        "audio_input_tokens",
        "text_input_tokens",
        "output_tokens",
        "thinking_tokens",
        "total_tokens",
    ):
        totals[key] = int(totals.get(key, 0)) + int(usage.get(key, 0) or 0)


def accumulate_cost(totals: dict[str, Any], cost: Mapping[str, Any] | None) -> None:
    """Add one cost estimate into cumulative totals (in place)."""
    if not cost:
        totals["unpriced_requests"] = int(totals.get("unpriced_requests", 0)) + 1
        return
    totals["priced_requests"] = int(totals.get("priced_requests", 0)) + 1
    for key in ("input_usd", "output_usd", "total_usd"):
        totals[key] = float(totals.get(key, 0.0)) + float(cost.get(key, 0.0) or 0.0)
    totals["currency"] = cost.get("currency", totals.get("currency", "USD"))
    totals["pricing_tier"] = cost.get("pricing_tier", totals.get("pricing_tier"))
    totals["rate_card_as_of"] = cost.get(
        "rate_card_as_of", totals.get("rate_card_as_of")
    )
    totals["estimated"] = bool(cost.get("estimated", True))


def aggregate_prediction_costs(predictions: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate `_usage` / `_cost` fields from verifier prediction dicts."""
    usage_totals = empty_usage_totals()
    cost_totals = empty_cost_totals()
    for pred in predictions:
        usage = pred.get("_usage")
        if isinstance(usage, Mapping):
            accumulate_usage(usage_totals, usage)
        cost = pred.get("_cost")
        if isinstance(cost, Mapping) or cost is None:
            if "_cost" in pred or "_usage" in pred:
                accumulate_cost(cost_totals, cost if isinstance(cost, Mapping) else None)
    for key in ("input_usd", "output_usd", "total_usd"):
        cost_totals[key] = round(float(cost_totals[key]), 9)
    return {"usage": usage_totals, "cost": cost_totals}
