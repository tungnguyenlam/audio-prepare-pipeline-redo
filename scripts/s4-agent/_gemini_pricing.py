"""Shared Gemini usage normalization and versioned paid cost estimates."""

from __future__ import annotations

from typing import Any, Mapping

# Paid list prices in USD per million tokens. Versioned because Google prices
# and introductory offers can change independently of this repository.
GEMINI_PRICE_CARD_AS_OF = "2026-09-11"
GEMINI_STANDARD_PRICES: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {
        "input": 0.75, "output": 3.75, "cached_input": 0.075,
        "batch_input": 0.375, "batch_output": 1.875,
        "batch_cached_input": 0.0375,
    },
    "gemini-3.7-flash": {
        "input": 0.75, "output": 3.75, "cached_input": 0.075,
        "batch_input": 0.375, "batch_output": 1.875,
        "batch_cached_input": 0.0375,
    },
    "gemini-3.6-flash": {
        "input": 0.75, "output": 3.75, "cached_input": 0.075,
        "batch_input": 0.375, "batch_output": 1.875,
        "batch_cached_input": 0.0375,
    },
    "gemini-3.5-flash": {
        "input": 1.50, "output": 9.00, "cached_input": 0.15,
        "batch_input": 0.75, "batch_output": 4.50,
        "batch_cached_input": 0.075,
    },
    "gemini-3.5-flash-lite": {
        "input": 0.30, "output": 2.50, "cached_input": 0.03,
        "batch_input": 0.15, "batch_output": 1.25,
        "batch_cached_input": 0.02,
    },
    "gemini-3.1-pro-preview": {
        "input": 2.00, "output": 12.00, "cached_input": 0.20,
        "batch_input": 1.00, "batch_output": 6.00,
        "batch_cached_input": 0.20,
    },
    "gemini-3.1-flash-lite": {
        "input": 0.25, "audio_input": 0.50, "output": 1.50,
        "cached_input": 0.025, "cached_audio_input": 0.05,
        "batch_input": 0.125, "batch_audio_input": 0.25,
        "batch_output": 0.75, "batch_cached_input": 0.0125,
        "batch_cached_audio_input": 0.025,
    },
}


def normalize_gemini_usage(
    value: Mapping[str, Any] | None,
    *,
    audio_duration_s: float | None = None,
) -> dict[str, Any]:
    """Normalize Gemini token metadata, including cache hits by modality."""
    metadata = value if isinstance(value, dict) else {}

    def modality_counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for detail in metadata.get(field, []):
            if not isinstance(detail, dict):
                continue
            modality = str(detail.get("modality", "unknown")).lower()
            count = detail.get("tokenCount", 0)
            if isinstance(count, int) and not isinstance(count, bool):
                result[modality] = result.get(modality, 0) + count
        return result

    modalities = modality_counts("promptTokensDetails")
    cached_modalities = modality_counts("cacheTokensDetails")
    prompt_tokens = int(metadata.get("promptTokenCount", 0) or 0)
    cached_tokens = min(
        prompt_tokens,
        int(metadata.get("cachedContentTokenCount", 0) or 0),
    )
    audio_tokens = modalities.get("audio", 0)
    text_tokens = modalities.get("text", 0)
    cached_audio_tokens = min(audio_tokens, cached_modalities.get("audio", 0))
    cached_text_tokens = min(text_tokens, cached_modalities.get("text", 0))
    audio_tokens_estimated = False
    if not audio_tokens and audio_duration_s and prompt_tokens:
        audio_tokens = min(prompt_tokens, round(float(audio_duration_s) * 32))
        text_tokens = max(text_tokens, prompt_tokens - audio_tokens)
        audio_tokens_estimated = True
    return {
        "prompt_tokens": prompt_tokens,
        "cached_input_tokens": cached_tokens,
        "audio_input_tokens": audio_tokens,
        "cached_audio_input_tokens": cached_audio_tokens,
        "audio_input_tokens_estimated": audio_tokens_estimated,
        "text_input_tokens": text_tokens,
        "cached_text_input_tokens": cached_text_tokens,
        "output_tokens": int(metadata.get("candidatesTokenCount", 0) or 0),
        "thinking_tokens": int(metadata.get("thoughtsTokenCount", 0) or 0),
        "total_tokens": int(metadata.get("totalTokenCount", 0) or 0),
        "service_tier": metadata.get("serviceTier"),
    }


def estimate_gemini_cost(
    model: str,
    usage: Mapping[str, Any],
    *,
    pricing_tier: str = "paid_standard",
) -> dict[str, Any] | None:
    """Estimate one request at Google's versioned paid list price."""
    rates = GEMINI_STANDARD_PRICES.get(model)
    if rates is None:
        return None
    # Flex is billed at the same 50% discount as Batch.
    if pricing_tier not in {"paid_standard", "paid_batch", "paid_flex"}:
        raise ValueError(f"Unsupported Gemini pricing tier: {pricing_tier}")
    prefix = "" if pricing_tier == "paid_standard" else "batch_"
    input_rate = rates[f"{prefix}input"]
    audio_input_rate = rates.get(f"{prefix}audio_input", input_rate)
    output_rate = rates[f"{prefix}output"]
    cached_input_rate = rates.get(f"{prefix}cached_input", input_rate)
    cached_audio_rate = rates.get(
        f"{prefix}cached_audio_input", cached_input_rate
    )

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    audio_tokens = min(prompt_tokens, int(usage.get("audio_input_tokens", 0) or 0))
    text_tokens = min(
        prompt_tokens - audio_tokens,
        int(usage.get("text_input_tokens", 0) or 0),
    )
    other_tokens = max(0, prompt_tokens - audio_tokens - text_tokens)
    cached_tokens = min(
        prompt_tokens, int(usage.get("cached_input_tokens", 0) or 0)
    )
    cached_audio_tokens = min(
        audio_tokens,
        cached_tokens,
        int(usage.get("cached_audio_input_tokens", 0) or 0),
    )
    cached_text_tokens = min(
        text_tokens,
        cached_tokens - cached_audio_tokens,
        int(usage.get("cached_text_input_tokens", 0) or 0),
    )
    unclassified_cached_tokens = max(
        0, cached_tokens - cached_audio_tokens - cached_text_tokens
    )
    cached_other_tokens = min(other_tokens, unclassified_cached_tokens)
    unclassified_cached_tokens -= cached_other_tokens
    inferred_cached_text = min(
        text_tokens - cached_text_tokens, unclassified_cached_tokens
    )
    cached_text_tokens += inferred_cached_text
    unclassified_cached_tokens -= inferred_cached_text
    cached_audio_tokens += min(
        audio_tokens - cached_audio_tokens, unclassified_cached_tokens
    )
    uncached_audio_tokens = audio_tokens - cached_audio_tokens
    uncached_text_tokens = text_tokens - cached_text_tokens
    uncached_other_tokens = max(0, other_tokens - cached_other_tokens)

    uncached_input_usd = (
        uncached_audio_tokens * audio_input_rate
        + (uncached_text_tokens + uncached_other_tokens) * input_rate
    ) / 1_000_000
    cached_input_usd = (
        cached_audio_tokens * cached_audio_rate
        + (cached_text_tokens + cached_other_tokens) * cached_input_rate
    ) / 1_000_000
    input_usd = uncached_input_usd + cached_input_usd
    billed_output_tokens = int(usage.get("output_tokens", 0) or 0) + int(
        usage.get("thinking_tokens", 0) or 0
    )
    output_usd = billed_output_tokens * output_rate / 1_000_000
    return {
        "uncached_input_usd": round(uncached_input_usd, 9),
        "cached_input_usd": round(cached_input_usd, 9),
        "input_usd": round(input_usd, 9),
        "output_usd": round(output_usd, 9),
        "total_usd": round(input_usd + output_usd, 9),
        "currency": "USD",
        "pricing_tier": pricing_tier,
        "rate_card_as_of": GEMINI_PRICE_CARD_AS_OF,
        "estimated": True,
        "model": model,
    }


def empty_usage_totals() -> dict[str, Any]:
    """Zeroed cumulative usage counters for a Gemini agent session."""
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "cached_input_tokens": 0,
        "audio_input_tokens": 0,
        "cached_audio_input_tokens": 0,
        "text_input_tokens": 0,
        "cached_text_input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
    }


def empty_cost_totals() -> dict[str, Any]:
    """Zeroed cumulative cost counters for a Gemini agent session."""
    return {
        "uncached_input_usd": 0.0,
        "cached_input_usd": 0.0,
        "input_usd": 0.0,
        "output_usd": 0.0,
        "total_usd": 0.0,
        "currency": "USD",
        "pricing_tier": None,
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
        "cached_input_tokens",
        "audio_input_tokens",
        "cached_audio_input_tokens",
        "text_input_tokens",
        "cached_text_input_tokens",
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
    for key in (
        "uncached_input_usd",
        "cached_input_usd",
        "input_usd",
        "output_usd",
        "total_usd",
    ):
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
    for key in (
        "uncached_input_usd",
        "cached_input_usd",
        "input_usd",
        "output_usd",
        "total_usd",
    ):
        cost_totals[key] = round(float(cost_totals[key]), 9)
    return {"usage": usage_totals, "cost": cost_totals}
