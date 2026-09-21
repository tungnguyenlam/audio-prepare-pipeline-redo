"""Readable stderr reporting shared by raw agents and verifier commands."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from _common.files import progress


def new_run_stats() -> dict[str, Any]:
    """Create mutable, thread-safe counters for one command invocation."""
    return {
        'attempted': 0,
        'succeeded': 0,
        'failed': 0,
        'decisions': {},
        'priced_requests': 0,
        'unpriced_requests': 0,
        'total_usd': 0.0,
        '_lock': threading.Lock(),
    }


def _cost(result: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(result, Mapping):
        return None
    value = result.get('cost')
    if value is None:
        value = result.get('_cost')
    return value if isinstance(value, Mapping) else None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record_result(
    stats: dict[str, Any],
    result: Mapping[str, Any] | None,
    *,
    success: bool,
    decision: str | None = None,
) -> float | None:
    """Record one result and return the running priced total, if available."""
    cost = _cost(result)
    with stats['_lock']:
        if success:
            stats['succeeded'] += 1
        else:
            stats['failed'] += 1
        if decision:
            decisions = stats['decisions']
            decisions[decision] = decisions.get(decision, 0) + 1
        if cost is None:
            stats['unpriced_requests'] += 1
            return None
        try:
            amount = float(cost.get('total_usd', 0.0) or 0.0)
        except (TypeError, ValueError):
            stats['unpriced_requests'] += 1
            return None
        stats['priced_requests'] += 1
        stats['total_usd'] += amount
        return stats['total_usd']


def item_detail(
    result: Mapping[str, Any] | None,
    *,
    text_length: int | None = None,
    running_total_usd: float | None = None,
) -> str:
    """Format non-sensitive per-item model details for stderr."""
    parts: list[str] = []
    latency = result.get('latency_s') if isinstance(result, Mapping) else None
    if latency is None and isinstance(result, Mapping):
        latency = result.get('_latency_s')
    if latency is not None:
        try:
            parts.append(f'latency={float(latency):.2f}s')
        except (TypeError, ValueError):
            pass
    usage = result.get('usage') if isinstance(result, Mapping) else None
    if usage is None and isinstance(result, Mapping):
        usage = result.get('_usage')
    if isinstance(usage, Mapping):
        prompt = _int(usage.get('prompt_tokens', 0))
        output = _int(usage.get('output_tokens', 0))
        thinking = _int(usage.get('thinking_tokens', 0))
        parts.append(f'tokens p/o/t={prompt}/{output}/{thinking}')
    if text_length is not None:
        parts.append(f'response={text_length} chars')
    cost = _cost(result)
    if cost is None:
        parts.append('cost=unavailable')
    else:
        try:
            amount = float(cost.get('total_usd', 0.0) or 0.0)
            parts.append(f'cost=${amount:.6f}')
            if running_total_usd is not None:
                parts.append(f'total=${running_total_usd:.6f}')
        except (TypeError, ValueError):
            parts.append('cost=unavailable')
    return '; '.join(parts)


def _summary_from_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'usage': {'requests': stats.get('succeeded', 0) + stats.get('failed', 0)},
        'cost': {
            'total_usd': stats.get('total_usd', 0.0),
            'priced_requests': stats.get('priced_requests', 0),
            'unpriced_requests': stats.get('unpriced_requests', 0),
            'available': stats.get('priced_requests', 0) > 0,
        },
    }


def report_cost_summary(
    label: str,
    stats: Mapping[str, Any],
    provider_summary: Mapping[str, Any] | None = None,
) -> None:
    """Print one consistent end-of-run summary, including total cost."""
    summary = provider_summary or _summary_from_stats(stats)
    usage = summary.get('usage') if isinstance(summary, Mapping) else None
    cost = summary.get('cost') if isinstance(summary, Mapping) else None
    usage = usage if isinstance(usage, Mapping) else {}
    cost = cost if isinstance(cost, Mapping) else {}
    requests = usage.get('requests')
    if requests is None:
        requests = stats.get('succeeded', 0) + stats.get('failed', 0)
    priced = _int(cost.get('priced_requests', stats.get('priced_requests', 0)))
    unpriced = _int(cost.get('unpriced_requests', stats.get('unpriced_requests', 0)))
    available = cost.get('available', priced > 0)
    try:
        total = float(cost.get('total_usd', 0.0) or 0.0)
    except (TypeError, ValueError):
        total = 0.0

    if available or priced > 0 or cost.get('cache_storage_usd'):
        cost_text = f'total=${total:.6f} USD'
        if cost.get('estimated'):
            cost_text += ' (estimated)'
        if unpriced:
            cost_text += f'; {unpriced} unpriced'
    elif requests:
        cost_text = 'total=unavailable (provider did not expose pricing)'
    else:
        cost_text = 'total=$0.000000 USD (no model requests)'

    details = [
        label,
        f'items={stats.get("succeeded", 0) + stats.get("failed", 0)}',
        f'succeeded={stats.get("succeeded", 0)}',
        f'failed={stats.get("failed", 0)}',
        f'requests={requests}',
        cost_text,
    ]
    if cost.get('cache_storage_usd'):
        details.append(f"cache_storage=${float(cost['cache_storage_usd']):.6f} (full TTL)")
    if cost.get('unpriced_caches'):
        details.append(f"unpriced_caches={cost['unpriced_caches']} (storage excluded)")
    pricing_tier = cost.get('pricing_tier')
    if pricing_tier:
        details.append(f'tier={pricing_tier}')
    decisions = stats.get('decisions') or {}
    if decisions:
        details.append('decisions=' + ','.join(f'{key}:{decisions[key]}' for key in sorted(decisions)))
    progress('TOTAL_COST', '; '.join(details))
