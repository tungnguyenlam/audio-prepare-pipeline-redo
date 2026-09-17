"""Audit audio duration loss and yield across diarization, silence, and post-merge filtering.

Computes:
1. Time lost from raw audio to raw diarization turns (non-speech / acoustic silence).
2. Time lost due to silence (raw non-speech, silence bridged in merge, unmerged gaps).
3. Time lost due to post-merge duration filtering (< 1.5s too-short and > 15.0s too-long).
4. Final valid clip duration, count, and yield rates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (
    LoggingArgumentParser,
    ROOT,
    identity,
    persist_path,
    positive_int,
    probe,
    progress,
    read_json,
    resolve_stored_path,
    write_json,
)


def _fmt_seconds(sec: float | None) -> str:
    if sec is None or not math.isfinite(sec):
        return "n/a"
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m):02d}:{s:05.2f} ({sec:.1f}s)"
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f} ({sec:.1f}s)"


def _fmt_pct(part: float | None, whole: float | None) -> str:
    if part is None or whole is None or whole <= 0:
        return "0.0%"
    return f"{(part / whole) * 100:.1f}%"


def _extract_turn_intervals(turns: Iterable[dict[str, Any]]) -> list[tuple[float, float, str]]:
    spans: list[tuple[float, float, str]] = []
    for t in turns:
        try:
            s, e = float(t["start_s"]), float(t["end_s"])
            spk = str(t.get("speaker_id", "unknown")).strip()
            if math.isfinite(s) and math.isfinite(e) and e > s:
                spans.append((s, e, spk))
        except (KeyError, TypeError, ValueError):
            continue
    spans.sort(key=lambda item: (item[0], item[1]))
    return spans


def _timeline_union_duration(intervals: list[tuple[float, float, str]]) -> float:
    if not intervals:
        return 0.0
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = intervals[0][0], intervals[0][1]
    for s, e, _ in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return sum(e - s for s, e in merged)


def _find_stage_manifests(target: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Given a file or directory, resolve (raw_manifest, merged_manifest, final_manifest)."""
    p = target.resolve()
    if p.is_dir():
        raw = p / "segments.raw.json"
        merged = p / "segments.merged.json"
        final = p / "segments.json"
        return (
            raw if raw.is_file() else None,
            merged if merged.is_file() else None,
            final if final.is_file() else None,
        )

    parent = p.parent
    name = p.name
    raw = parent / "segments.raw.json"
    merged = parent / "segments.merged.json"
    final = parent / "segments.json"

    if name == "segments.raw.json":
        return p, (merged if merged.is_file() else None), (final if final.is_file() else None)
    if name == "segments.merged.json":
        return (raw if raw.is_file() else None), p, (final if final.is_file() else None)
    if name == "segments.json":
        return (raw if raw.is_file() else None), (merged if merged.is_file() else None), p

    return None, None, p


def audit_item(
    item_target: Path,
    *,
    audio_override: Path | None = None,
    min_duration_s: float | None = None,
    max_duration_s: float | None = None,
) -> dict[str, Any]:
    """Audit duration loss and yield across stages for one audio item."""
    raw_path, merged_path, final_path = _find_stage_manifests(item_target)
    ref_manifest_path = final_path or merged_path or raw_path
    if ref_manifest_path is None or not ref_manifest_path.is_file():
        raise FileNotFoundError(f"No manifest found at or within {item_target}")

    primary_manifest = read_json(ref_manifest_path)
    stem = primary_manifest.get("stem") or primary_manifest.get("audio_id") or ref_manifest_path.parent.name
    params = primary_manifest.get("parameters") or {}

    min_dur = min_duration_s if min_duration_s is not None else float(params.get("min_duration_s", 1.5))
    max_dur = max_duration_s if max_duration_s is not None else float(params.get("max_duration_s", 15.0))

    # Resolve source audio
    source_audio_path: Path | None = None
    if audio_override is not None and audio_override.exists():
        source_audio_path = audio_override.resolve()
    elif "source" in primary_manifest and primary_manifest["source"].get("path"):
        source_val = primary_manifest["source"]["path"]
        candidate = resolve_stored_path(source_val, base=ref_manifest_path.parent)
        if candidate.is_file():
            source_audio_path = candidate

    raw_audio_duration_s: float | None = None
    sample_rate: int | None = None
    probed = False

    if source_audio_path is not None and source_audio_path.is_file():
        try:
            info = probe(source_audio_path)
            raw_audio_duration_s = float(info["duration_s"])
            sample_rate = int(info["sample_rate"])
            probed = True
        except Exception:
            pass

    if raw_audio_duration_s is None:
        if "duration_s" in primary_manifest:
            raw_audio_duration_s = float(primary_manifest["duration_s"])
        elif "source" in primary_manifest and primary_manifest["source"].get("duration_s"):
            raw_audio_duration_s = float(primary_manifest["source"]["duration_s"])

    # Load manifests if available
    raw_manifest = read_json(raw_path) if raw_path and raw_path.is_file() else None
    merged_manifest = read_json(merged_path) if merged_path and merged_path.is_file() else None
    final_manifest = read_json(final_path) if final_path and final_path.is_file() else None

    # Determine turn arrays with smart fallback
    has_explicit_raw = raw_manifest is not None
    has_explicit_merged = merged_manifest is not None
    has_explicit_final = final_manifest is not None

    raw_turns_data = raw_manifest.get("turns") if raw_manifest else None
    merged_turns_data = merged_manifest.get("turns") if merged_manifest else None
    final_turns_data = final_manifest.get("turns") if final_manifest else None

    # Fallbacks when stage files were not separately produced
    if raw_turns_data is None:
        raw_turns_data = primary_manifest.get("turns", [])
    if merged_turns_data is None:
        merged_turns_data = primary_manifest.get("turns", [])
    if final_turns_data is None:
        final_turns_data = primary_manifest.get("turns", [])

    # Extract turn intervals
    raw_intervals = _extract_turn_intervals(raw_turns_data)
    merged_intervals = _extract_turn_intervals(merged_turns_data)
    final_intervals = _extract_turn_intervals(final_turns_data)

    # 1. Stage: Raw Diarization Turns
    raw_turns_count = len(raw_intervals)
    raw_speech_sum_s = sum(e - s for s, e, _ in raw_intervals)
    raw_speech_union_s = _timeline_union_duration(raw_intervals)
    raw_overlap_s = max(0.0, raw_speech_sum_s - raw_speech_union_s)

    # If raw audio duration was never specified, lower-bound it by turns span
    all_known_turns = raw_intervals or merged_intervals or final_intervals
    if raw_audio_duration_s is None and all_known_turns:
        raw_audio_duration_s = max(e for _, e, _ in all_known_turns)
    elif raw_audio_duration_s is None:
        raw_audio_duration_s = 0.0

    raw_silence_lost_s = max(0.0, raw_audio_duration_s - raw_speech_union_s) if raw_audio_duration_s > 0 else 0.0
    raw_silence_lost_pct = (raw_silence_lost_s / raw_audio_duration_s * 100.0) if raw_audio_duration_s > 0 else 0.0

    # 2. Stage: Merged Turns (before duration filtering)
    merged_turns_count = len(merged_intervals)
    merged_speech_sum_s = sum(e - s for s, e, _ in merged_intervals)
    merged_speech_union_s = _timeline_union_duration(merged_intervals)
    silence_absorbed_in_merge_s = max(0.0, merged_speech_union_s - raw_speech_union_s) if has_explicit_raw else 0.0

    # 3. Stage: Duration Filtering on Merged Turns (< min_duration_s, > max_duration_s)
    merge_stats = primary_manifest.get("merge_statistics") or {}
    merge_attempts = primary_manifest.get("merge_mean_adjustment", {}).get("attempts", [])
    last_attempt = merge_attempts[-1] if merge_attempts else {}

    if has_explicit_merged:
        too_short_intervals = [(s, e, spk) for s, e, spk in merged_intervals if (e - s) < min_dur]
        too_long_intervals = [(s, e, spk) for s, e, spk in merged_intervals if (e - s) > max_dur]
        too_short_count = len(too_short_intervals)
        too_short_duration_s = sum(e - s for s, e, _ in too_short_intervals)
        too_long_count = len(too_long_intervals)
        too_long_duration_s = sum(e - s for s, e, _ in too_long_intervals)
    else:
        too_short_in_final = [(s, e, spk) for s, e, spk in final_intervals if (e - s) < min_dur]
        too_long_in_final = [(s, e, spk) for s, e, spk in final_intervals if (e - s) > max_dur]
        meta_too_short = int(last_attempt.get("too_short") or merge_stats.get("duration_filter_rejections", 0))
        meta_too_long = int(last_attempt.get("too_long", 0))
        too_short_count = meta_too_short if meta_too_short > 0 else len(too_short_in_final)
        too_long_count = meta_too_long if meta_too_long > 0 else len(too_long_in_final)
        too_short_duration_s = sum(e - s for s, e, _ in too_short_in_final)
        too_long_duration_s = sum(e - s for s, e, _ in too_long_in_final)

    too_short_pct_of_raw = (too_short_duration_s / raw_audio_duration_s * 100.0) if raw_audio_duration_s > 0 else 0.0
    too_short_pct_of_merged = (too_short_duration_s / merged_speech_sum_s * 100.0) if merged_speech_sum_s > 0 else 0.0
    too_long_pct_of_raw = (too_long_duration_s / raw_audio_duration_s * 100.0) if raw_audio_duration_s > 0 else 0.0

    # 4. Stage: Final Exported Clips
    final_clips_count = len(final_intervals)
    final_speech_duration_s = sum(e - s for s, e, _ in final_intervals)
    final_speech_union_s = _timeline_union_duration(final_intervals)
    final_mean_duration_s = (final_speech_duration_s / final_clips_count) if final_clips_count > 0 else 0.0

    # Yields
    yield_pct_of_raw = (final_speech_duration_s / raw_audio_duration_s * 100.0) if raw_audio_duration_s > 0 else 0.0
    yield_pct_of_raw_speech = (
        (final_speech_duration_s / raw_speech_union_s * 100.0)
        if raw_speech_union_s > 0
        else (100.0 if final_speech_duration_s == 0 else 0.0)
    )
    yield_pct_of_merged_speech = (
        (final_speech_duration_s / merged_speech_sum_s * 100.0)
        if merged_speech_sum_s > 0
        else (100.0 if final_speech_duration_s == 0 else 0.0)
    )

    # Net loss
    net_lost_s = max(0.0, raw_audio_duration_s - final_speech_duration_s) if raw_audio_duration_s > 0 else 0.0
    net_lost_pct = (net_lost_s / raw_audio_duration_s * 100.0) if raw_audio_duration_s > 0 else 0.0

    return {
        "stem": stem,
        "manifest_dir": persist_path(ref_manifest_path.parent),
        "source_audio": persist_path(source_audio_path) if source_audio_path else None,
        "probed_audio": probed,
        "sample_rate": sample_rate,
        "raw_duration_s": round(raw_audio_duration_s, 4),
        "thresholds": {"min_duration_s": min_dur, "max_duration_s": max_dur},
        "stages": {
            "has_explicit_raw": has_explicit_raw,
            "has_explicit_merged": has_explicit_merged,
            "has_explicit_final": has_explicit_final,
        },
        "raw_diarization": {
            "turns_count": raw_turns_count,
            "speech_sum_s": round(raw_speech_sum_s, 4),
            "speech_union_s": round(raw_speech_union_s, 4),
            "overlap_s": round(raw_overlap_s, 4),
            "silence_lost_s": round(raw_silence_lost_s, 4),
            "silence_lost_pct": round(raw_silence_lost_pct, 2),
        },
        "merged_stage": {
            "turns_count": merged_turns_count,
            "speech_sum_s": round(merged_speech_sum_s, 4),
            "speech_union_s": round(merged_speech_union_s, 4),
            "silence_absorbed_s": round(silence_absorbed_in_merge_s, 4),
        },
        "duration_filtering": {
            "too_short_count": too_short_count,
            "too_short_s": round(too_short_duration_s, 4),
            "too_short_pct_of_raw": round(too_short_pct_of_raw, 2),
            "too_short_pct_of_merged": round(too_short_pct_of_merged, 2),
            "too_long_count": too_long_count,
            "too_long_s": round(too_long_duration_s, 4),
            "too_long_pct_of_raw": round(too_long_pct_of_raw, 2),
        },
        "final_clips": {
            "clips_count": final_clips_count,
            "speech_duration_s": round(final_speech_duration_s, 4),
            "speech_union_s": round(final_speech_union_s, 4),
            "mean_duration_s": round(final_mean_duration_s, 4),
            "yield_pct_of_raw": round(yield_pct_of_raw, 2),
            "yield_pct_of_raw_speech": round(yield_pct_of_raw_speech, 2),
            "yield_pct_of_merged_speech": round(yield_pct_of_merged_speech, 2),
        },
        "net_loss": {
            "total_lost_s": round(net_lost_s, 4),
            "total_lost_pct": round(net_lost_pct, 2),
            "silence_lost_s": round(raw_silence_lost_s, 4),
            "too_short_lost_s": round(too_short_duration_s, 4),
            "too_long_lost_s": round(too_long_duration_s, 4),
        },
    }


def aggregate_audits(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item audits into dataset totals and summary metrics."""
    total_raw_s = sum(i["raw_duration_s"] for i in items)
    total_raw_turns = sum(i["raw_diarization"]["turns_count"] for i in items)
    total_raw_speech_sum_s = sum(i["raw_diarization"]["speech_sum_s"] for i in items)
    total_raw_speech_union_s = sum(i["raw_diarization"]["speech_union_s"] for i in items)
    total_raw_silence_s = sum(i["raw_diarization"]["silence_lost_s"] for i in items)

    total_merged_turns = sum(i["merged_stage"]["turns_count"] for i in items)
    total_merged_speech_sum_s = sum(i["merged_stage"]["speech_sum_s"] for i in items)
    total_merged_speech_union_s = sum(i["merged_stage"]["speech_union_s"] for i in items)
    total_silence_absorbed_s = sum(i["merged_stage"]["silence_absorbed_s"] for i in items)

    total_too_short_count = sum(i["duration_filtering"]["too_short_count"] for i in items)
    total_too_short_s = sum(i["duration_filtering"]["too_short_s"] for i in items)
    total_too_long_count = sum(i["duration_filtering"]["too_long_count"] for i in items)
    total_too_long_s = sum(i["duration_filtering"]["too_long_s"] for i in items)

    total_final_clips = sum(i["final_clips"]["clips_count"] for i in items)
    total_final_speech_s = sum(i["final_clips"]["speech_duration_s"] for i in items)
    total_final_speech_union_s = sum(i["final_clips"]["speech_union_s"] for i in items)

    total_net_lost_s = max(0.0, total_raw_s - total_final_speech_s) if total_raw_s > 0 else 0.0

    return {
        "item_count": len(items),
        "total_raw_duration_s": round(total_raw_s, 4),
        "raw_diarization": {
            "total_turns": total_raw_turns,
            "total_speech_sum_s": round(total_raw_speech_sum_s, 4),
            "total_speech_union_s": round(total_raw_speech_union_s, 4),
            "total_silence_lost_s": round(total_raw_silence_s, 4),
            "silence_lost_pct_of_raw": round((total_raw_silence_s / total_raw_s * 100.0) if total_raw_s > 0 else 0.0, 2),
        },
        "merged_stage": {
            "total_turns": total_merged_turns,
            "total_speech_sum_s": round(total_merged_speech_sum_s, 4),
            "total_speech_union_s": round(total_merged_speech_union_s, 4),
            "total_silence_absorbed_s": round(total_silence_absorbed_s, 4),
        },
        "duration_filtering": {
            "total_too_short_count": total_too_short_count,
            "total_too_short_s": round(total_too_short_s, 4),
            "too_short_pct_of_raw": round((total_too_short_s / total_raw_s * 100.0) if total_raw_s > 0 else 0.0, 2),
            "too_short_pct_of_merged": round((total_too_short_s / total_merged_speech_sum_s * 100.0) if total_merged_speech_sum_s > 0 else 0.0, 2),
            "total_too_long_count": total_too_long_count,
            "total_too_long_s": round(total_too_long_s, 4),
            "too_long_pct_of_raw": round((total_too_long_s / total_raw_s * 100.0) if total_raw_s > 0 else 0.0, 2),
        },
        "final_clips": {
            "total_clips": total_final_clips,
            "total_speech_duration_s": round(total_final_speech_s, 4),
            "total_speech_union_s": round(total_final_speech_union_s, 4),
            "mean_duration_s": round((total_final_speech_s / total_final_clips) if total_final_clips > 0 else 0.0, 4),
            "yield_pct_of_raw": round((total_final_speech_s / total_raw_s * 100.0) if total_raw_s > 0 else 0.0, 2),
            "yield_pct_of_raw_speech": round((total_final_speech_s / total_raw_speech_union_s * 100.0) if total_raw_speech_union_s > 0 else 0.0, 2),
            "yield_pct_of_merged_speech": round((total_final_speech_s / total_merged_speech_sum_s * 100.0) if total_merged_speech_sum_s > 0 else 0.0, 2),
        },
        "net_loss": {
            "total_lost_s": round(total_net_lost_s, 4),
            "total_lost_pct": round((total_net_lost_s / total_raw_s * 100.0) if total_raw_s > 0 else 0.0, 2),
            "total_silence_lost_s": round(total_raw_silence_s, 4),
            "total_too_short_lost_s": round(total_too_short_s, 4),
            "total_too_long_lost_s": round(total_too_long_s, 4),
        },
    }


def render_single_detail(item: dict[str, Any]) -> str:
    lines = [
        "=" * 78,
        f"AUDIO DURATION LOSS AUDIT: {item['stem']}",
        "=" * 78,
        f"Source Audio:       {item['source_audio'] or '(unresolved on disk)'}",
        f"Manifest Directory: {item['manifest_dir']}",
        f"Raw Audio Duration: {_fmt_seconds(item['raw_duration_s'])}",
        "",
        "1. Raw Audio -> Diarization Step (Speech Detection vs Silence):",
        f"   - Raw Speech Union:        {_fmt_seconds(item['raw_diarization']['speech_union_s'])} ({_fmt_pct(item['raw_diarization']['speech_union_s'], item['raw_duration_s'])} of raw audio)",
        f"   - Raw Turns Count:         {item['raw_diarization']['turns_count']} turns (sum: {_fmt_seconds(item['raw_diarization']['speech_sum_s'])}, overlap: {_fmt_seconds(item['raw_diarization']['overlap_s'])})",
        f"   - Lost to Silence/Unvoiced: {_fmt_seconds(item['raw_diarization']['silence_lost_s'])} ({item['raw_diarization']['silence_lost_pct']}%)",
        "",
        "2. Merge & VAD Split Step (Same-Speaker Turn Merging):",
        f"   - Merged Speech:           {_fmt_seconds(item['merged_stage']['speech_sum_s'])} across {item['merged_stage']['turns_count']} turns",
        f"   - Silence Gaps Merged:     {_fmt_seconds(item['merged_stage']['silence_absorbed_s'])} absorbed into speech turns",
        "",
        f"3. Post-Merge Duration Filtering (< {item['thresholds']['min_duration_s']:.1f}s):",
        f"   - Lost to < {item['thresholds']['min_duration_s']:.1f}s Filter:   {_fmt_seconds(item['duration_filtering']['too_short_s'])} ({item['duration_filtering']['too_short_pct_of_raw']}% of raw, {item['duration_filtering']['too_short_pct_of_merged']}% of merged) across {item['duration_filtering']['too_short_count']} turn(s)",
        f"   - Lost to > {item['thresholds']['max_duration_s']:.1f}s Filter:   {_fmt_seconds(item['duration_filtering']['too_long_s'])} across {item['duration_filtering']['too_long_count']} turn(s)",
        "",
        "4. Final Exported Clips:",
        f"   - Valid Clips:             {item['final_clips']['clips_count']} clips",
        f"   - Final Speech Duration:   {_fmt_seconds(item['final_clips']['speech_duration_s'])} (mean: {_fmt_seconds(item['final_clips']['mean_duration_s'])})",
        f"   - Final Yield (of Raw):    {item['final_clips']['yield_pct_of_raw']}%",
        f"   - Final Yield (of Speech): {item['final_clips']['yield_pct_of_raw_speech']}%",
        "",
        "Summary of Loss Breakdown:",
        f"   - Lost to Silence/Non-speech: {_fmt_seconds(item['net_loss']['silence_lost_s'])} ({_fmt_pct(item['net_loss']['silence_lost_s'], item['raw_duration_s'])} of raw)",
        f"   - Lost to < {item['thresholds']['min_duration_s']:.1f}s Filter:     {_fmt_seconds(item['net_loss']['too_short_lost_s'])} ({_fmt_pct(item['net_loss']['too_short_lost_s'], item['raw_duration_s'])} of raw)",
        f"   - Surviving Audio Yield:      {_fmt_seconds(item['final_clips']['speech_duration_s'])} ({item['final_clips']['yield_pct_of_raw']}%)",
        "=" * 78,
    ]
    return "\n".join(lines)


def render_table(items: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = [
        "Item",
        "Raw Dur",
        "Diar Spk",
        "Silence Lost",
        "Merged Spk",
        "Lost <1.5s",
        "Clips",
        "Final Dur",
        "Yield (Raw)",
    ]
    col_widths = [24, 11, 11, 17, 12, 17, 7, 11, 12]

    def fmt_row(row_cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(row_cells):
            w = col_widths[i]
            parts.append(cell.ljust(w) if i == 0 else cell.rjust(w))
        return "  ".join(parts)

    border = "-" * (sum(col_widths) + 2 * (len(col_widths) - 1))
    double_border = "=" * len(border)

    lines = [
        double_border,
        "DURATION LOSS & YIELD SUMMARY ACROSS PIPELINE STAGES",
        double_border,
        fmt_row(headers),
        border,
    ]

    for it in items:
        stem = it["stem"][: col_widths[0]]
        raw_dur = f"{it['raw_duration_s']:.2f}s"
        diar_spk = f"{it['raw_diarization']['speech_union_s']:.2f}s"
        silence_lost = f"{it['raw_diarization']['silence_lost_s']:.2f}s ({it['raw_diarization']['silence_lost_pct']}%)"
        merged_spk = f"{it['merged_stage']['speech_sum_s']:.2f}s"
        lost_short = f"{it['duration_filtering']['too_short_s']:.2f}s ({it['duration_filtering']['too_short_pct_of_raw']}%)"
        clips = str(it["final_clips"]["clips_count"])
        final_dur = f"{it['final_clips']['speech_duration_s']:.2f}s"
        yield_raw = f"{it['final_clips']['yield_pct_of_raw']}%"

        lines.append(fmt_row([stem, raw_dur, diar_spk, silence_lost, merged_spk, lost_short, clips, final_dur, yield_raw]))

    lines.append(border)
    tot_row = [
        f"TOTAL ({summary['item_count']} items)"[: col_widths[0]],
        f"{summary['total_raw_duration_s']:.2f}s",
        f"{summary['raw_diarization']['total_speech_union_s']:.2f}s",
        f"{summary['raw_diarization']['total_silence_lost_s']:.2f}s ({summary['raw_diarization']['silence_lost_pct_of_raw']}%)",
        f"{summary['merged_stage']['total_speech_sum_s']:.2f}s",
        f"{summary['duration_filtering']['total_too_short_s']:.2f}s ({summary['duration_filtering']['too_short_pct_of_raw']}%)",
        str(summary["final_clips"]["total_clips"]),
        f"{summary['final_clips']['total_speech_duration_s']:.2f}s",
        f"{summary['final_clips']['yield_pct_of_raw']}%",
    ]
    lines.append(fmt_row(tot_row))
    lines.append(double_border)

    lines.extend([
        "",
        "AGGREGATE LOSS BREAKDOWN:",
        f"  Total Raw Audio:            {_fmt_seconds(summary['total_raw_duration_s'])} (100.0%)",
        f"  1. Lost to Silence/Unvoiced: {_fmt_seconds(summary['raw_diarization']['total_silence_lost_s'])} ({summary['raw_diarization']['silence_lost_pct_of_raw']}%)",
        f"  2. Lost to < 1.5s Filter:    {_fmt_seconds(summary['duration_filtering']['total_too_short_s'])} ({summary['duration_filtering']['too_short_pct_of_raw']}%) across {summary['duration_filtering']['total_too_short_count']} turn(s)",
        f"  3. Final Surviving Audio:    {_fmt_seconds(summary['final_clips']['total_speech_duration_s'])} ({summary['final_clips']['yield_pct_of_raw']}%) in {summary['final_clips']['total_clips']} clips",
        "",
    ])

    return "\n".join(lines)


def render_csv(items: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fields = [
        "stem",
        "source_audio",
        "raw_duration_s",
        "raw_turns_count",
        "raw_speech_sum_s",
        "raw_speech_union_s",
        "silence_lost_s",
        "silence_lost_pct",
        "merged_turns_count",
        "merged_speech_sum_s",
        "silence_absorbed_in_merge_s",
        "too_short_count",
        "too_short_s",
        "too_short_pct_of_raw",
        "too_short_pct_of_merged",
        "too_long_count",
        "too_long_s",
        "final_clips_count",
        "final_speech_duration_s",
        "final_mean_duration_s",
        "yield_pct_of_raw",
        "yield_pct_of_raw_speech",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for it in items:
        writer.writerow({
            "stem": it["stem"],
            "source_audio": it["source_audio"] or "",
            "raw_duration_s": it["raw_duration_s"],
            "raw_turns_count": it["raw_diarization"]["turns_count"],
            "raw_speech_sum_s": it["raw_diarization"]["speech_sum_s"],
            "raw_speech_union_s": it["raw_diarization"]["speech_union_s"],
            "silence_lost_s": it["raw_diarization"]["silence_lost_s"],
            "silence_lost_pct": it["raw_diarization"]["silence_lost_pct"],
            "merged_turns_count": it["merged_stage"]["turns_count"],
            "merged_speech_sum_s": it["merged_stage"]["speech_sum_s"],
            "silence_absorbed_in_merge_s": it["merged_stage"]["silence_absorbed_s"],
            "too_short_count": it["duration_filtering"]["too_short_count"],
            "too_short_s": it["duration_filtering"]["too_short_s"],
            "too_short_pct_of_raw": it["duration_filtering"]["too_short_pct_of_raw"],
            "too_short_pct_of_merged": it["duration_filtering"]["too_short_pct_of_merged"],
            "too_long_count": it["duration_filtering"]["too_long_count"],
            "too_long_s": it["duration_filtering"]["too_long_s"],
            "final_clips_count": it["final_clips"]["clips_count"],
            "final_speech_duration_s": it["final_clips"]["speech_duration_s"],
            "final_mean_duration_s": it["final_clips"]["mean_duration_s"],
            "yield_pct_of_raw": it["final_clips"]["yield_pct_of_raw"],
            "yield_pct_of_raw_speech": it["final_clips"]["yield_pct_of_raw_speech"],
        })
    return output.getvalue()


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    inputs_grp = p.add_mutually_exclusive_group(required=True)
    inputs_grp.add_argument("-im", "--input-manifest", type=Path, help="Path to segments.json (or stage manifest)")
    inputs_grp.add_argument("-id", "--input-dir", type=Path, help="Directory containing diarization subdirectories with manifests")
    p.add_argument("-i", "-if", "--input-file", type=Path, help="Optional source audio file override")
    p.add_argument("-o", "-of", "--output-file", type=Path, help="Output destination for audit report (.json, .csv, or .txt)")
    p.add_argument("-f", "--format", choices=("table", "json", "csv"), default="table",
                   help="Report format when writing to stdout or text output (default: table)")
    p.add_argument("-min", "--min-duration-s", type=float, default=None,
                   help="Minimum turn duration threshold in seconds (default: manifest parameter or 1.5)")
    p.add_argument("-max", "--max-duration-s", type=float, default=None,
                   help="Maximum turn duration threshold in seconds (default: manifest parameter or 15.0)")
    p.add_argument("-w", "-ow", "--overwrite", action="store_true", help="Overwrite existing output file")
    p.add_argument("-c", "--concurrency", type=positive_int, default=1, help="Parallel worker threads for directory runs")
    p.add_argument("-b", "-bs", "--batch-size", type=positive_int, default=1, help="Batch size for parallel processing")

    args = p.parse_args()

    if args.min_duration_s is not None and (not math.isfinite(args.min_duration_s) or args.min_duration_s < 0):
        p.error("--min-duration-s must be finite and non-negative")
    if args.max_duration_s is not None and (not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0):
        p.error("--max-duration-s must be finite and positive")
    if args.min_duration_s is not None and args.max_duration_s is not None and args.min_duration_s > args.max_duration_s:
        p.error("--min-duration-s cannot exceed --max-duration-s")

    targets: list[Path] = []
    if args.input_manifest is not None:
        manifest_path = args.input_manifest.resolve()
        if not manifest_path.is_file():
            p.error(f"Input manifest not found: {manifest_path}")
        targets.append(manifest_path)
    else:
        input_dir = args.input_dir.resolve()
        if not input_dir.is_dir():
            p.error(f"Input directory not found: {input_dir}")

        manifests = sorted(input_dir.rglob("segments.json"))
        if not manifests:
            manifests = sorted(input_dir.rglob("segments.merged.json"))
        if not manifests:
            manifests = sorted(input_dir.rglob("segments.raw.json"))
        if not manifests:
            p.error(f"No diarization manifests found below: {input_dir}")
        targets.extend(manifests)

    total = len(targets)
    progress("AUDIT_START", f"Auditing duration loss for {total} manifest(s)")

    items: list[dict[str, Any]] = []

    def process_target(t: Path) -> dict[str, Any]:
        return audit_item(
            t,
            audio_override=args.input_file,
            min_duration_s=args.min_duration_s,
            max_duration_s=args.max_duration_s,
        )

    if args.concurrency > 1 and total > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(args.concurrency, total)) as pool:
            for item in pool.map(process_target, targets):
                items.append(item)
    else:
        for idx, t in enumerate(targets, 1):
            items.append(process_target(t))
            if idx == 1 or idx == total or idx % 50 == 0:
                progress("AUDITED", f"{t.parent.name}", current=idx, total=total)

    summary = aggregate_audits(items)
    report_data = {
        "schema_version": 1,
        "operation": "duration_loss_audit",
        "parameters": {
            "min_duration_s": args.min_duration_s,
            "max_duration_s": args.max_duration_s,
        },
        "summary": summary,
        "items": items,
    }

    # Format output
    out_format = args.format
    if args.output_file is not None:
        dest = args.output_file.resolve()
        if dest.exists() and not args.overwrite:
            p.error(f"Destination exists: {dest}; use --overwrite")
        if dest.suffix == ".json":
            out_format = "json"
        elif dest.suffix == ".csv":
            out_format = "csv"

    if out_format == "json":
        text = json.dumps(report_data, indent=2)
    elif out_format == "csv":
        text = render_csv(items)
    else:
        if len(items) == 1:
            text = render_single_detail(items[0])
        else:
            text = render_table(items, summary)

    if args.output_file is not None:
        dest = args.output_file.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if out_format == "json":
            write_json(dest, report_data)
        else:
            dest.write_text(text, encoding="utf-8")
        progress("AUDIT_DONE", f"Saved audit report to {dest.name}")
        print(dest)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
