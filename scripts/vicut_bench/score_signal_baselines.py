#!/usr/bin/env python3
"""Score deterministic signal baselines on ViCut-Bench defects and clean anchors.

Measures baseline physical signal properties without neural models or API calls:
  1. Edge energy & boundary discontinuity:
     - End edge ratio (RMS of last 10ms / active RMS)
     - Start edge ratio (RMS of first 10ms / active RMS)
     - Boundary step jump at offset (|audio[-1]|) and onset (|audio[0]|)
  2. Noise floor & dynamic range (music detection baseline):
     - Peak-to-floor envelope range (99th - 5th percentile dB envelope)
     - Minimum frame energy across 50ms windows
  3. Spectral & energy instability (secondary speaker baseline):
     - Variance of 250ms window RMS across the clip
     - High-frequency flux variation

Outputs detection rates across severities and false-reject rate on clean anchors.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, persist_path, progress, resolve_stored_path, write_json

logger = logging.getLogger("vicut_bench.signal_baselines")

FRAME_S = 0.005
HOP_S = 0.001


def calculate_active_rms(audio: np.ndarray, threshold_db: float = -45.0) -> float:
    if len(audio) == 0:
        return 1e-9
    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))
    if peak <= 1e-7:
        return 1e-9
    threshold = peak * (10.0 ** (threshold_db / 20.0))
    active = audio[abs_audio >= threshold]
    if len(active) == 0:
        active = audio
    return float(np.sqrt(np.mean(active**2)))


def extract_signal_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    if len(audio) == 0:
        return {
            "end_edge_ratio_10ms": 0.0,
            "end_edge_ratio_20ms": 0.0,
            "start_edge_ratio_10ms": 0.0,
            "start_edge_ratio_20ms": 0.0,
            "boundary_step_end": 0.0,
            "boundary_step_start": 0.0,
            "snr_envelope_db": 0.0,
            "min_window_rms_db": -100.0,
            "window_rms_var": 0.0,
        }

    active_rms = max(calculate_active_rms(audio), 1e-9)

    # 1. Edge energy (last / first 10ms and 20ms)
    n_10ms = max(1, int(round(0.010 * sr)))
    n_20ms = max(1, int(round(0.020 * sr)))

    end_10ms = audio[-n_10ms:]
    end_20ms = audio[-n_20ms:]
    start_10ms = audio[:n_10ms]
    start_20ms = audio[:n_20ms]

    end_rms_10 = float(np.sqrt(np.mean(end_10ms**2)))
    end_rms_20 = float(np.sqrt(np.mean(end_20ms**2)))
    start_rms_10 = float(np.sqrt(np.mean(start_10ms**2)))
    start_rms_20 = float(np.sqrt(np.mean(start_20ms**2)))

    end_ratio_10 = end_rms_10 / active_rms
    end_ratio_20 = end_rms_20 / active_rms
    start_ratio_10 = start_rms_10 / active_rms
    start_ratio_20 = start_rms_20 / active_rms

    # Boundary discontinuity jump (normalized to active RMS)
    boundary_step_end = float(abs(audio[-1])) / active_rms
    boundary_step_start = float(abs(audio[0])) / active_rms

    # 2. Envelope and dynamic range
    frame_len = max(1, round(FRAME_S * sr))
    hop_len = max(1, round(HOP_S * sr))
    if len(audio) >= frame_len:
        padded = np.pad(audio, (frame_len // 2, frame_len - frame_len // 2 - 1))
        windows = np.lib.stride_tricks.sliding_window_view(padded, frame_len)[::hop_len]
        env_rms = np.sqrt(np.mean(windows**2, axis=1))
        env_db = 20 * np.log10(np.maximum(env_rms, 1e-6))
        snr_envelope_db = float(np.percentile(env_db, 99) - np.percentile(env_db, 5))
    else:
        snr_envelope_db = 0.0

    # 3. Minimum window energy across 50ms windows (sensitive to music bed filling gaps)
    win_50ms = max(1, int(round(0.050 * sr)))
    hop_25ms = max(1, int(round(0.025 * sr)))
    if len(audio) >= win_50ms:
        sub_wins = np.lib.stride_tricks.sliding_window_view(audio, win_50ms)[::hop_25ms]
        sub_rms = np.sqrt(np.mean(sub_wins**2, axis=1))
        min_window_rms_db = float(20 * np.log10(max(np.min(sub_rms), 1e-6)))
        # Window variance across 250ms chunks (detects secondary burst)
        win_250ms = max(1, int(round(0.250 * sr)))
        hop_125ms = max(1, int(round(0.125 * sr)))
        if len(audio) >= win_250ms:
            macro_wins = np.lib.stride_tricks.sliding_window_view(audio, win_250ms)[::hop_125ms]
            macro_rms = np.sqrt(np.mean(macro_wins**2, axis=1))
            window_rms_var = float(np.var(macro_rms / active_rms))
        else:
            window_rms_var = 0.0
    else:
        min_window_rms_db = -100.0
        window_rms_var = 0.0

    return {
        "end_edge_ratio_10ms": round(end_ratio_10, 4),
        "end_edge_ratio_20ms": round(end_ratio_20, 4),
        "start_edge_ratio_10ms": round(start_ratio_10, 4),
        "start_edge_ratio_20ms": round(start_ratio_20, 4),
        "boundary_step_end": round(boundary_step_end, 4),
        "boundary_step_start": round(boundary_step_start, 4),
        "snr_envelope_db": round(snr_envelope_db, 2),
        "min_window_rms_db": round(min_window_rms_db, 2),
        "window_rms_var": round(window_rms_var, 6),
    }


def compute_auroc(pos_scores: list[float], neg_scores: list[float], higher_is_pos: bool = True) -> float:
    """Compute AUROC using Mann-Whitney U test formula."""
    if not pos_scores or not neg_scores:
        return 0.5
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    if not higher_is_pos:
        pos = -pos
        neg = -neg
    n_pos, n_neg = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    pos_ranks = ranks[:n_pos]
    u = np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def evaluate_detection(
    defects: list[dict],
    anchors: list[dict],
    defect_type: str,
    feature_name: str,
    higher_is_pos: bool = True,
) -> dict[str, Any]:
    # Extract negative scores from clean anchors
    anchor_scores = [a["features"][feature_name] for a in anchors]

    # Threshold for 5% and 10% false reject rates on clean anchors
    if higher_is_pos:
        thresh_5pct_frr = float(np.percentile(anchor_scores, 95))
        thresh_10pct_frr = float(np.percentile(anchor_scores, 90))
    else:
        thresh_5pct_frr = float(np.percentile(anchor_scores, 5))
        thresh_10pct_frr = float(np.percentile(anchor_scores, 10))

    # Group defects by severity
    type_defects = [d for d in defects if d.get("defect_type") == defect_type]
    severities = sorted(list({str(d.get("severity")) for d in type_defects}))

    overall_pos_scores = [d["features"][feature_name] for d in type_defects]
    overall_auroc = compute_auroc(overall_pos_scores, anchor_scores, higher_is_pos=higher_is_pos)

    per_severity_results = {}
    for sev in severities:
        sev_clips = [d for d in type_defects if str(d.get("severity")) == sev]
        scores = [d["features"][feature_name] for d in sev_clips]
        auroc = compute_auroc(scores, anchor_scores, higher_is_pos=higher_is_pos)

        if higher_is_pos:
            det_5pct = float(np.mean([s >= thresh_5pct_frr for s in scores]))
            det_10pct = float(np.mean([s >= thresh_10pct_frr for s in scores]))
        else:
            det_5pct = float(np.mean([s <= thresh_5pct_frr for s in scores]))
            det_10pct = float(np.mean([s <= thresh_10pct_frr for s in scores]))

        per_severity_results[sev] = {
            "n": len(scores),
            "auroc": round(auroc, 4),
            "detection_rate_at_5pct_frr": round(det_5pct, 4),
            "detection_rate_at_10pct_frr": round(det_10pct, 4),
            "mean_score": round(float(np.mean(scores)), 4),
        }

    return {
        "defect_type": defect_type,
        "feature": feature_name,
        "overall_auroc": round(overall_auroc, 4),
        "thresh_at_5pct_frr": round(thresh_5pct_frr, 4),
        "thresh_at_10pct_frr": round(thresh_10pct_frr, 4),
        "severities": per_severity_results,
    }


def main() -> int:
    parser = LoggingArgumentParser(description=__doc__)
    parser.add_argument(
        "--defects-manifest",
        type=Path,
        default=Path(".data/vicut_bench/defects/vicut_bench_defects.jsonl"),
        help="Path to ViCut-Bench defects jsonl.",
    )
    parser.add_argument(
        "--anchors-manifest",
        type=Path,
        default=Path(".data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl"),
        help="Path to clean anchors jsonl.",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        type=Path,
        default=Path(".data/vicut_bench/reports"),
        help="Directory to store evaluation reports.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.output_dir / "signal_baselines_report.json"
    report_md = args.output_dir / "SIGNAL_BASELINES_REPORT.md"

    logger.info("Loading anchors from %s...", args.anchors_manifest)
    anchors = [json.loads(line) for line in args.anchors_manifest.read_text().splitlines() if line.strip()]

    logger.info("Extracting signal features for %d clean anchors...", len(anchors))
    for idx, a in enumerate(anchors, 1):
        if idx % 50 == 0 or idx == len(anchors):
            progress("ANCHORS", f"{idx}/{len(anchors)}", current=idx, total=len(anchors))
        raw_path = a.get("audio_path") or a.get("rel_path")
        audio_path = Path(raw_path)
        if not audio_path.is_file():
            audio_path = Path(".data/vicut_bench/anchors") / raw_path
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        a["features"] = extract_signal_features(data, sr)

    logger.info("Loading defects from %s...", args.defects_manifest)
    defects = [json.loads(line) for line in args.defects_manifest.read_text().splitlines() if line.strip()]

    logger.info("Extracting signal features for %d synthesized defect clips...", len(defects))
    for idx, d in enumerate(defects, 1):
        if idx % 500 == 0 or idx == len(defects):
            progress("DEFECTS", f"{idx}/{len(defects)}", current=idx, total=len(defects))
        raw_path = d.get("audio_path")
        audio_path = Path(raw_path)
        if not audio_path.is_file():
            audio_path = Path(".data/vicut_bench/defects") / raw_path
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        d["features"] = extract_signal_features(data, sr)

    # Evaluate best matched signal baseline for each defect type
    eval_specs = [
        ("clipped_word_end", "end_edge_ratio_10ms", True),
        ("clipped_word_end", "boundary_step_end", True),
        ("clipped_word_start", "start_edge_ratio_10ms", True),
        ("clipped_word_start", "boundary_step_start", True),
        ("background_music", "min_window_rms_db", True),
        ("background_music", "snr_envelope_db", False),
        ("secondary_speaker", "window_rms_var", True),
    ]

    benchmark_results = {}
    for defect_type, feature, higher_is_pos in eval_specs:
        key = f"{defect_type}__{feature}"
        logger.info("Evaluating baseline: %s (feature: %s)...", defect_type, feature)
        benchmark_results[key] = evaluate_detection(
            defects, anchors, defect_type, feature, higher_is_pos=higher_is_pos
        )

    write_json(report_json, benchmark_results)

    # Build Markdown Report
    lines = [
        "# ViCut-Bench Deterministic Signal Baselines Report",
        "",
        f"- Analyzed **{len(anchors)} clean anchors** and **{len(defects)} defect clips**.",
        f"- Full report JSON: `{persist_path(report_json)}`",
        "",
        "## 1. Summary of Baseline Detection Performance",
        "",
        "| Defect Type | Baseline Signal Feature | Overall AUROC | Det @ 5% FRR (Mildest) | Det @ 5% FRR (Most Severe) |",
        "|---|---|---|---|---|",
    ]

    for key, res in benchmark_results.items():
        dtype = res["defect_type"]
        feat = res["feature"]
        auroc = res["overall_auroc"]
        sevs = list(res["severities"].keys())
        mildest = sevs[0]
        severest = sevs[-1]
        det_mild = res["severities"][mildest]["detection_rate_at_5pct_frr"] * 100.0
        det_sev = res["severities"][severest]["detection_rate_at_5pct_frr"] * 100.0
        lines.append(
            f"| `{dtype}` | `{feat}` | **{auroc:.4f}** | {det_mild:.1f} % ({mildest}) | {det_sev:.1f} % ({severest}) |"
        )

    lines.extend([
        "",
        "## 2. Granular Detection vs Severity Breakdown",
        "",
    ])

    for key, res in benchmark_results.items():
        dtype = res["defect_type"]
        feat = res["feature"]
        lines.append(f"### `{dtype}` with `{feat}` (Overall AUROC: {res['overall_auroc']:.4f})")
        lines.append("")
        lines.append("| Severity | AUROC | Det @ 5% FRR | Det @ 10% FRR | Mean Feature Score |")
        lines.append("|---|---|---|---|---|")
        for sev, sdata in res["severities"].items():
            lines.append(
                f"| {sev} | {sdata['auroc']:.4f} | {sdata['detection_rate_at_5pct_frr']*100:.1f} % | {sdata['detection_rate_at_10pct_frr']*100:.1f} % | {sdata['mean_score']:.4f} |"
            )
        lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved report to %s and %s", report_json, report_md)
    print(str(report_md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
