"""Compare audio verifiers globally or locally per diarization folder.

Evaluates candidate verifiers against a reference teacher (default: Gemini 3.8
Flash Medium), provides per-criterion breakdown (e.g. clipped words, music bleed,
secondary speakers), generates visual plots, and exports detailed conflict cases.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

from _common.files import (  # noqa: E402
    LoggingArgumentParser,
    ROOT,
    progress,
    read_json,
    safe_name,
    write_json,
)

FAILURE_CODES = (
    "clipped_word_start",
    "clipped_word_end",
    "secondary_speaker",
    "overlapping_speech",
    "music_bleed",
    "noisy_reverberant",
    "distorted",
)

SHORT_NAMES = {
    "clipped_word_start": "clip_start",
    "clipped_word_end": "clip_end",
    "secondary_speaker": "2nd_spk",
    "overlapping_speech": "overlap",
    "music_bleed": "music",
    "noisy_reverberant": "noise",
    "distorted": "distort",
}

BACKEND_CLEAN_RE = re.compile(
    r"_(gemini|hf|vllm|unsloth|endpoint|minicpm|kimi|moss|vibevoice)$",
    re.IGNORECASE,
)


def clean_audio_key(stem: str) -> str:
    """Normalize file stem by stripping backend suffixes."""
    return BACKEND_CLEAN_RE.sub("", stem)


def extract_verdict(data: dict[str, Any], file_path: Path) -> dict[str, Any]:
    """Normalize loaded JSON data into a uniform verdict dictionary."""
    verdict = data.get("verdict") if isinstance(data.get("verdict"), dict) else data
    result = dict(verdict)

    # Resolve audio path
    audio_path = None
    if isinstance(data.get("source"), dict):
        audio_path = data["source"].get("path")
    if not audio_path:
        audio_path = result.get("audio_path")
    if not audio_path:
        # Check if sibling audio file exists
        stem_clean = clean_audio_key(file_path.stem)
        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            sibling = file_path.with_name(f"{stem_clean}{ext}")
            if sibling.is_file():
                audio_path = str(sibling)
                break
    result["_resolved_audio_path"] = audio_path or clean_audio_key(file_path.stem)

    # Extract failure codes
    raw_codes = result.get("failure_codes")
    codes: set[str] = set()
    if isinstance(raw_codes, list):
        codes.update(c for c in raw_codes if isinstance(c, str) and c in FAILURE_CODES)

    # Fallback to dimension indicators
    sp = result.get("speaker_purity")
    if sp in ("secondary_speaker", "overlapping_speech"):
        codes.add(sp)
    wc = result.get("word_completeness")
    if wc in ("clipped_word_start", "clipped_word_end"):
        codes.add(wc)
    aq = result.get("audio_quality")
    if aq in ("music_bleed", "noisy_reverberant", "distorted"):
        codes.add(aq)

    result["_normalized_codes"] = sorted(codes)
    result["_decision"] = result.get("decision", "uncertain")
    result["_reason"] = result.get("reason") or ""
    return result


def discover_verifier_runs(base_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover all family folders and verifier backends under base_dir.

    Returns:
        mapping of family_name -> {backend_label: directory_path}
    """
    runs: dict[str, dict[str, Path]] = defaultdict(dict)
    if not base_dir.is_dir():
        return runs

    for root_str, dirs, files in os.walk(base_dir):
        # Skip internal or output directories
        p = Path(root_str)
        if any(part in ("comparisons", "work", "plot", "plots") or part.startswith(".") for part in p.parts):
            continue

        json_files = [f for f in files if f.endswith(".json")]
        if not json_files:
            continue

        rel_parts = p.relative_to(base_dir).parts
        if not rel_parts:
            continue

        # Pattern 1: gemini/<model>/<effort>/<family>
        if rel_parts[0] == "gemini" and len(rel_parts) >= 4:
            model = rel_parts[1]
            effort = rel_parts[2]
            family = rel_parts[3]
            backend_label = f"gemini/{model}/{effort}"
            runs[family][backend_label] = p
        # Pattern 2: <backend>/<family>
        elif len(rel_parts) >= 2:
            backend_label = rel_parts[0]
            family = rel_parts[1]
            runs[family][backend_label] = p
        elif len(rel_parts) == 1:
            # Flat family or backend
            runs[rel_parts[0]][rel_parts[0]] = p

    return runs


def evaluate_pair(
    ref_records: dict[str, dict[str, Any]],
    cand_records: dict[str, dict[str, Any]],
    cand_name: str,
    family_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute benchmark metrics and conflict details between reference and candidate."""
    matched_keys = sorted(set(ref_records.keys()) & set(cand_records.keys()))
    tp, fp, tn, fn = 0, 0, 0, 0
    latencies: list[float] = []

    # Per-criterion stats: {code: {"ref": count, "caught": count, "missed": count, "overcalled": count}}
    code_stats = {
        code: {"ref": 0, "caught": 0, "missed": 0, "overcalled": 0}
        for code in FAILURE_CODES
    }

    conflicts: list[dict[str, Any]] = []

    for key in matched_keys:
        ref_v = ref_records[key]
        cand_v = cand_records[key]

        ref_dec = ref_v["_decision"]
        cand_dec = cand_v["_decision"]

        ref_codes = set(ref_v["_normalized_codes"])
        cand_codes = set(cand_v["_normalized_codes"])

        if "_latency_s" in cand_v and cand_v["_latency_s"] is not None:
            with contextlib_suppress():
                latencies.append(float(cand_v["_latency_s"]))

        # Count reference occurrences
        for code in ref_codes:
            code_stats[code]["ref"] += 1

        # Decision-level confusion matrix
        # Defect = Positive (reject), Clean = Negative (pass)
        if ref_dec == "reject":
            if cand_dec == "reject":
                tp += 1
                # Both reject: check code coverage
                for code in ref_codes:
                    if code in cand_codes:
                        code_stats[code]["caught"] += 1
                    else:
                        code_stats[code]["missed"] += 1
                for code in cand_codes:
                    if code not in ref_codes:
                        code_stats[code]["overcalled"] += 1

                # If codes differ, record as code_mismatch conflict
                if ref_codes != cand_codes:
                    conflicts.append({
                        "key": key,
                        "audio_path": cand_v.get("_resolved_audio_path") or ref_v.get("_resolved_audio_path") or key,
                        "family": family_name,
                        "candidate": cand_name,
                        "conflict_type": "code_mismatch",
                        "ref_decision": ref_dec,
                        "cand_decision": cand_dec,
                        "ref_codes": ";".join(sorted(ref_codes)),
                        "cand_codes": ";".join(sorted(cand_codes)),
                        "missed_codes": ";".join(sorted(ref_codes - cand_codes)),
                        "overcalled_codes": ";".join(sorted(cand_codes - ref_codes)),
                        "ref_reason": ref_v["_reason"],
                        "cand_reason": cand_v["_reason"],
                    })
            else:
                fn += 1  # Bad Accept (missed defect)
                for code in ref_codes:
                    code_stats[code]["missed"] += 1

                conflicts.append({
                    "key": key,
                    "audio_path": cand_v.get("_resolved_audio_path") or ref_v.get("_resolved_audio_path") or key,
                    "family": family_name,
                    "candidate": cand_name,
                    "conflict_type": "bad_accept",
                    "ref_decision": ref_dec,
                    "cand_decision": cand_dec,
                    "ref_codes": ";".join(sorted(ref_codes)),
                    "cand_codes": ";".join(sorted(cand_codes)),
                    "missed_codes": ";".join(sorted(ref_codes)),
                    "overcalled_codes": "",
                    "ref_reason": ref_v["_reason"],
                    "cand_reason": cand_v["_reason"],
                })
        elif ref_dec == "pass":
            if cand_dec == "pass":
                tn += 1
            else:
                fp += 1  # False Reject (over-rejection / lost yield)
                for code in cand_codes:
                    code_stats[code]["overcalled"] += 1

                conflicts.append({
                    "key": key,
                    "audio_path": cand_v.get("_resolved_audio_path") or ref_v.get("_resolved_audio_path") or key,
                    "family": family_name,
                    "candidate": cand_name,
                    "conflict_type": "false_reject",
                    "ref_decision": ref_dec,
                    "cand_decision": cand_dec,
                    "ref_codes": "",
                    "cand_codes": ";".join(sorted(cand_codes)),
                    "missed_codes": "",
                    "overcalled_codes": ";".join(sorted(cand_codes)),
                    "ref_reason": ref_v["_reason"],
                    "cand_reason": cand_v["_reason"],
                })

    total = len(matched_keys)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    reject_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    reject_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    reject_f1 = (2 * reject_precision * reject_recall) / (reject_precision + reject_recall) if (reject_precision + reject_recall) > 0 else 0.0
    false_rejection_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    cand_pass_rate = (tn + fn) / total if total > 0 else 0.0
    ref_pass_rate = (tn + fp) / total if total > 0 else 0.0

    latencies.sort()
    p50_latency = latencies[len(latencies) // 2] if latencies else 0.0
    p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    criterion_metrics = {}
    for code, s in code_stats.items():
        ref_c = s["ref"]
        caught = s["caught"]
        missed = s["missed"]
        overcalled = s["overcalled"]
        recall = (caught / ref_c) if ref_c > 0 else 1.0
        criterion_metrics[code] = {
            "ref_count": ref_c,
            "caught": caught,
            "missed": missed,
            "overcalled": overcalled,
            "recall": round(recall, 4),
        }

    summary = {
        "candidate": cand_name,
        "family": family_name,
        "matched_clips": total,
        "overall": {
            "accuracy": round(accuracy, 4),
            "defect_recall": round(reject_recall, 4),
            "false_rejection_rate": round(false_rejection_rate, 4),
            "reject_precision": round(reject_precision, 4),
            "reject_f1": round(reject_f1, 4),
            "cand_pass_rate": round(cand_pass_rate, 4),
            "ref_pass_rate": round(ref_pass_rate, 4),
        },
        "confusion_matrix": {
            "true_rejects_caught": tp,
            "bad_accepts_missed": fn,
            "true_passes_kept": tn,
            "false_rejects_dropped": fp,
        },
        "latency": {
            "mean_s": round(mean_latency, 3),
            "p50_s": round(p50_latency, 3),
            "p95_s": round(p95_latency, 3),
        },
        "per_criterion": criterion_metrics,
        "conflict_counts": {
            "total": len(conflicts),
            "bad_accepts": sum(1 for c in conflicts if c["conflict_type"] == "bad_accept"),
            "false_rejects": sum(1 for c in conflicts if c["conflict_type"] == "false_reject"),
            "code_mismatch": sum(1 for c in conflicts if c["conflict_type"] == "code_mismatch"),
        },
    }

    return summary, conflicts


class contextlib_suppress:
    """Lightweight exception suppressor."""
    def __enter__(self) -> None:
        pass
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return True


def render_comparison_plots(
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Generate visual comparison plots."""
    if not summaries:
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # 1. Per-Criterion Recall Comparison Plot
    # Shows Defect Recall for each failure code across all candidates
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FAILURE_CODES))
    num_cands = len(summaries)
    bar_width = 0.8 / max(1, num_cands)

    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]

    for i, s in enumerate(summaries):
        cand_name = s["candidate"]
        recalls = [
            s["per_criterion"][code]["recall"] * 100
            for code in FAILURE_CODES
        ]
        color = palette[i % len(palette)]
        offset = (i - (num_cands - 1) / 2) * bar_width
        bars = ax.bar(x + offset, recalls, bar_width, label=cand_name, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.0,
                    f"{h:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    ax.set_ylabel("Defect Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Defect Recall per Acoustic Criterion vs Reference Teacher", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(FAILURE_CODES, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p1 = output_dir / "defect_recall_per_criterion.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    generated_plots.append(p1)

    # 2. Caught vs Missed Defects Breakdown (for the primary or all candidates)
    # Stacked bar chart of Caught vs Missed counts per defect
    for s in summaries:
        cand_name = safe_name(s["candidate"])
        fig, ax = plt.subplots(figsize=(11, 5))
        codes_rev = list(reversed(FAILURE_CODES))
        caught = [s["per_criterion"][c]["caught"] for c in codes_rev]
        missed = [s["per_criterion"][c]["missed"] for c in codes_rev]
        ref_counts = [s["per_criterion"][c]["ref_count"] for c in codes_rev]

        y = np.arange(len(codes_rev))
        h = 0.55

        bars_c = ax.barh(y, caught, h, label="Caught (Bắt trúng)", color="#22c55e", alpha=0.9)
        bars_m = ax.barh(y, missed, h, left=caught, label="Missed (Bắt thiếu / Lọt)", color="#ef4444", alpha=0.9)

        for idx, (c_val, m_val, total) in enumerate(zip(caught, missed, ref_counts)):
            if total > 0:
                label_text = f" {total} total (missed {m_val})"
                ax.text(total + 0.5, idx, label_text, va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(codes_rev, fontsize=10)
        ax.set_xlabel("Number of Clips with Defect", fontsize=11, fontweight="bold")
        ax.set_title(f"Defect Detection: {s['candidate']} vs Teacher", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        p2 = output_dir / f"defects_caught_vs_missed_{cand_name}.png"
        fig.savefig(p2, dpi=180)
        plt.close(fig)
        generated_plots.append(p2)

    # 3. Overall Tradeoff Plot (Yield vs Bad Accepts vs False Rejection)
    fig, ax = plt.subplots(figsize=(10, 5))
    cands = [s["candidate"] for s in summaries]
    y_pos = np.arange(len(cands))
    bar_h = 0.25

    recalls = [s["overall"]["defect_recall"] * 100 for s in summaries]
    false_rejects = [s["overall"]["false_rejection_rate"] * 100 for s in summaries]
    bad_accepts = [
        (s["confusion_matrix"]["bad_accepts_missed"] / max(1, s["matched_clips"])) * 100
        for s in summaries
    ]

    ax.barh(y_pos - bar_h, recalls, bar_h, label="Defect Recall % (Higher better)", color="#10b981")
    ax.barh(y_pos, false_rejects, bar_h, label="False Rejection % (Lost clean yield)", color="#f59e0b")
    ax.barh(y_pos + bar_h, bad_accepts, bar_h, label="Bad Accepts % of Total (Bad audio leak)", color="#ef4444")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cands, fontsize=10, fontweight="bold")
    ax.set_xlabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Verifier Tradeoffs: Recall vs Yield Loss vs Audio Leak", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p3 = output_dir / "tradeoffs_overview.png"
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    generated_plots.append(p3)

    return generated_plots


def export_conflicts_csv(conflicts: list[dict[str, Any]], output_file: Path) -> None:
    """Export conflict records to a tabular CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "key",
        "family",
        "candidate",
        "conflict_type",
        "missed_codes",
        "overcalled_codes",
        "ref_decision",
        "cand_decision",
        "ref_codes",
        "cand_codes",
        "audio_path",
        "ref_reason",
        "cand_reason",
    ]
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in conflicts:
            writer.writerow({k: row.get(k, "") for k in fields})


def export_conflicts_markdown(
    conflicts: list[dict[str, Any]],
    output_file: Path,
    title: str,
) -> None:
    """Export clean human-readable conflict report with clickable links."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    bad_accepts = [c for c in conflicts if c["conflict_type"] == "bad_accept"]
    false_rejects = [c for c in conflicts if c["conflict_type"] == "false_reject"]
    mismatches = [c for c in conflicts if c["conflict_type"] == "code_mismatch"]

    lines = [
        f"# Conflict Cases Report: {title}",
        "",
        f"Total conflict clips: **{len(conflicts)}** "
        f"({len(bad_accepts)} bad accepts, {len(false_rejects)} false rejects, {len(mismatches)} code mismatches).",
        "",
        "---",
        "",
        "## 1. 🚨 Bad Accepts (Bỏ sót lỗi - Nguy cơ lọt rác TTS)",
        "> Reference Teacher báo `REJECT` (clip có khuyết tật âm học), nhưng Candidate báo `PASS`.",
        "",
    ]

    if not bad_accepts:
        lines.append("*Không có trường hợp nào!*")
        lines.append("")
    else:
        lines.extend([
            "| Audio Clip | Model | Bắt thiếu lỗi | Reference Reason (Teacher) | Candidate Reason |",
            "| :--- | :---: | :---: | :--- | :--- |",
        ])
        for c in bad_accepts:
            audio_p = c["audio_path"]
            link = f"[{c['key']}](file://{audio_p})" if audio_p.startswith("/") else c["key"]
            missed = f"`{c['missed_codes']}`" if c["missed_codes"] else "*unspecified*"
            ref_r = c["ref_reason"].replace("|", "\\|").replace("\n", " ")
            cand_r = c["cand_reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {link} | {c['candidate']} | {missed} | {ref_r} | {cand_r} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. ⚠️ False Rejects (Phạt oan - Làm hụt Yield dữ liệu sạch)",
        "> Reference Teacher báo `PASS` (clip sạch đạt chuẩn), nhưng Candidate lại báo `REJECT`.",
        "",
    ])

    if not false_rejects:
        lines.append("*Không có trường hợp nào!*")
        lines.append("")
    else:
        lines.extend([
            "| Audio Clip | Model | Bắt oan lỗi | Reference Reason (Teacher) | Candidate Reason |",
            "| :--- | :---: | :---: | :--- | :--- |",
        ])
        for c in false_rejects:
            audio_p = c["audio_path"]
            link = f"[{c['key']}](file://{audio_p})" if audio_p.startswith("/") else c["key"]
            over = f"`{c['overcalled_codes']}`" if c["overcalled_codes"] else "*unspecified*"
            ref_r = c["ref_reason"].replace("|", "\\|").replace("\n", " ")
            cand_r = c["cand_reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {link} | {c['candidate']} | {over} | {ref_r} | {cand_r} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. 🔄 Code Mismatches (Cả hai đều Reject nhưng lệch mã lỗi)",
        "> Cả hai bên đều đồng thuận loại clip, nhưng gán nhãn acoustic defect khác nhau.",
        "",
    ])

    if not mismatches:
        lines.append("*Không có trường hợp nào!*")
        lines.append("")
    else:
        lines.extend([
            "| Audio Clip | Model | Teacher Codes | Candidate Codes | Teacher Reason | Candidate Reason |",
            "| :--- | :---: | :---: | :---: | :--- | :--- |",
        ])
        for c in mismatches:
            audio_p = c["audio_path"]
            link = f"[{c['key']}](file://{audio_p})" if audio_p.startswith("/") else c["key"]
            ref_c = f"`{c['ref_codes']}`"
            cand_c = f"`{c['cand_codes']}`"
            ref_r = c["ref_reason"].replace("|", "\\|").replace("\n", " ")
            cand_r = c["cand_reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {link} | {c['candidate']} | {ref_c} | {cand_c} | {ref_r} | {cand_r} |")
        lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")


def load_verdicts_from_dir(directory: Path) -> dict[str, dict[str, Any]]:
    """Load all verdict JSON files in directory into a dict mapped by clean audio key."""
    result = {}
    if not directory.is_dir():
        return result
    for f in sorted(directory.glob("*.json")):
        key = clean_audio_key(f.stem)
        try:
            data = read_json(f)
            result[key] = extract_verdict(data, f)
        except Exception:
            continue
    return result


def main() -> int:
    p = LoggingArgumentParser(
        description="Benchmark and compare audio verifiers globally or locally per diarization folder."
    )
    p.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Optional local family/folder name (e.g. 'khanhvy', 'example'). Default: compare all.",
    )
    p.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Explicit reference teacher directory (default: .data/agent/verifier/gemini/gemini-3-8-flash/medium[/<folder>])",
    )
    p.add_argument(
        "--candidates-dir",
        type=Path,
        action="append",
        default=None,
        help="Explicit candidate verifier directory (can specify multiple times)",
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=ROOT / ".data" / "agent" / "verifier",
        help="Base root directory containing verifier runs (default: .data/agent/verifier)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports, plots, and conflict files (default: .data/agent/verifier/comparisons/<folder_or_global>)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Report title",
    )
    args = p.parse_args()

    base_dir = args.base_dir.resolve()
    target_folder = args.folder.strip() if args.folder else None

    # Discover runs
    discovered = discover_verifier_runs(base_dir)

    all_summaries: list[dict[str, Any]] = []
    all_conflicts: list[dict[str, Any]] = []

    # Identify families to evaluate
    if target_folder:
        matched_families = [f for f in discovered if f == target_folder or target_folder in f]
        if not matched_families and not args.reference_dir:
            # Check if target_folder exists directly under base_dir
            candidate_p = base_dir / target_folder
            if candidate_p.is_dir():
                matched_families = [target_folder]
            else:
                p.error(f"Folder '{target_folder}' not found in {base_dir}. Available folders: {list(discovered.keys())}")
        families = matched_families or [target_folder]
    else:
        families = sorted(discovered.keys())
        if not families and not args.reference_dir:
            p.error(f"No verifier runs discovered under {base_dir}")

    dest_folder_name = target_folder or "global"
    output_dir = args.output_dir.resolve() if args.output_dir else base_dir / "comparisons" / dest_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    report_title = args.title or f"Verifier Comparison ({dest_folder_name})"
    progress("COMPARE_START", f"Starting comparison for '{dest_folder_name}' across {len(families)} families")

    # Global aggregation accumulators
    global_ref_records: dict[str, dict[str, Any]] = {}
    global_cand_records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for family in families:
        fam_runs = discovered.get(family, {})

        # Determine reference directory
        if args.reference_dir:
            ref_dir = args.reference_dir.resolve()
        else:
            # Default reference: Gemini 3.8 Flash Medium
            ref_dir = (
                fam_runs.get("gemini/gemini-3-8-flash/medium")
                or fam_runs.get("gemini/gemini-3.8-flash/medium")
                or base_dir / "gemini" / "gemini-3-8-flash" / "medium" / family
            )

        ref_records = load_verdicts_from_dir(ref_dir)
        if not ref_records and target_folder:
            progress("WARN", f"Reference directory has no verdicts: {ref_dir}")
            continue

        # Add to global reference
        for k, v in ref_records.items():
            global_ref_records[f"{family}::{k}"] = v

        # Determine candidate directories
        cand_dirs: dict[str, Path] = {}
        if args.candidates_dir:
            for cd in args.candidates_dir:
                cd_path = cd.resolve()
                cand_dirs[cd_path.name] = cd_path
        else:
            for b_name, b_path in fam_runs.items():
                if "gemini-3-8-flash/medium" in b_name or "gemini-3.8-flash/medium" in b_name:
                    continue  # Skip self comparison
                cand_dirs[b_name] = b_path

        # Evaluate each candidate in this family
        for cand_name, c_path in cand_dirs.items():
            c_records = load_verdicts_from_dir(c_path)
            if not c_records:
                continue

            for k, v in c_records.items():
                global_cand_records[cand_name][f"{family}::{k}"] = v

            summary, conflicts = evaluate_pair(ref_records, c_records, cand_name, family)
            if summary["matched_clips"] > 0:
                all_summaries.append(summary)
                all_conflicts.extend(conflicts)

    # If in global mode or multiple families, also compute Global Aggregation
    if len(families) > 1 and global_ref_records:
        global_summaries = []
        for cand_name, c_dict in global_cand_records.items():
            g_summary, _ = evaluate_pair(global_ref_records, c_dict, cand_name, "global_all")
            if g_summary["matched_clips"] > 0:
                global_summaries.append(g_summary)

    # Render plots
    # If target_folder or single family, plot its summaries. If global, plot global aggregated.
    plots_to_render = global_summaries if (len(families) > 1 and 'global_summaries' in locals() and global_summaries) else all_summaries
    generated_plots = render_comparison_plots(plots_to_render, plots_dir)

    # Export conflicts CSV & Markdown
    conflicts_csv_file = output_dir / "conflicts.csv"
    conflicts_md_file = output_dir / "conflicts.md"
    export_conflicts_csv(all_conflicts, conflicts_csv_file)
    export_conflicts_markdown(all_conflicts, conflicts_md_file, report_title)

    # Export summary JSON
    summary_json_file = output_dir / "summary.json"
    write_json(
        summary_json_file,
        {
            "title": report_title,
            "target": dest_folder_name,
            "families": families,
            "plots": [str(p) for p in generated_plots],
            "conflicts_csv": str(conflicts_csv_file),
            "conflicts_md": str(conflicts_md_file),
            "summaries": all_summaries,
            "global_summaries": global_summaries if 'global_summaries' in locals() else [],
        },
    )

    # Print clean readable report to terminal
    print(f"\n{'=' * 80}", file=sys.stderr)
    print(f"  {report_title.upper()}", file=sys.stderr)
    print(f"{'=' * 80}", file=sys.stderr)
    print(f"Reference Teacher: gemini-3.8-flash (medium)", file=sys.stderr)
    print(f"Output Directory:  {output_dir}", file=sys.stderr)
    print(f"Conflicts CSV:     {conflicts_csv_file}", file=sys.stderr)
    print(f"Conflicts Report:  {conflicts_md_file}", file=sys.stderr)
    if generated_plots:
        print(f"Plots Generated:   {len(generated_plots)} files in {plots_dir}/", file=sys.stderr)
    print(f"{'-' * 80}", file=sys.stderr)

    # 1. Main Leaderboard Table
    print(f"\n[1] OVERALL LEADERBOARD", file=sys.stderr)
    header = f"{'Verifier':<20} | {'Matched':<9} | {'Recall':<8} | {'False Rej':<9} | {'Acc':<6} | {'Bad Accepts':<12} | {'p50 Lat':<8}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    display_summaries = plots_to_render
    for s in display_summaries:
        ov = s["overall"]
        cm = s["confusion_matrix"]
        lat = s["latency"]
        line = (
            f"{s['candidate']:<20} | "
            f"{s['matched_clips']:<9} | "
            f"{ov['defect_recall'] * 100:>6.1f}% | "
            f"{ov['false_rejection_rate'] * 100:>7.1f}% | "
            f"{ov['accuracy'] * 100:>5.1f}% | "
            f"{cm['bad_accepts_missed']:>5}/{cm['true_rejects_caught'] + cm['bad_accepts_missed']:<5} | "
            f"{lat['p50_s']:>6.2f}s"
        )
        print(line, file=sys.stderr)

    # 2. Per-Criterion Breakdown Table (Crucial for TTS Defects like clipped words)
    print(f"\n[2] DEFECT BREAKDOWN: CAUGHT vs MISSED (Lẹm chữ, lẫn giọng, nhạc nền)", file=sys.stderr)
    crit_header = f"{'Verifier':<20} | " + " | ".join(f"{SHORT_NAMES.get(c, c):^10}" for c in FAILURE_CODES)
    print(crit_header, file=sys.stderr)
    print("-" * len(crit_header), file=sys.stderr)

    for s in display_summaries:
        row_items = []
        for code in FAILURE_CODES:
            st = s["per_criterion"][code]
            total_ref = st["ref_count"]
            if total_ref == 0:
                row_items.append(f"{'-':^10}")
            else:
                val_str = f"{st['caught']}/{total_ref}"
                row_items.append(f"{val_str:^10}")
        print(f"{s['candidate']:<20} | " + " | ".join(row_items), file=sys.stderr)

    print(f"\n[3] CONFLICT CASES SUMMARY", file=sys.stderr)
    print(f"- Bad Accepts (Bỏ sót lỗi nguy hiểm): {sum(1 for c in all_conflicts if c['conflict_type'] == 'bad_accept')} clips", file=sys.stderr)
    print(f"- False Rejects (Phạt oan giảm yield):  {sum(1 for c in all_conflicts if c['conflict_type'] == 'false_reject')} clips", file=sys.stderr)
    print(f"- Code Mismatch (Lệch loại khuyết tật): {sum(1 for c in all_conflicts if c['conflict_type'] == 'code_mismatch')} clips", file=sys.stderr)
    print(f"👉 To view individual clips and acoustic reasons, open:\n   {conflicts_md_file}\n", file=sys.stderr)

    progress("COMPARE_DONE", f"Results written to {output_dir}")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
