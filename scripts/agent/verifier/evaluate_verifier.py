"""Evaluate and benchmark verifier predictions against reference teacher verdicts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.files import LoggingArgumentParser, progress, read_json, write_json
from agent.verifier._verdicts import _known_prompts, _validate_verdict


BACKEND_SUFFIX = re.compile(r"_(gemini|hf|vllm|unsloth|endpoint|minicpm|kimi|moss|vibevoice)$")


def _validated_verdict(data: dict[str, Any], known_prompts: dict[str, str]) -> tuple[dict[str, Any], str | None]:
    verdict = data.get("verdict", data)
    parameters = data.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    if data.get("status") not in (None, "success"):
        return {}, "verifier_failed"
    _, error = _validate_verdict(
        verdict,
        parameters.get("prompt"),
        str(data.get("model", "")),
        known_prompts,
    )
    return (verdict if isinstance(verdict, dict) else {}), error


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument("--predictions-dir", type=Path, required=True, help="Directory containing candidate verdict JSON files")
    p.add_argument("--reference-dir", type=Path, required=True, help="Directory containing reference verdict JSON files")
    p.add_argument("--output-file", type=Path, required=True, help="Path to write evaluation results JSON")
    p.add_argument("--title", type=str, default="Verifier Evaluation", help="Optional report title")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing evaluation results JSON")
    p.add_argument("--concurrency", type=int, default=1, help="Number of concurrent workers for loading verdict files")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size of verdict files to read per worker task")
    args = p.parse_args()
    if args.concurrency < 1:
        p.error("--concurrency must be at least 1")
    if args.batch_size < 1:
        p.error("--batch-size must be at least 1")

    dest = args.output_file.resolve()
    if dest.exists() and not args.overwrite:
        p.error(f"Destination exists: {dest}; use --overwrite")

    pred_files = {BACKEND_SUFFIX.sub("", f.stem): f
                  for f in args.predictions_dir.glob("*.json")}
    ref_files = {BACKEND_SUFFIX.sub("", f.stem): f
                 for f in args.reference_dir.glob("*.json")}

    matched_keys = sorted(set(pred_files.keys()) & set(ref_files.keys()))
    if not matched_keys:
        p.error(f"No matching audio keys between {args.predictions_dir} and {args.reference_dir}")

    progress('EVAL_START', f'Evaluating {len(matched_keys)} matched predictions vs references')

    def _load_pair(key: str) -> tuple[str, dict, dict]:
        return key, read_json(pred_files[key]), read_json(ref_files[key])

    def _load_batch(batch_keys: list[str]) -> list[tuple[str, dict, dict]]:
        return [_load_pair(k) for k in batch_keys]

    batches = [matched_keys[i:i + args.batch_size] for i in range(0, len(matched_keys), args.batch_size)]
    records: list[tuple[str, dict, dict]] = []
    if args.concurrency > 1 and len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
            for b_records in pool.map(_load_batch, batches):
                records.extend(b_records)
    else:
        for b in batches:
            records.extend(_load_batch(b))

    tp, fp, tn, fn = 0, 0, 0, 0
    # Positive = reject (defect caught); Negative = pass (clean)
    reasons_breakdown: dict[str, dict[str, int]] = {}
    latencies = []
    invalid_pairs = []
    transcript_statuses: dict[str, int] = {}
    emotion_statuses: dict[str, int] = {}
    known_prompts = _known_prompts()

    for key, pred_data, ref_data in records:
        pred_v, pred_error = _validated_verdict(pred_data, known_prompts)
        ref_v, ref_error = _validated_verdict(ref_data, known_prompts)
        if pred_error or ref_error:
            invalid_pairs.append({"key": key, "prediction_error": pred_error, "reference_error": ref_error})
            continue

        pred_dec = pred_v.get("decision", "uncertain")
        ref_dec = ref_v.get("decision", "uncertain")

        if "_latency_s" in pred_v:
            latencies.append(float(pred_v["_latency_s"]))

        ref_reason = ref_v.get("reason") or "unspecified"

        if ref_dec == "reject":
            if pred_dec == "reject":
                tp += 1  # True Reject
                reasons_breakdown.setdefault(ref_reason, {"caught": 0, "missed": 0})["caught"] += 1
            else:
                fn += 1  # False Accept (bad accept!)
                reasons_breakdown.setdefault(ref_reason, {"caught": 0, "missed": 0})["missed"] += 1
        elif ref_dec == "pass":
            if pred_dec == "pass":
                tn += 1  # True Pass
            else:
                fp += 1  # False Reject (over-rejection)

        ref_transcript = ref_v.get("transcript")
        pred_transcript = pred_v.get("transcript")
        if isinstance(ref_transcript, str) and ref_transcript:
            transcript_status = (
                "missing_prediction"
                if not isinstance(pred_transcript, str) or not pred_transcript
                else "exact_match"
                if pred_transcript == ref_transcript
                else "different"
            )
            transcript_statuses[transcript_status] = transcript_statuses.get(transcript_status, 0) + 1

        ref_emotion = ref_v.get("emotion")
        pred_emotion = pred_v.get("emotion")
        if isinstance(ref_emotion, str) and ref_emotion.strip():
            ref_e = ref_emotion.strip().lower()
            pred_e = pred_emotion.strip().lower() if isinstance(pred_emotion, str) else ""
            emotion_status = (
                "missing_prediction"
                if not pred_e
                else "exact_match"
                if pred_e == ref_e
                else "different"
            )
            emotion_statuses[emotion_status] = emotion_statuses.get(emotion_status, 0) + 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    reject_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    reject_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    reject_f1 = (2 * reject_precision * reject_recall) / (reject_precision + reject_recall) if (reject_precision + reject_recall) > 0 else 0.0
    false_rejection_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    latencies.sort()
    p50_latency = latencies[len(latencies) // 2] if latencies else 0.0
    p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    summary = {
        "schema_version": 1,
        "title": args.title,
        "matched_clips": len(matched_keys),
        "evaluated_clips": total,
        "invalid_pairs": invalid_pairs,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "reject_precision": round(reject_precision, 4),
            "reject_recall": round(reject_recall, 4),
            "reject_f1": round(reject_f1, 4),
            "false_rejection_rate": round(false_rejection_rate, 4),
        },
        "confusion_matrix": {
            "true_rejects_caught": tp,
            "bad_accepts_missed": fn,
            "true_passes_kept": tn,
            "false_rejects_dropped": fp,
        },
        "latency_stats": {
            "mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p50_s": round(p50_latency, 3),
            "p95_s": round(p95_latency, 3),
        },
        "transcripts": {
            "reference_available": sum(transcript_statuses.values()),
            "statuses": transcript_statuses,
            "exact_match_rate": round(
                transcript_statuses.get("exact_match", 0) / sum(transcript_statuses.values()), 4
            ) if transcript_statuses else None,
        },
        "emotions": {
            "reference_available": sum(emotion_statuses.values()),
            "statuses": emotion_statuses,
            "exact_match_rate": round(
                emotion_statuses.get("exact_match", 0) / sum(emotion_statuses.values()), 4
            ) if emotion_statuses else None,
        },
        "reasons_breakdown": reasons_breakdown,
        "predictions_dir": str(args.predictions_dir),
        "reference_dir": str(args.reference_dir),
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, summary)
    progress('EVAL_DONE', f'Accuracy: {accuracy * 100:.2f}%, Defect Recall: {reject_recall * 100:.2f}% -> {dest.name}')
    print(dest)

    # Print clean readable report to stderr
    report = [
        f"\n=== {args.title} ===",
        f"Evaluated: {total}/{len(matched_keys)} matched clips ({len(invalid_pairs)} invalid pairs excluded)",
        f"Accuracy:                 {accuracy * 100:.2f}%",
        f"Defect Recall (Caught):   {reject_recall * 100:.2f}% ({tp}/{tp + fn})",
        f"Bad Accepts (Missed):     {fn}/{tp + fn}",
        f"False Rejection Rate:     {false_rejection_rate * 100:.2f}% ({fp}/{fp + tn})",
        f"Latency p50 / p95:        {p50_latency:.3f}s / {p95_latency:.3f}s",
    ]
    print("\n".join(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
