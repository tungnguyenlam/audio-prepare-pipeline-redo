#!/usr/bin/env python3
"""Audit existing TTS manifests/predictions without inference or external calls."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = (
    "speaker_purity", "overlap", "word_start", "word_end", "music",
    "sound_effect", "noise", "reverberation", "voice_damage",
)


def local_path(value: str) -> Path:
    """Resolve repository paths, including old-machine paths containing .data/."""
    path = Path(value)
    if not path.is_absolute():
        return (ROOT / path).resolve()
    if path.is_file():
        return path.resolve()
    if ".data/" in value:
        return (ROOT / ".data" / value.split(".data/", 1)[1]).resolve()
    return path.resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL rows; reject malformed data rather than skipping it."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    """Write an audit artifact as UTF-8 JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write an audit artifact as UTF-8 JSONL."""
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def exact_upper(bad: int, total: int, alpha: float = 0.05) -> float | None:
    """Return the one-sided Clopper-Pearson upper bound under iid sampling."""
    if total == 0:
        return None
    if bad == total:
        return 1.0
    if bad == 0:
        return -math.expm1(math.log(alpha) / total)
    coefficients = [
        math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1)
        for k in range(bad + 1)
    ]
    low, high = bad / total, 1.0
    for _ in range(64):
        p = (low + high) / 2
        terms = [c + k * math.log(p) + (total - k) * math.log1p(-p)
                 for k, c in enumerate(coefficients)]
        maximum = max(terms)
        log_cdf = maximum + math.log(sum(math.exp(t - maximum) for t in terms))
        if log_cdf > math.log(alpha):
            low = p
        else:
            high = p
    return high


def inspect_audio(path: Path) -> dict[str, Any]:
    """Read audio metadata and exact file hash without modifying the audio."""
    if not path.is_file():
        return {"exists": False, "error": "missing_audio"}
    import soundfile as sf

    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    result: dict[str, Any] = {"exists": True, "sha256": digest}
    try:
        info = sf.info(path)
        result.update(duration_s=info.frames / info.samplerate,
                      sample_rate=info.samplerate, channels=info.channels,
                      frames=info.frames, subtype=info.subtype)
    except (RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
    return result


def inventory(manifests: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inventory existing splits; filename hints never count as verified lineage."""
    rows = []
    cache: dict[Path, dict[str, Any]] = {}
    for split, path in manifests.items():
        for index, item in enumerate(read_jsonl(path), 1):
            audio = local_path(item["audio_path"])
            if audio not in cache:
                cache[audio] = inspect_audio(audio)
            rows.append({
                "split": split, "row": index, "audio_path": str(audio),
                "recording_id": item.get("recording_id"),
                "filename_group_hint": audio.stem.split("_turn_", 1)[0]
                if "_turn_" in audio.stem else None,
                "reference_decision": item.get("decision"),
                "reference_provenance": item.get("label_provenance", "unknown_legacy"),
                **cache[audio],
            })
    overlaps = {}
    for field in ("audio_path", "sha256", "recording_id", "filename_group_hint"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get(field):
                groups[row[field]].append({"split": row["split"], "row": row["row"]})
        overlaps[field] = {key: value for key, value in groups.items()
                           if len({v["split"] for v in value}) > 1}
    summary = {
        "splits": {split: {
            "rows": sum(r["split"] == split for r in rows),
            "labels": dict(Counter(r["reference_decision"] for r in rows if r["split"] == split)),
            "missing_audio": sum(not r["exists"] for r in rows if r["split"] == split),
            "unknown_recording_id": sum(not r["recording_id"] for r in rows if r["split"] == split),
            "outside_2_15_s": sum(not 2 <= r["duration_s"] <= 15 for r in rows
                                    if r["split"] == split and "duration_s" in r),
        } for split in manifests},
        "cross_split_overlap": overlaps,
        "limitations": ["File hash does not detect alternate encodings or overlapping cuts.",
                        "Filename groups flag risk; they do not certify source identity.",
                        "Unknown recording identity prevents disjointness certification."],
    }
    return rows, summary


def summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure acceptance against saved references, never calling them human truth."""
    accepted = [r for r in rows if r["prediction_decision"] == "pass"]
    labeled = [r for r in accepted if r["reference_decision"] in {"pass", "reject"}]
    bad = sum(r["reference_decision"] == "reject" for r in labeled)
    return {
        "candidates": len(rows), "accepted": len(accepted),
        "accepted_with_reference": len(labeled), "accepted_unknown_reference": len(accepted) - len(labeled),
        "accepted_reference_rejects": bad,
        "accepted_reference_error": bad / len(labeled) if labeled else None,
        "human_accepted_error": None,
        "reference_reject_defects": dict(Counter(code for r in labeled
            if r["reference_decision"] == "reject" for code in r["reference_failure_codes"])),
    }


def prepare(args: argparse.Namespace) -> None:
    """Create a new inventory and blind review packet from saved artifacts."""
    output = local_path(args.output)
    if not output.is_relative_to(ROOT / ".data"):
        raise ValueError("Audit output must be under repository .data/.")
    # Never overwrite review decisions or a previous audit.
    output.mkdir(parents=True, exist_ok=False)
    rows, summary = inventory({"train": local_path(args.train), "validation": local_path(args.validation)})
    write_jsonl(output / "inventory.jsonl", rows)
    write_json(output / "inventory_summary.json", summary)
    saved = json.loads(local_path(args.evaluation).read_text())
    candidates = []
    cache = {}
    for result in saved["results"]:
        item = result["item"]
        path = local_path(item.get("audio_path") or item["wav_path"])
        if path not in cache:
            cache[path] = inspect_audio(path)
        ref = item.get("gemini") or item.get("target_json") or {}
        candidates.append({
            "audio_path": str(path), "candidate_id": item.get("id") or item.get("turn_id"),
            "start_s": item.get("start_s"), "end_s": item.get("end_s"),
            "recording_id": item.get("recording_id"),
            "prediction_decision": (result.get("prediction") or {}).get("decision"),
            "reference_decision": ref.get("decision"),
            "reference_failure_codes": ref.get("failure_codes", []),
            "reference_quality_assessed": "audio_quality" in ref,
            "reference_reason": ref.get("reason"), **cache[path],
        })
    all_summary = summarize_candidates(candidates)
    eligible = [r for r in candidates if "duration_s" in r and 2 <= r["duration_s"] <= 15]
    report = {
        "sampling": "legacy_challenge_set_not_random_production",
        "model": saved.get("model"), "adapter_path": saved.get("adapter_path"),
        "all_durations": all_summary, "duration_2_15_s": summarize_candidates(eligible),
        "references_without_audio_quality": sum(not r["reference_quality_assessed"] for r in candidates),
        "missing_audio": sum(not r["exists"] for r in candidates),
        "evaluation_audio_in_training": sum(any(t.get("sha256") == r.get("sha256")
            for t in rows if t["split"] == "train" and t.get("sha256")) for r in candidates),
        "limitations": ["Legacy reference model/reasoning provenance requires fresh MEDIUM evaluation.",
                        "Challenge sampling does not establish representative production risk.",
                        "Legacy references may not assess every production quality dimension."],
    }
    random.Random(42).shuffle(candidates)
    blind = []
    (output / "review_audio").mkdir()
    for index, row in enumerate(candidates, 1):
        review_id = f"clip_{index:04d}"
        row["review_id"] = review_id
        rel = f"review_audio/{review_id}.wav"
        if row["exists"]:
            shutil.copyfile(row["audio_path"], output / rel)
        blind.append({
            "review_id": review_id, "audio_path": rel, "audio_sha256": row.get("sha256"),
            "duration_s": row.get("duration_s"), "rubric_version": "tts-v1",
            "reviewer_id": None, "decision": None,
            "dimensions": dict.fromkeys(DIMENSIONS), "defects": [],
            "context_reviewed": False, "notes": "",
        })
    write_jsonl(output / "blind_review.jsonl", blind)
    write_jsonl(output / "private_reference.jsonl", candidates)
    write_json(output / "baseline.json", report)
    write_json(output / "inputs.json", {
        "train": str(local_path(args.train)), "validation": str(local_path(args.validation)),
        "evaluation": str(local_path(args.evaluation)),
        "input_sha256": {name: hashlib.sha256(local_path(value).read_bytes()).hexdigest()
                         for name, value in vars(args).items()
                         if name in {"train", "validation", "evaluation"}},
    })
    (output / "REVIEW.md").write_text(
        "# Blind challenge review\n\nReview only blind_review.jsonl and review_audio/. "
        "Keep private_reference.jsonl hidden until independent judgments are saved.\n\n"
        "Follow docs/TTS_PHASE1.md. Each dimension is acceptable, defective, or unresolved. "
        "Every pass requires all dimensions acceptable. Reject requires a defect with "
        "a code; optional start_s/end_s and uncertainty_ms are clip-relative. "
        "Do not invent intervals. Leave unknown decisions null or unresolved.\n\n"
        "Save completed/adjudicated rows to a separate JSONL; never overwrite this packet. "
        "The report command requires one final human judgment per review_id. "
        "Context has not been verified or exported by this command.\n\n"
        "This is a challenge set, not representative production sampling.\n"
    )
    print(json.dumps({"output": str(output), "inventory": summary["splits"], "baseline": report}, indent=2))


def label_report(args: argparse.Namespace) -> None:
    """Report ground-truth risk and missing-label bounds on the frozen packet."""
    packet = local_path(args.packet)
    references = {r["review_id"]: r for r in read_jsonl(packet / "private_reference.jsonl")}
    labels = {}
    provenance = Counter()
    for row in read_jsonl(local_path(args.labels)):
        key = row["review_id"]
        if key not in references or key in labels:
            raise ValueError(f"Unknown or duplicate review_id: {key}")
        if row.get("audio_sha256") != references[key].get("sha256"):
            raise ValueError(f"Audio identity mismatch: {key}")
        decision = row.get("decision")
        if decision not in {None, "pass", "reject", "unresolved"}:
            raise ValueError(f"Invalid decision: {key}")
        if decision in {"pass", "reject"}:
            if not row.get("reviewer_id") or row.get("rubric_version") != "tts-v1":
                raise ValueError(f"Missing reviewer or incompatible rubric: {key}")
            dims = row.get("dimensions", {})
            if decision == "pass" and any(dims.get(d) != "acceptable" for d in DIMENSIONS):
                raise ValueError(f"Pass lacks complete dimension assessment: {key}")
            if decision == "reject" and not any(d.get("code") for d in row.get("defects", [])):
                raise ValueError(f"Reject lacks a defect code: {key}")
        labels[key] = decision
        provenance[row.get("label_provenance", "human_review")] += 1
    accepted = [r for r in references.values() if r["prediction_decision"] == "pass"
                and (not args.eligible_only or 2 <= r.get("duration_s", -1) <= 15)]
    known = [r for r in accepted if labels.get(r["review_id"]) in {"pass", "reject"}]
    bad = sum(labels[r["review_id"]] == "reject" for r in known)
    total, unknown = len(accepted), len(accepted) - len(known)
    print(json.dumps({
        "sampling": "legacy_challenge_set_not_random_production", "accepted": total,
        "label_provenance": dict(provenance),
        "labeled_accepted": len(known), "bad_accepted": bad, "unknown_accepted": unknown,
        "observed_risk_among_labeled": bad / len(known) if known else None,
        "all_accepted_risk_lower": bad / total if total else None,
        "all_accepted_risk_upper_missing_as_bad": (bad + unknown) / total if total else None,
        "iid_95_upper_diagnostic": exact_upper(bad, total) if total and not unknown else None,
        "release_qualified": False,
        "caveat": "Challenge sampling and recording dependence prevent production qualification.",
    }, indent=2))


def teacher_audit(args: argparse.Namespace) -> None:
    """Run bounded, resumable MEDIUM ground-truth calls on a blind packet."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")
    if not 1 <= args.limit <= 100:
        raise ValueError("Each invocation must request 1–100 clips.")
    packet = local_path(args.packet)
    rows = read_jsonl(packet / "blind_review.jsonl")[:args.limit]
    output = packet / "gemini_medium"
    output.mkdir(exist_ok=True)
    prompt = (
        "Listen directly to this exact audio candidate for Vietnamese TTS training. "
        "Assess speaker purity (including brief secondary speech/breath/laughter), "
        "overlap, intact first and last spoken words, music, important sound effects, "
        "unacceptable noise/reverb, and processing distortion or damaged voice quality. "
        "Pay attention to 20–100 ms intrusions anywhere, especially the first/last 500 ms. "
        "Do not reject grammatical fragments if acoustically complete. Natural Vietnamese "
        "unreleased final stops, same-speaker breaths, and natural abrupt endings are not "
        "automatically clipping. Do not infer a defect from semantics alone. "
        "Each dimension must be acceptable, defective, or unresolved. Reject if any "
        "important defect is audible; pass only if all dimensions are acceptable. "
        "Use unresolved when evidence is insufficient. Give concise acoustic evidence. "
        "Defect codes: secondary_speaker, overlapping_speech, clipped_word_start, "
        "clipped_word_end, music, sound_effect, excessive_noise, reverberation, "
        "voice_damage. Approximate defect times are seconds within this clip; "
        "use null if not localizable. Never claim millisecond accuracy. "
        "The clip duration policy is checked separately; assess sound regardless of length."
    )
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["pass", "reject", "unresolved"]},
            "dimensions": {"type": "object", "additionalProperties": False,
                "properties": {d: {"type": "string", "enum": ["acceptable", "defective", "unresolved"]}
                               for d in DIMENSIONS}, "required": list(DIMENSIONS)},
            "defects": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "code": {"type": "string", "enum": ["secondary_speaker", "overlapping_speech",
                        "clipped_word_start", "clipped_word_end", "music", "sound_effect",
                        "excessive_noise", "reverberation", "voice_damage"]},
                    "start_s": {"type": ["number", "null"]},
                    "end_s": {"type": ["number", "null"]},
                    "evidence": {"type": "string"}},
                "required": ["code", "start_s", "end_s", "evidence"]}},
            "notes": {"type": "string"}},
        "required": ["decision", "dimensions", "defects", "notes"],
    }
    configuration = {"model": "gemini-3.8-flash", "thinking_level": "MEDIUM",
                     "prompt": prompt, "schema": schema, "max_output_tokens": 8192,
                     "temperature": 0.0, "rubric_version": "tts-v1"}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    decisions = []
    for row in rows:
        audio_path = packet / row["audio_path"]
        audio_bytes = audio_path.read_bytes()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != row["audio_sha256"]:
            raise ValueError(f"Audio changed: {row['review_id']}")
        destination = output / f"{row['review_id']}.json"
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("config_sha256") != config_hash or saved.get("audio_sha256") != audio_hash:
                raise ValueError(f"Existing evaluation has different inputs: {destination}")
            decisions.append(saved)
            continue
        payload = {
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(audio_bytes).decode()}},
                {"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192,
                "thinkingConfig": {"thinkingLevel": "MEDIUM"},
                "responseMimeType": "application/json", "responseJsonSchema": schema},
        }
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not expose request credentials or make unbounded paid retries.
            raise RuntimeError(f"Gemini HTTP {exc.code}; completed calls remain cached.") from None
        write_json(output / f"{row['review_id']}.response.json", {
            "response": raw, "configuration": configuration,
            "config_sha256": config_hash, "audio_sha256": audio_hash,
        })
        candidate = raw.get("candidates", [{}])[0]
        if candidate.get("finishReason") != "STOP":
            raise ValueError(f"Incomplete Gemini response: {row['review_id']}; inspect saved response.")
        response_text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
                                if not p.get("thought"))
        label = json.loads(response_text)
        dims = label.get("dimensions", {})
        if set(dims) != set(DIMENSIONS) or any(v not in {"acceptable", "defective", "unresolved"} for v in dims.values()):
            raise ValueError(f"Incomplete dimensions: {row['review_id']}")
        decision = label.get("decision")
        if decision not in {"pass", "reject", "unresolved"}:
            raise ValueError(f"Invalid decision: {row['review_id']}")
        if decision == "pass" and (any(v != "acceptable" for v in dims.values()) or label.get("defects")):
            raise ValueError(f"Contradictory pass: {row['review_id']}")
        if decision == "reject" and not label.get("defects"):
            raise ValueError(f"Reject without defects: {row['review_id']}")
        saved = {
            **label, "review_id": row["review_id"], "audio_sha256": audio_hash,
            "label_provenance": "user_accepted_gemini_ground_truth",
            "reviewer_id": "gemini-3.8-flash:MEDIUM", "rubric_version": "tts-v1",
            "config_sha256": config_hash, "configuration": configuration,
            "response_model_version": raw.get("modelVersion"),
            "usage": raw.get("usageMetadata"), "context_reviewed": False,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "latency_s": round(time.monotonic() - started, 3),
        }
        write_json(destination, saved)
        decisions.append(saved)
        print(f"{row['review_id']}: {decision} ({saved['latency_s']}s)", flush=True)
    write_jsonl(output / "labels.jsonl", decisions)
    print(json.dumps({"completed": len(decisions),
                      "decisions": dict(Counter(r["decision"] for r in decisions))}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--train", default=".data/distillation/train_v3.jsonl")
    prepare_parser.add_argument("--validation", default=".data/distillation/val_v3.jsonl")
    prepare_parser.add_argument("--evaluation", default=".data/distillation/reports/finetuned_e2b_v3_eval.json")
    prepare_parser.add_argument("--output", required=True, help="New directory under .data/; must not exist")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--packet", required=True)
    report_parser.add_argument("--labels", required=True)
    report_parser.add_argument("--eligible-only", action="store_true")
    teacher_parser = commands.add_parser("teacher", help="User-authorized Gemini MEDIUM ground-truth audit")
    teacher_parser.add_argument("--packet", required=True)
    teacher_parser.add_argument("--limit", type=int, default=31)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "report":
        label_report(args)
    else:
        teacher_audit(args)


if __name__ == "__main__":
    main()
