#!/usr/bin/env python3
"""Audit TTS manifests and evaluate boundary repairs against source audio."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import shutil
import sys
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
    saved = ({"results": [{"item": r, "prediction": {}} for r in read_jsonl(local_path(args.candidates))]}
             if args.candidates else json.loads(local_path(args.evaluation).read_text()))
    candidates = []
    cache = {}
    for result in saved["results"]:
        item = result["item"]
        path = local_path(item.get("audio_path") or item["wav_path"])
        if path not in cache:
            cache[path] = inspect_audio(path)
        ref = item.get("gemini") or item.get("target_json") or {}
        candidates.append({
            "audio_path": str(path), "candidate_id": item.get("id") or item.get("turn_id") or path.stem,
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
        "sampling": "legacy_training_pool_not_random_production" if args.candidates else "legacy_challenge_set_not_random_production",
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
        "candidates": str(local_path(args.candidates)) if args.candidates else None,
        "input_sha256": {name: hashlib.sha256(local_path(value).read_bytes()).hexdigest()
                         for name, value in vars(args).items()
                         if name in {"train", "validation", "evaluation", "candidates"} and value},
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


def restore_lineage(args: argparse.Namespace) -> None:
    """Recover legacy recording IDs from manifests and documented generators."""
    output = local_path(args.output)
    if not output.is_relative_to(ROOT / ".data"):
        raise ValueError("Output must be under .data/.")
    output.mkdir(parents=True, exist_ok=False)
    crawled = json.loads((ROOT / ".data/crawled/crawled_manifest.json").read_text())
    audited = json.loads((ROOT / ".data/distillation_e2b/audit_manifest.json").read_text())
    audited_by_path = {local_path(r["path"]): r["video_id"] for r in audited}
    prefix_ids: dict[str, set[str]] = defaultdict(set)
    for row in crawled:
        prefix_ids[Path(row["path"]).stem[:25]].add(row["id"])
    parent_dirs = [ROOT / ".data/distillation" / p for p in
                   ("audio", "extended/audio", "haveasip/audio", "crawled_cuts", "vietcetera_cuts")]

    def identify(path: Path) -> tuple[str | None, str]:
        relative = path.relative_to(ROOT).as_posix()
        if path in audited_by_path:
            return "youtube:" + audited_by_path[path], "distillation_e2b/audit_manifest.json:path+video_id"
        if relative.startswith((".data/distillation/crawled_cuts/", ".data/distillation/vietcetera_cuts/")):
            prefix = path.stem.split("_turn_", 1)[0]
            ids = prefix_ids.get(prefix, set())
            if len(ids) == 1:
                return "youtube:" + next(iter(ids)), "unique crawled manifest prefix; build_distillation_dataset slice uses stem[:25]"
        if relative.startswith((".data/distillation/audio/", ".data/distillation/extended/audio/")):
            return "youtube:H0VpjeULCck", "archived generate_distillation_dataset/generate_extended_distillation_data + compare_verifiers_khanhvy source"
        if relative.startswith(".data/distillation/haveasip/audio/"):
            return None, "local HaveASip source family known; original recording/video ID missing"
        if relative.startswith(".data/distillation/augmented_audio/"):
            for suffix in ("_synth_clipped_start", "_synth_clipped_end"):
                if path.stem.endswith(suffix):
                    base_name = path.stem[:-len(suffix)] + ".wav"
                    parents = [directory / base_name for directory in parent_dirs if (directory / base_name).is_file()]
                    identities = [identify(parent) for parent in parents]
                    ids = {identity for identity, _ in identities}
                    if len(ids) == 1 and None not in ids:
                        return next(iter(ids)), "augmentation inherits existing parent: " + ",".join(str(p.relative_to(ROOT)) for p in parents)
                    return None, "missing/ambiguous/unresolved augmentation parent recording"
        return None, "unresolved lineage"

    known, quarantine = [], []
    seen: dict[str, str | None] = {}
    for manifest in args.manifests:
        for row in read_jsonl(local_path(manifest)):
            path = local_path(row["audio_path"])
            identity, evidence = identify(path)
            audio = inspect_audio(path)
            restored = {**row, "audio_path": str(path.relative_to(ROOT)),
                        "recording_id": identity, "lineage_evidence": evidence,
                        "audio_sha256": audio.get("sha256"), "duration_s": audio.get("duration_s"),
                        "label_provenance": "legacy_not_reauthenticated_medium"}
            if not identity or not audio.get("sha256"):
                quarantine.append(restored)
                continue
            digest = audio["sha256"]
            if digest in seen:
                if seen[digest] != identity:
                    # Shared intros/reused audio can cross otherwise distinct recordings.
                    previous = [r for r in known if r["audio_sha256"] == digest]
                    for duplicate in previous:
                        duplicate["quarantine_reason"] = "duplicate_audio_across_recordings"
                        quarantine.append(duplicate)
                        known.remove(duplicate)
                    seen[digest] = None
                    restored["quarantine_reason"] = "duplicate_audio_across_recordings"
                else:
                    restored["quarantine_reason"] = "duplicate_audio_bytes"
                quarantine.append(restored)
                continue
            seen[digest] = identity
            known.append(restored)
    # Challenge recording is permanently reserved, never presented as unseen.
    challenge = [r for r in known if r["recording_id"] == "youtube:H0VpjeULCck"]
    eligible = [r for r in known if r not in challenge and 2 <= r["duration_s"] <= 15]
    outside = [r for r in known if r not in challenge and not 2 <= r["duration_s"] <= 15]
    for name, rows in (("known_lineage", known), ("quarantine", quarantine),
                       ("reserved_challenge_recording", challenge), ("eligible_pool", eligible),
                       ("outside_duration", outside)):
        write_jsonl(output / f"{name}.jsonl", rows)
    summary = {
        "known_unique": len(known), "quarantined": len(quarantine),
        "reserved_challenge_recording": len(challenge), "eligible_pool": len(eligible),
        "outside_duration": len(outside),
        "eligible_recordings": dict(Counter(r["recording_id"] for r in eligible)),
        "quarantine_reasons": dict(Counter(r.get("quarantine_reason", r["lineage_evidence"]) for r in quarantine)),
        "qualification": "Lineage recovery only; legacy labels are not fresh full-rubric MEDIUM ground truth.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def export_labels(args: argparse.Namespace) -> None:
    """Join completed ground truth to source-resolved candidate identities."""
    packet = local_path(args.packet)
    references = {r["review_id"]: r for r in read_jsonl(packet / "private_reference.jsonl")}
    labels = read_jsonl(packet / "gemini_medium/labels.jsonl")
    output = local_path(args.output)
    if not output.is_relative_to(ROOT / ".data") or output.exists():
        raise ValueError("Choose a new JSONL under .data/.")
    exported = []
    for label in labels:
        ref = references[label["review_id"]]
        if not ref.get("recording_id"):
            raise ValueError("Resolve recording IDs before exporting training examples.")
        if label["audio_sha256"] != ref["sha256"]:
            raise ValueError("Label/audio identity mismatch.")
        if label["decision"] not in {"pass", "reject"}:
            continue
        path = local_path(ref["audio_path"])
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != label["audio_sha256"]:
                raise ValueError("Original audio has changed since annotation.")
        exported.append({
            "audio_path": str(path.relative_to(ROOT)), "audio_sha256": ref["sha256"],
            "recording_id": ref["recording_id"], "duration_s": ref["duration_s"],
            "decision": label["decision"], "dimensions": label["dimensions"],
            "defects": label["defects"], "label_provenance": label["label_provenance"],
            "evaluator": label["reviewer_id"], "rubric_version": label["rubric_version"],
            "evaluation_config_sha256": label["config_sha256"],
            "evaluation_path": str((packet / "gemini_medium" / f"{label['review_id']}.json").relative_to(ROOT)),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, exported)
    print(json.dumps({"exported": len(exported), "output": str(output),
                      "recordings": dict(Counter(r["recording_id"] for r in exported)),
                      "decisions": dict(Counter(r["decision"] for r in exported))}, indent=2))


def speaker_id_from_candidate(candidate_id: str | None) -> str:
    """Recover a speaker label from legacy turn filenames; unknown stays explicit."""
    parts = (candidate_id or "").split("_")
    for index, part in enumerate(parts[:-1]):
        if part == "spk":
            return f"spk_{parts[index + 1]}"
    return "spk_unknown"


def exact_start_frame(source, cut) -> int:
    """Return the first sample index where cut is a byte-identical crop of source."""
    import numpy as np

    source = np.asarray(source)
    cut = np.asarray(cut)
    if source.ndim != 1 or cut.ndim != 1 or source.dtype != cut.dtype:
        raise ValueError("Source and cut must be mono arrays with the same dtype.")
    if len(cut) > len(source):
        raise ValueError("Cut is longer than the source recording.")
    width = source.dtype.itemsize
    needle = cut[: min(256, len(cut))].tobytes()
    blob = source.tobytes()
    cursor = 0
    while True:
        index = blob.find(needle, cursor)
        if index < 0:
            raise ValueError("Cut is not an exact crop of the source.")
        if index % width == 0:
            frame = index // width
            if frame + len(cut) <= len(source) and (source[frame:frame + len(cut)] == cut).all():
                return int(frame)
        cursor = index + width


def competitor_intervals(turns) -> dict[str, list[tuple[float, float]]]:
    """Map each speaker to other speakers' intervals from the same candidate set."""
    intervals = {}
    for turn in turns:
        intervals[turn.speaker_id] = [
            (other.start_s, other.end_s) for other in turns
            if other.speaker_id != turn.speaker_id
        ]
    return intervals


def evaluate_boundaries(args: argparse.Namespace) -> None:
    """Locate legacy cuts in source audio, then relock/split with public APIs."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("HF_HOME", str(ROOT / ".data/huggingface"))
    import soundfile as sf
    from src.diarization.schemas import SpeakerTurn
    from src.diarization.zero_contamination import (
        align_and_lock_syllable_boundaries,
        smart_segment_speaker_turns,
    )
    from src.utils.AudioClass import Audio
    from src.utils.AudioCutter import AudioCutter

    packet = local_path(args.packet)
    source_path = local_path(args.source)
    output = local_path(args.output)
    if not output.is_relative_to(ROOT / ".data") or output.exists():
        raise ValueError("Choose a new directory under .data/.")
    output.mkdir(parents=True)
    references = read_jsonl(packet / "private_reference.jsonl")
    labels = {}
    label_path = packet / "gemini_medium/labels.jsonl"
    if label_path.is_file():
        labels = {row["review_id"]: row for row in read_jsonl(label_path)}
    source, sample_rate = sf.read(source_path, dtype="int16", always_2d=False)
    if getattr(source, "ndim", 1) > 1:
        source = source[:, 0]
    locations = []
    turns = []
    for ref in references:
        cut, cut_rate = sf.read(local_path(ref["audio_path"]), dtype="int16", always_2d=False)
        if getattr(cut, "ndim", 1) > 1:
            cut = cut[:, 0]
        if cut_rate != sample_rate:
            raise ValueError(f"Sample-rate mismatch for {ref['review_id']}: {cut_rate} vs {sample_rate}.")
        frame = exact_start_frame(source, cut)
        start_s = frame / sample_rate
        end_s = (frame + len(cut)) / sample_rate
        meta_start = ref.get("start_s")
        location = {
            "review_id": ref["review_id"], "candidate_id": ref.get("candidate_id"),
            "audio_path": str(Path(ref["audio_path"])), "audio_sha256": ref.get("sha256"),
            "source_path": str(source_path.relative_to(ROOT)),
            "sample_rate": sample_rate, "start_frame": frame, "n_samples": int(len(cut)),
            "located_start_s": start_s, "located_end_s": end_s,
            "meta_start_s": meta_start, "meta_end_s": ref.get("end_s"),
            "meta_delta_s": None if meta_start is None else start_s - float(meta_start),
            "speaker_id": speaker_id_from_candidate(ref.get("candidate_id")),
        }
        locations.append(location)
        turn = SpeakerTurn(speaker_id=location["speaker_id"], start_s=start_s, end_s=end_s)
        turn._original_start_s = start_s
        turn._original_end_s = end_s
        turn._raw_start_s = start_s
        turn._raw_end_s = end_s
        turns.append(turn)
    write_jsonl(output / "locations.jsonl", locations)

    audio = Audio.from_file(source_path, source_id="youtube:H0VpjeULCck",
                            title=source_path.stem)
    locked, lock_audits = align_and_lock_syllable_boundaries(
        audio, turns, aligner_engine=args.aligner_engine, aligner_model=args.aligner_model,
        aligner_language="vi", aligner_device=args.device,
        competitor_intervals_by_speaker=competitor_intervals(turns),
    )
    write_jsonl(output / "lock_audits.jsonl", lock_audits)
    if len(lock_audits) != len(references):
        raise ValueError("Word-lock audits must preserve one record per incoming cut.")
    segmented, segment_audits = smart_segment_speaker_turns(
        audio, locked, min_duration_s=args.min_duration_s, max_duration_s=args.max_duration_s,
    )
    write_jsonl(output / "segment_audits.jsonl", segment_audits)

    cutter = AudioCutter(output_dir=output / "locked_cuts")
    children_dir = output / "locked_cuts"
    children_dir.mkdir(exist_ok=True)
    children_by_parent: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for child in segmented:
        parent_key = (
            round(float(getattr(child, "_original_start_s", child.start_s)), 6),
            round(float(getattr(child, "_original_end_s", child.end_s)), 6),
        )
        cut_audio = cutter.cut(audio, child.start_s, child.end_s, unit="seconds",
                               output_path=children_dir / f"{child.speaker_id}_{child.start_s:.4f}-{child.end_s:.4f}.wav")
        digest = inspect_audio(Path(cut_audio.path))["sha256"]
        children_by_parent[parent_key].append({
            "start_s": child.start_s, "end_s": child.end_s, "duration_s": child.duration_s,
            "speaker_id": child.speaker_id, "audio_path": str(Path(cut_audio.path).relative_to(ROOT)),
            "audio_sha256": digest, "transcript": getattr(child, "_transcript", None),
        })

    review_packet = output / "review_packet"
    review_audio = review_packet / "review_audio"
    review_audio.mkdir(parents=True)
    blind, comparison = [], []
    for ref, location, audit in zip(references, locations, lock_audits):
        parent_key = (round(location["located_start_s"], 6), round(location["located_end_s"], 6))
        children = children_by_parent.get(parent_key, [])
        old = labels.get(ref["review_id"], {})
        lock_rejected = audit.get("action") == "reject"
        row = {
            "review_id": ref["review_id"], "location": location,
            "old_decision": old.get("decision"), "old_defects": old.get("defects", []),
            "lock": {k: audit.get(k) for k in (
                "action", "error", "policy", "start_s", "end_s",
                "delta_start_ms", "delta_end_ms", "tail_rescued", "transcript")},
            "children": children,
            "pipeline_decision": "reject" if lock_rejected or not children else "emit",
        }
        comparison.append(row)
        for index, child in enumerate(children, 1):
            child_start = int(round(child["start_s"] * sample_rate))
            child_end = int(round(child["end_s"] * sample_rate))
            child["bounds_unchanged"] = (
                child_start == location["start_frame"]
                and child_end == location["start_frame"] + location["n_samples"]
            )
            if child["bounds_unchanged"]:
                continue
            review_id = f"{ref['review_id']}_c{index:02d}"
            rel = f"review_audio/{review_id}.wav"
            shutil.copyfile(ROOT / child["audio_path"], review_packet / rel)
            child["teacher_review_id"] = review_id
            blind.append({
                "review_id": review_id, "audio_path": rel, "audio_sha256": child["audio_sha256"],
                "duration_s": child["duration_s"], "parent_review_id": ref["review_id"],
                "rubric_version": "tts-v1", "reviewer_id": None, "decision": None,
                "dimensions": dict.fromkeys(DIMENSIONS), "defects": [],
                "context_reviewed": False, "notes": "",
            })
    write_jsonl(output / "comparison.jsonl", comparison)
    write_jsonl(review_packet / "blind_review.jsonl", blind)
    write_jsonl(review_packet / "private_reference.jsonl", comparison)
    clipping_codes = {"clipped_word_start", "clipped_word_end"}
    summary = {
        "source": str(source_path.relative_to(ROOT)), "clips": len(references),
        "aligner_engine": args.aligner_engine, "aligner_model": args.aligner_model,
        "aligner_device": args.device,
        "duration_policy_s": [args.min_duration_s, args.max_duration_s],
        "located_exact": len(locations),
        "meta_delta_s": {
            "min": min((row["meta_delta_s"] for row in locations if row["meta_delta_s"] is not None), default=None),
            "max": max((row["meta_delta_s"] for row in locations if row["meta_delta_s"] is not None), default=None),
        },
        "lock_rejected": sum(row["pipeline_decision"] == "reject" and row["lock"].get("action") == "reject"
                             for row in comparison),
        "emitted": sum(row["pipeline_decision"] == "emit" for row in comparison),
        "emitted_children": sum(len(row["children"]) for row in comparison),
        "bounds_unchanged": sum(c.get("bounds_unchanged") for row in comparison for c in row["children"]),
        "changed_children_for_teacher": len(blind),
        "old_clipped_still_present": sum(
            row["old_decision"] == "reject"
            and any(d.get("code") in clipping_codes for d in row["old_defects"])
            for row in comparison),
        "old_clipped_pipeline_reject": sum(
            row["pipeline_decision"] == "reject"
            and any(d.get("code") in clipping_codes for d in row["old_defects"])
            for row in comparison),
        "old_pass_pipeline_reject": sum(
            row["old_decision"] == "pass" and row["pipeline_decision"] == "reject" for row in comparison),
        "caveat": "Sample-accurate location is established. Quality of emitted repairs requires a fresh MEDIUM audit of changed children.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


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
    prepare_parser.add_argument("--candidates", help="Instead prepare an unlabeled prediction packet from candidate JSONL")
    prepare_parser.add_argument("--output", required=True, help="New directory under .data/; must not exist")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--packet", required=True)
    report_parser.add_argument("--labels", required=True)
    report_parser.add_argument("--eligible-only", action="store_true")
    teacher_parser = commands.add_parser("teacher", help="User-authorized Gemini MEDIUM ground-truth audit")
    teacher_parser.add_argument("--packet", required=True)
    teacher_parser.add_argument("--limit", type=int, default=31)
    lineage_parser = commands.add_parser("lineage", help="Recover legacy source IDs with explicit evidence")
    lineage_parser.add_argument("--manifests", nargs="+", default=[".data/distillation/train_v3.jsonl", ".data/distillation/val_v3.jsonl"])
    lineage_parser.add_argument("--output", required=True)
    export_parser = commands.add_parser("export", help="Join fresh labels with verified source identities")
    export_parser.add_argument("--packet", required=True)
    export_parser.add_argument("--output", required=True)
    bounds_parser = commands.add_parser(
        "boundaries", help="Locate cuts in source audio and relock/split with public APIs")
    bounds_parser.add_argument("--packet", required=True)
    bounds_parser.add_argument("--source", required=True)
    bounds_parser.add_argument("--output", required=True)
    bounds_parser.add_argument("--aligner-engine", default="whisper_timestamped")
    bounds_parser.add_argument("--aligner-model", default="vinai/PhoWhisper-small")
    bounds_parser.add_argument("--device", default="cuda:0")
    bounds_parser.add_argument("--min-duration-s", type=float, default=2.0)
    bounds_parser.add_argument("--max-duration-s", type=float, default=15.0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "report":
        label_report(args)
    elif args.command == "teacher":
        teacher_audit(args)
    elif args.command == "lineage":
        restore_lineage(args)
    elif args.command == "export":
        export_labels(args)
    else:
        evaluate_boundaries(args)


if __name__ == "__main__":
    main()
