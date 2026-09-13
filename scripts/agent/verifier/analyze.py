"""Analyze a saved verifier run; write plots, statistics and error cases into INPUT/plot/."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

from _common.files import (  # noqa: E402
    digest,
    infer_audio_family,
    positive_int,
    progress,
    read_json,
    safe_name,
    write_json,
)
from agent.verifier._verdicts import (
    ACOUSTIC_FAILURE_CODES,
    ELIGIBILITY_FAILURE_CODES,
    FAILURE_CODES,
    _known_prompts,
    _validate_verdict,
)


CSV_FIELDS = (
    "audio_path",
    "final_verdict",
    "assistant_raw_response",
    "transcript",
    "transcript_chars",
    "transcript_words",
    "verifier_status",
    "failure_stage",
    "failure_code",
    "accepted",
    "is_valid",
    "is_pass",
    "is_reject",
    "is_missing",
    "raw_response_available",
    "response_kind",
    "sample_scope",
    "family",
    "sample_id",
    "turn_index",
    "speaker_id",
    "start_s",
    "end_s",
    "duration_s",
    "start_sample",
    "end_sample",
    "overlap",
    "overlap_with_json",
    "diarization_confidence",
    "clip_sha256",
    "source_recording_path",
    "source_recording_sha256",
    "manifest_path",
    "manifest_sha256",
    "diarizer_model",
    "diarizer_parameters_json",
    "verdict_file",
    "verdict_file_sha256",
    "raw_response_file",
    "raw_response_sha256",
    "verifier_backend",
    "verifier_model",
    "verifier_parameters_json",
    "prompt",
    "prompt_sha256",
    "schema_profile",
    "parsed_decision",
    "speaker_purity",
    "word_completeness",
    "audio_quality",
    "boundary_start",
    "boundary_end",
    "num_speakers",
    "secondary_speech_s",
    "dominant_speaker_id",
    "failure_codes_json",
    "reason",
    "parsed_response_json",
    "latency_s",
    "confidence",
    "usage_json",
    "cost_json",
    *(f"failure_{code}" for code in FAILURE_CODES),
)


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _prompt_hash(prompt: Any) -> str:
    if not isinstance(prompt, str):
        return ""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _parallel_load(
    paths: list[Path], concurrency: int, batch_size: int
) -> list[tuple[Path, dict[str, Any] | None, str | None]]:
    def load_one(path: Path) -> tuple[Path, dict[str, Any] | None, str | None]:
        try:
            return path, read_json(path), None
        except (OSError, ValueError, json.JSONDecodeError):
            return path, None, "invalid_json"

    def load_batch(batch: list[Path]) -> list[tuple[Path, dict[str, Any] | None, str | None]]:
        return [load_one(path) for path in batch]

    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
    if concurrency > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(batches))) as pool:
            return [item for group in pool.map(load_batch, batches) for item in group]
    return [item for group in map(load_batch, batches) for item in group]


def _empty_row() -> dict[str, Any]:
    return {field: "" for field in CSV_FIELDS}


def _manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    manifest_sha = digest(manifest_path)
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    family = infer_audio_family(manifest_path)
    result = []
    for index, turn in enumerate(manifest.get("turns", [])):
        if not isinstance(turn, dict):
            continue
        clip_name = turn.get("clip")
        clip_path = (
            (manifest_path.parent / clip_name).resolve()
            if isinstance(clip_name, str) and clip_name
            else None
        )
        start = turn.get("start_s")
        end = turn.get("end_s")
        try:
            duration = max(0.0, float(end) - float(start))
        except (TypeError, ValueError):
            duration = ""
        row = _empty_row()
        row.update(
            {
                "audio_path": str(clip_path) if clip_path else "",
                "verifier_status": "missing",
                "accepted": False,
                "is_valid": False,
                "is_pass": False,
                "is_reject": False,
                "is_missing": True,
                "raw_response_available": False,
                "sample_scope": "expected",
                "family": family,
                "sample_id": f"{manifest_sha[:12]}:{index:06d}",
                "turn_index": index,
                "speaker_id": turn.get("speaker_id", ""),
                "start_s": start if start is not None else "",
                "end_s": end if end is not None else "",
                "duration_s": duration,
                "start_sample": turn.get("start_sample", ""),
                "end_sample": turn.get("end_sample", ""),
                "overlap": turn.get("overlap", ""),
                "overlap_with_json": _json_cell(turn.get("overlap_with")),
                "diarization_confidence": turn.get("confidence", ""),
                "clip_sha256": turn.get("clip_sha256", ""),
                "source_recording_path": source.get("path", ""),
                "source_recording_sha256": source.get("sha256", ""),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest_sha,
                "diarizer_model": manifest.get("model", ""),
                "diarizer_parameters_json": _json_cell(manifest.get("parameters")),
            }
        )
        for code in FAILURE_CODES:
            row[f"failure_{code}"] = ""
        result.append(row)
    return result


def _read_raw_response(artifact_path: Path, data: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    response = data.get("response") if isinstance(data.get("response"), dict) else {}
    response_path_value = response.get("path")
    if not isinstance(response_path_value, str) or not response_path_value:
        return "", response, None
    response_path = Path(response_path_value)
    if not response_path.is_absolute():
        response_path = (artifact_path.parent / response_path).resolve()
    if not response_path.is_file():
        response_path = artifact_path.with_suffix(".txt")
    if not response_path.is_file():
        return "", response, "missing_raw_response"
    expected_sha = response.get("sha256")
    actual_sha = digest(response_path)
    if isinstance(expected_sha, str) and expected_sha and actual_sha != expected_sha:
        return "", response, "raw_response_hash_mismatch"
    with response_path.open("r", encoding="utf-8", newline="") as stream:
        raw = stream.read()
    response = {**response, "resolved_path": str(response_path), "actual_sha256": actual_sha}
    return raw, response, None


def _artifact_values(
    artifact_path: Path,
    data: dict[str, Any],
    known_prompts: dict[str, str],
) -> dict[str, Any]:
    values = _empty_row()
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    parameters = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
    prompt = parameters.get("prompt")
    backend = str(data.get("model") or "")
    raw, response, raw_error = _read_raw_response(artifact_path, data)
    verdict = data.get("verdict", data)
    response_kind = response.get("kind", "")
    schema_profile = (
        known_prompts.get(prompt.strip(), "custom")
        if isinstance(prompt, str)
        else "vibevoice_v1"
        if backend == "vibevoice"
        else "custom"
    )

    values.update(
        {
            "audio_path": source.get("path", ""),
            "assistant_raw_response": raw,
            "raw_response_available": bool(response.get("path")) and raw_error is None,
            "response_kind": response_kind,
            "clip_sha256": source.get("sha256", ""),
            "verdict_file": str(artifact_path.resolve()),
            "verdict_file_sha256": digest(artifact_path),
            "raw_response_file": response.get("resolved_path", response.get("path", "")),
            "raw_response_sha256": response.get("actual_sha256", response.get("sha256", "")),
            "verifier_backend": backend,
            "verifier_model": (
                parameters.get("model")
                or parameters.get("model_id")
                or (verdict.get("_model") if isinstance(verdict, dict) else "")
                or ""
            ),
            "verifier_parameters_json": _json_cell(parameters),
            "prompt": prompt if isinstance(prompt, str) else "",
            "prompt_sha256": _prompt_hash(prompt),
            "schema_profile": schema_profile,
        }
    )

    if data.get("status") == "fail":
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        values.update(
            {
                "verifier_status": "fail",
                "failure_stage": error.get("stage", "verifier"),
                "failure_code": error.get("code", "verifier_failed"),
                "accepted": False,
                "is_valid": False,
                "is_pass": False,
                "is_reject": False,
                "is_missing": False,
            }
        )
        return values

    if isinstance(verdict, dict):
        codes = verdict.get("failure_codes")
        code_list = codes if isinstance(codes, list) else []
        transcript = verdict.get("transcript")
        transcript = transcript if isinstance(transcript, str) else ""
        values.update(
            {
                "parsed_decision": verdict.get("decision", ""),
                "speaker_purity": verdict.get("speaker_purity", ""),
                "word_completeness": verdict.get("word_completeness", ""),
                "audio_quality": verdict.get("audio_quality", ""),
                "boundary_start": verdict.get("boundary_start", ""),
                "boundary_end": verdict.get("boundary_end", ""),
                "num_speakers": verdict.get("num_speakers", ""),
                "secondary_speech_s": verdict.get("secondary_speech_s", ""),
                "dominant_speaker_id": verdict.get("dominant_speaker_id", ""),
                "failure_codes_json": _json_cell(codes),
                "reason": verdict.get("reason", ""),
                "transcript": transcript,
                "transcript_chars": len(transcript),
                "transcript_words": len(transcript.split()),
                "parsed_response_json": _json_cell(verdict),
                "latency_s": verdict.get("_latency_s", ""),
                "confidence": verdict.get("confidence", ""),
                "usage_json": _json_cell(verdict.get("_usage")),
                "cost_json": _json_cell(verdict.get("_cost")),
            }
        )
        for field in ("speaker_purity", "word_completeness", "audio_quality"):
            if verdict.get(field) in FAILURE_CODES:
                code_list = [*code_list, verdict[field]]
        for boundary, code in (("boundary_start", "clipped_word_start"), ("boundary_end", "clipped_word_end")):
            if verdict.get(boundary) == "clipped":
                code_list = [*code_list, code]
        for code in FAILURE_CODES:
            values[f"failure_{code}"] = code in code_list

    profile, schema_error = _validate_verdict(verdict, prompt, backend, known_prompts)
    values["schema_profile"] = profile
    if raw_error is not None and data.get("status") == "success":
        schema_error = raw_error
    if schema_error is not None:
        values.update(
            {
                "verifier_status": "fail",
                "failure_stage": "response" if raw_error else "schema",
                "failure_code": schema_error,
                "accepted": False,
                "is_valid": False,
                "is_pass": False,
                "is_reject": False,
                "is_missing": False,
            }
        )
        return values

    assert isinstance(verdict, dict)
    decision = verdict["decision"]
    values.update(
        {
            "final_verdict": decision,
            "verifier_status": "success",
            "accepted": decision == "pass",
            "is_valid": True,
            "is_pass": decision == "pass",
            "is_reject": decision == "reject",
            "is_missing": False,
        }
    )
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _save_figure(plt: Any, fig: Any, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def _model_groups(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    groups = {}
    for row in rows:
        settings = json.loads(row["verifier_parameters_json"] or "{}")
        identity = [row["verifier_backend"], row["verifier_model"], settings]
        key = hashlib.sha256(_json_cell(identity).encode()).hexdigest()[:10]
        if key not in groups:
            model = row["verifier_model"] or row["verifier_backend"] or "unidentified"
            effort = settings.get("reasoning_effort") or settings.get("thinking_level")
            variant = ", ".join(str(value) for value in (effort, settings.get("inference_mode")) if value)
            label = f"{model} ({variant})" if variant else model
            groups[key] = {"label": label, "backend": row["verifier_backend"], "parameters": settings, "rows": []}
        groups[key]["rows"].append(row)
    return groups


def _case_categories(row: dict[str, str]) -> list[tuple[str, str]]:
    if row["verifier_status"] != "success":
        return [("processing", row["failure_code"] or row["verifier_status"] or "unknown")]
    codes = [
        *(('acoustic', code) for code in ACOUSTIC_FAILURE_CODES if row[f"failure_{code}"].lower() == "true"),
        *(('eligibility', code) for code in ELIGIBILITY_FAILURE_CODES if row[f"failure_{code}"].lower() == "true"),
    ]
    if row["final_verdict"] == "reject" and not codes:
        return [("acoustic", "reject_without_defect_code")]
    return codes


def _markdown_relpath(path_str: str, base_dir: Path) -> str:
    if not path_str:
        return ""
    try:
        p = Path(path_str)
        base = base_dir.resolve()
        if p.is_absolute():
            target_p = p.resolve() if p.exists() else None
            if target_p is None:
                for anchor in (".data", base.name):
                    if anchor in p.parts:
                        idx = p.parts.index(anchor)
                        subpath = Path(*p.parts[idx:])
                        curr = base
                        while curr != curr.parent:
                            if (curr / subpath).exists() or (curr / anchor).exists():
                                target_p = (curr / subpath).resolve()
                                break
                            curr = curr.parent
                        if target_p:
                            break
            if target_p is None:
                target_p = p
            try:
                rel = os.path.relpath(target_p, base)
            except ValueError:
                rel = str(target_p)
        else:
            rel = str(p)
        return Path(rel).as_posix()
    except Exception:
        return path_str


def _write_error_reports(all_csv: Path, output_dir: Path, verdict_dir: Path) -> dict[str, Any]:
    """Read the canonical CSV to produce model-specific counts and reviewable cases."""
    rows = _read_csv(all_csv)
    groups = _model_groups(rows)
    cases, stats, model_stats = [], [], []
    report = ["# Verifier analysis", "", f"Input: `{verdict_dir}`", "",
              "Acoustic defects and eligibility failures are labels reported by the model; they are not verified model mistakes.",
              "Processing failures (generation, parsing, schema, missing responses) are counted separately.",
              "Without a diarization manifest, coverage describes discovered artifacts only; absent inputs cannot be counted.",
              "A clip may have several defect types and appear in several case rows.", "",
              "## Model runs", "", "| Model / configuration ID | Samples | Pass | Reject | Transcripts | Invalid | Missing |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]

    def cell(value: Any) -> str:
        return str(value or "").replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def link(value: str, label: str) -> str:
        if not value:
            return ""
        target = _markdown_relpath(value, output_dir)
        if not target:
            return cell(label)
        if any(c in target for c in (" ", "(", ")")):
            return f"[{cell(label)}](<{target}>)"
        return f"[{cell(label)}]({target})"

    for key, group in groups.items():
        group_rows = group["rows"]
        total = len(group_rows)
        counts = Counter(row["verifier_status"] for row in group_rows)
        decisions = Counter(row["final_verdict"] for row in group_rows if row["verifier_status"] == "success")
        transcripts = sum(
            row["verifier_status"] == "success"
            and row["final_verdict"] == "pass"
            and bool(row["transcript"].strip())
            for row in group_rows
        )
        model_stats.append({"id": key, "label": group["label"], "backend": group["backend"],
                            "samples": total, "statuses": dict(counts), "decisions": dict(decisions),
                            "transcripts": transcripts})
        report.append(f"| {cell(group['label'])} / `{key}` | {total} | {decisions['pass']} | {decisions['reject']} | {transcripts} | {counts['fail']} | {counts['missing']} |")
        category_counts = Counter()
        for row in group_rows:
            for kind, category in _case_categories(row):
                category_counts[(kind, category)] += 1
                cases.append({"model_id": key, "model": group["label"], "kind": kind, "category": category,
                              "family": row["family"], "audio_path": row["audio_path"],
                              "verdict_file": row["verdict_file"], "decision": row["final_verdict"],
                              "reason": row["reason"], "failure_stage": row["failure_stage"],
                              "transcript": row["transcript"],
                              "assistant_raw_response": row["assistant_raw_response"]})
        known_categories = {
            *(("acoustic", code) for code in ACOUSTIC_FAILURE_CODES),
            *(("eligibility", code) for code in ELIGIBILITY_FAILURE_CODES),
        }
        for kind, category in sorted(set(category_counts) | known_categories):
            count = category_counts[(kind, category)]
            denominator = total if kind == "processing" else counts["success"]
            stats.append({"model_id": key, "model": group["label"], "kind": kind, "category": category,
                          "count": count, "denominator": denominator,
                          "rate": round(count / denominator, 6) if denominator else None})
    report.extend(["", "## Run parameters", ""])
    parameter_fields = ("model", "model_id", "reasoning_effort", "thinking_level", "inference_mode", "max_tokens",
                        "max_new_tokens", "temperature", "top_p", "device", "dtype", "min_secondary_speech_s")
    for key, group in groups.items():
        settings = {name: group["parameters"][name] for name in parameter_fields if name in group["parameters"]}
        report.extend([f"### {cell(group['label'])} / {key}", "", "```json", json.dumps(settings, ensure_ascii=False, indent=2), "```", ""])
    report.extend(["## Error statistics", "",
                   "Acoustic rates use valid artifacts as denominator; processing rates use all artifacts in that model group.",
                   "A zero count means no such label was observed, not proof that the model can assess that criterion.", "",
                   "| Model / ID | Kind | Category | Count | Denominator | Rate |",
                   "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in stats:
        rate = f"{row['rate'] * 100:.1f}%" if row["rate"] is not None else "N/A"
        report.append(f"| {cell(row['model'])} / {row['model_id']} | {row['kind']} | {row['category']} | {row['count']} | {row['denominator']} | {rate} |")
    report.extend(["", "## Error cases", ""])
    for kind, category in sorted({(row["kind"], row["category"]) for row in cases}):
        selected = [row for row in cases if row["kind"] == kind and row["category"] == category]
        report.extend([f"### {kind}: {cell(category)} ({len(selected)})", "",
                       "| Model / ID | Family | Audio | Verdict JSON | Reason / stage |",
                       "| --- | --- | --- | --- | --- |"])
        for row in selected:
            report.append(f"| {cell(row['model'])} / {row['model_id']} | {cell(row['family'])} | {link(row['audio_path'], 'audio')} | "
                          f"{link(row['verdict_file'], Path(row['verdict_file']).name)} | {cell(row['reason'] or row['failure_stage'])} |")
        report.append("")
    if not cases:
        report.append("No reported defects or processing failures in the discovered artifacts.")
    case_fields = ("model_id", "model", "kind", "category", "family", "audio_path", "verdict_file", "decision", "reason", "transcript", "failure_stage", "assistant_raw_response")
    stat_fields = ("model_id", "model", "kind", "category", "count", "denominator", "rate")
    for name, fields, records in (("error_cases.csv", case_fields, cases), ("error_stats.csv", stat_fields, stats)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"model_stats": model_stats, "error_stats": stats,
            "error_cases_csv": str(output_dir / "error_cases.csv"), "error_stats_csv": str(output_dir / "error_stats.csv"),
            "report_md": str(output_dir / "report.md")}


def _make_plots(
    all_csv: Path, successful_csv: Path, output_dir: Path
) -> list[str]:
    all_rows = _read_csv(all_csv)
    successful_rows = _read_csv(successful_csv)
    expected_rows = [row for row in all_rows if row["sample_scope"] == "expected"]
    coverage_rows = expected_rows or all_rows
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    plots = []
    status_order = ("success", "fail", "missing")
    status_colors = {"success": "#22c55e", "fail": "#f97316", "missing": "#9ca3af"}
    status_counts = Counter(row["verifier_status"] for row in coverage_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(status_order, [status_counts[s] for s in status_order], color=[status_colors[s] for s in status_order])
    ax.bar_label(bars)
    ax.set_ylabel("Samples")
    ax.set_title("Verifier coverage")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plots.append(_save_figure(plt, fig, output_dir / "coverage.png"))

    decisions = Counter(row["final_verdict"] for row in successful_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    decision_order = ("pass", "reject")
    decision_colors = ("#22c55e", "#ef4444")
    bars = axes[0].bar(decision_order, [decisions[d] for d in decision_order], color=decision_colors)
    axes[0].bar_label(bars)
    axes[0].set_title("Valid decisions by sample")
    axes[0].set_ylabel("Samples")
    durations = {
        decision: sum(_number(row["duration_s"]) for row in successful_rows if row["final_verdict"] == decision)
        for decision in decision_order
    }
    bars = axes[1].bar(decision_order, [durations[d] for d in decision_order], color=decision_colors)
    axes[1].bar_label(bars, fmt="%.1f")
    axes[1].set_title("Valid decisions by duration")
    axes[1].set_ylabel("Seconds")
    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plots.append(_save_figure(plt, fig, output_dir / "decisions.png"))

    transcript_rows = [row for row in all_rows if row["schema_profile"] == "acoustic_defect_v3"]
    if transcript_rows:
        transcript_schema_errors = {"missing_transcript", "unexpected_transcript", "transcript_not_last"}
        transcript_counts = Counter(
            "pass_with_transcript"
            if row["verifier_status"] == "success" and row["final_verdict"] == "pass"
            else "reject_without_transcript"
            if row["verifier_status"] == "success" and row["final_verdict"] == "reject"
            else "transcript_schema_failure"
            if row["failure_code"] in transcript_schema_errors
            else "other_failure"
            for row in transcript_rows
        )
        transcript_order = (
            "pass_with_transcript",
            "reject_without_transcript",
            "transcript_schema_failure",
            "other_failure",
        )
        lengths = [
            int(row["transcript_chars"])
            for row in transcript_rows
            if row["verifier_status"] == "success"
            and row["final_verdict"] == "pass"
            and row["transcript_chars"]
        ]
        fig, axes = plt.subplots(1, 2 if lengths else 1, figsize=(12 if lengths else 8, 5))
        if not lengths:
            axes = [axes]
        bars = axes[0].bar(
            transcript_order,
            [transcript_counts[label] for label in transcript_order],
            color=("#22c55e", "#64748b", "#ef4444", "#f97316"),
        )
        axes[0].bar_label(bars)
        axes[0].set_title("Acoustic v3 transcript contract")
        axes[0].set_ylabel("Samples")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
        if lengths:
            axes[1].hist(
                lengths,
                bins=min(30, max(1, len(set(lengths)))),
                color="#3b82f6",
                edgecolor="white",
            )
            axes[1].set_title("Pass transcript lengths")
            axes[1].set_xlabel("Characters")
            axes[1].set_ylabel("Transcripts")
        plots.append(_save_figure(plt, fig, output_dir / "transcripts.png"))

    defect_counts = {
        code: sum(row[f"failure_{code}"].lower() == "true" for row in successful_rows)
        for code in FAILURE_CODES
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(list(reversed(FAILURE_CODES)), [defect_counts[c] for c in reversed(FAILURE_CODES)], color="#ef4444")
    ax.bar_label(bars)
    ax.set_xlabel("Valid samples")
    ax.set_title("Detected failure codes")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plots.append(_save_figure(plt, fig, output_dir / "defects.png"))

    dimension_columns = [
        ("speaker_purity", "Speaker purity"),
        ("word_completeness", "Word completeness"),
        ("audio_quality", "Audio quality"),
        ("boundary_start", "Start boundary"),
        ("boundary_end", "End boundary"),
    ]
    available = [(column, title) for column, title in dimension_columns if any(row[column] for row in successful_rows)]
    if available:
        fig, axes = plt.subplots(len(available), 1, figsize=(11, max(4, 3.2 * len(available))))
        if len(available) == 1:
            axes = [axes]
        non_defect_tags = {"pure", "complete", "studio_clean", "clean", "none"}
        dim_canonical_order = {
            "speaker_purity": ["pure", "secondary_speaker", "overlapping_speech", "tail_speaker_intrusion"],
            "word_completeness": ["complete", "clipped_word_start", "clipped_word_end"],
            "audio_quality": ["studio_clean", "music_bleed", "noisy_reverberant", "distorted"],
            "boundary_start": ["clean", "clipped"],
            "boundary_end": ["clean", "clipped"],
        }
        for i, (ax, (column, title)) in enumerate(zip(axes, available)):
            counts = Counter(row[column] for row in successful_rows if row[column])
            canonical = dim_canonical_order.get(column, [])

            def _sort_key(label: str) -> tuple[int, int, str]:
                norm = label.lower().strip().replace(" ", "_")
                is_non_defect = 0 if norm in non_defect_tags else 1
                order_idx = canonical.index(norm) if norm in canonical else 999
                return (is_non_defect, order_idx, label)

            labels = sorted(counts, key=_sort_key)
            colors = [
                "#22c55e" if lbl.lower().strip().replace(" ", "_") in non_defect_tags else "#ef4444"
                for lbl in labels
            ]
            vals = [counts[label] for label in labels]
            total_dim = sum(vals)
            bars = ax.bar(labels, vals, color=colors)
            bar_labels = [
                f"{v} ({v / total_dim * 100:.1f}%)" if total_dim > 0 else str(v)
                for v in vals
            ]
            ax.bar_label(bars, labels=bar_labels, padding=3)
            ax.set_title(title)
            ax.set_ylabel("Samples")
            ax.margins(y=0.22)
            ax.tick_params(axis="x", rotation=15)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            if i == 0:
                legend_elements = [
                    Patch(facecolor="#22c55e", label="Pass / Quality target"),
                    Patch(facecolor="#ef4444", label="Defect / Failure tag"),
                ]
                ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
        plots.append(_save_figure(plt, fig, output_dir / "dimensions.png"))

    speaker_rows = [row for row in all_rows if row["speaker_id"]]
    if speaker_rows:
        speakers = sorted({row["speaker_id"] for row in speaker_rows})
        fig, ax = plt.subplots(figsize=(max(9, len(speakers) * 1.1), 5))
        bottoms = [0.0] * len(speakers)
        for status in status_order:
            values = [
                sum(_number(row["duration_s"]) for row in speaker_rows if row["speaker_id"] == speaker and row["verifier_status"] == status)
                for speaker in speakers
            ]
            ax.bar(speakers, values, bottom=bottoms, label=status, color=status_colors[status])
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax.set_ylabel("Duration (seconds)")
        ax.set_title("Verifier coverage by diarized speaker")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plots.append(_save_figure(plt, fig, output_dir / "by_speaker.png"))

    # Measurement distributions reflect only fields actually present in valid verdicts.
    measurements = []
    for column, label in (("latency_s", "Latency (s)"), ("confidence", "Confidence"),
                          ("num_speakers", "Speaker count"), ("secondary_speech_s", "Secondary speech (s)")):
        values = []
        for row in successful_rows:
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            measurements.append((label, values))
    if measurements:
        fig, axes = plt.subplots(len(measurements), 1, squeeze=False, figsize=(10, 3 * len(measurements)))
        for ax, (label, values) in zip(axes[:, 0], measurements):
            ax.hist(values, bins=min(30, max(1, len(set(values)))), color="#3b82f6", edgecolor="white")
            ax.set_xlabel(label)
            ax.set_ylabel("Valid samples")
        plots.append(_save_figure(plt, fig, output_dir / "measurements.png"))

    error_counts = Counter(row["failure_code"] or row["verifier_status"] for row in all_rows if row["verifier_status"] != "success")
    if error_counts:
        fig, ax = plt.subplots(figsize=(11, max(4, len(error_counts) * .4)))
        bars = ax.barh(list(error_counts), list(error_counts.values()), color="#f97316")
        ax.bar_label(bars)
        ax.set_title("Processing failures and missing artifacts")
        ax.set_xlabel("Samples")
        plots.append(_save_figure(plt, fig, output_dir / "processing_errors.png"))

    groups = _model_groups(all_rows)
    if len(groups) > 1:
        labels = [f"{group['label']} / {key}" for key, group in groups.items()]
        fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * .6)))
        left = [0] * len(labels)
        for status, decision, color in (("success", "pass", "#22c55e"), ("success", "reject", "#ef4444"),
                                         ("fail", "", "#f97316"), ("missing", "", "#9ca3af")):
            values = [sum(row["verifier_status"] == status and (not decision or row["final_verdict"] == decision)
                          for row in group["rows"]) for group in groups.values()]
            ax.barh(labels, values, left=left, label=decision or status, color=color)
            left = [a + b for a, b in zip(left, values)]
        ax.set_xlabel("Samples")
        ax.set_title("Decisions and processing status by model configuration")
        ax.legend()
        plots.append(_save_figure(plt, fig, output_dir / "by_model.png"))

    timeline_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        if row["manifest_path"] and row["speaker_id"] and row["start_s"] and row["end_s"]:
            timeline_groups[row["manifest_path"]].append(row)
    for manifest_path, rows in timeline_groups.items():
        speakers = sorted({row["speaker_id"] for row in rows})
        speaker_y = {speaker: index for index, speaker in enumerate(reversed(speakers))}
        fig, ax = plt.subplots(figsize=(14, max(3.0, 1.2 + 0.5 * len(speakers))))
        decision_colors_by_status = {
            ("success", "pass"): "#22c55e",
            ("success", "reject"): "#ef4444",
            ("fail", ""): "#f97316",
            ("missing", ""): "#9ca3af",
        }
        for row in rows:
            start = _number(row["start_s"])
            duration = max(0.0, _number(row["end_s"]) - start)
            key = (row["verifier_status"], row["final_verdict"])
            color = decision_colors_by_status.get(key, "#9ca3af")
            overlap = row["overlap"].lower() == "true"
            ax.barh(
                speaker_y[row["speaker_id"]],
                duration,
                left=start,
                height=0.62,
                color=color,
                edgecolor="#111827" if overlap else "none",
                linewidth=1.4 if overlap else 0,
            )
        ax.set_yticks([speaker_y[speaker] for speaker in speakers])
        ax.set_yticklabels(speakers)
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Verifier timeline: {Path(manifest_path).parent.name}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        ax.legend(
            handles=[
                Patch(color="#22c55e", label="pass"),
                Patch(color="#ef4444", label="reject"),
                Patch(color="#f97316", label="invalid"),
                Patch(color="#9ca3af", label="missing"),
                Patch(facecolor="white", edgecolor="#111827", label="diarizer overlap"),
            ],
            loc="upper right",
        )
        suffix = "" if len(timeline_groups) == 1 else f"_{safe_name(Path(manifest_path).parent.name)}"
        plots.append(_save_figure(plt, fig, output_dir / f"timeline{suffix}.png"))
    return plots


def _summary_from_csv(
    all_csv: Path, successful_csv: Path, plots: list[str], verdict_dir: Path
) -> dict[str, Any]:
    all_rows = _read_csv(all_csv)
    successful_rows = _read_csv(successful_csv)
    expected_rows = [row for row in all_rows if row["sample_scope"] == "expected"]
    coverage_rows = expected_rows or all_rows
    successful_coverage_rows = [row for row in coverage_rows if row["verifier_status"] == "success"]
    statuses = Counter(row["verifier_status"] for row in coverage_rows)
    decisions = Counter(row["final_verdict"] for row in successful_coverage_rows)
    expected = len(coverage_rows)
    valid = statuses.get("success", 0)
    duration_by_status = {
        status: round(sum(_number(row["duration_s"]) for row in coverage_rows if row["verifier_status"] == status), 6)
        for status in ("success", "fail", "missing")
    }
    duration_by_decision = {
        decision: round(sum(_number(row["duration_s"]) for row in successful_coverage_rows if row["final_verdict"] == decision), 6)
        for decision in ("pass", "reject")
    }
    transcript_profile_rows = [
        row for row in coverage_rows if row["schema_profile"] == "acoustic_defect_v3"
    ]
    valid_transcript_rows = [
        row for row in transcript_profile_rows if row["verifier_status"] == "success"
    ]
    transcript_pass_rows = [
        row for row in valid_transcript_rows if row["final_verdict"] == "pass"
    ]
    prompt_groups = sorted(
        {
            (row["verifier_backend"], row["verifier_model"], row["prompt_sha256"], row["schema_profile"])
            for row in successful_rows
        }
    )
    return {
        "schema_version": 2,
        "operation": "analyze_verifier",
        "status": "partial" if statuses.get("fail", 0) or statuses.get("missing", 0) or any(row["sample_scope"] == "orphan" for row in all_rows) else "complete",
        "source": {"verdict_dir": str(verdict_dir)},
        "coverage": {
            "expected_rows": expected,
            "valid": valid,
            "invalid": statuses.get("fail", 0),
            "missing": statuses.get("missing", 0),
            "unmatched_artifacts": sum(row["sample_scope"] == "orphan" for row in all_rows),
            "coverage_rate": round(valid / expected, 6) if expected else 0.0,
        },
        "decisions": {
            "pass": decisions.get("pass", 0),
            "reject": decisions.get("reject", 0),
            "pass_rate_among_valid": round(decisions.get("pass", 0) / valid, 6) if valid else 0.0,
            "conservative_pass_rate": round(decisions.get("pass", 0) / expected, 6) if expected else 0.0,
        },
        "duration_s": {
            "by_status": duration_by_status,
            "by_decision": duration_by_decision,
        },
        "transcripts": {
            "profile_rows": len(transcript_profile_rows),
            "valid_pass_with_transcript": sum(bool(row["transcript"].strip()) for row in transcript_pass_rows),
            "valid_reject_without_transcript": sum(
                row["final_verdict"] == "reject" and not row["transcript"].strip()
                for row in valid_transcript_rows
            ),
            "characters": sum(int(row["transcript_chars"] or 0) for row in transcript_pass_rows),
            "words": sum(int(row["transcript_words"] or 0) for row in transcript_pass_rows),
            "schema_failures": sum(
                row["failure_code"] in {"missing_transcript", "unexpected_transcript", "transcript_not_last"}
                for row in transcript_profile_rows
            ),
        },
        "prompt_groups": [
            {"backend": backend, "model": model, "prompt_sha256": prompt_sha, "schema_profile": profile}
            for backend, model, prompt_sha, profile in prompt_groups
        ],
        "failures": [
            {
                "audio_path": row["audio_path"],
                "verdict_file": row["verdict_file"],
                "stage": row["failure_stage"],
                "code": row["failure_code"],
            }
            for row in all_rows
            if row["verifier_status"] == "fail"
        ],
        "outputs": {
            "all_samples_csv": {"path": str(all_csv), "sha256": digest(all_csv), "rows": len(all_rows)},
            "successful_samples_csv": {"path": str(successful_csv), "sha256": digest(successful_csv), "rows": len(successful_rows)},
            "plots": plots,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--verdict-dir", dest="verdict_dir", type=Path, required=True, help="Saved model/effort run or one family; recursively read verifier JSON artifacts")
    parser.add_argument("--input-manifest", type=Path, action="append", default=[], help="Diarization segments.json to join (repeatable)")
    parser.add_argument("--manifest-dir", type=Path, help="Directory searched recursively for diarization segments.json files")
    parser.add_argument("--output-dir", type=Path, help="Analysis directory (default: <input-dir>/plot; regenerated on each run)")
    parser.add_argument("--overwrite", action="store_true", help="Allow refreshing a nonempty custom --output-dir (default plot/ refreshes automatically)")
    parser.add_argument("--concurrency", type=positive_int, default=1, help="Number of worker threads for JSON loading")
    parser.add_argument("--batch-size", type=positive_int, default=1, help="Number of JSON files loaded per worker task")
    args = parser.parse_args()

    verdict_dir = args.verdict_dir.expanduser().resolve()
    if not verdict_dir.is_dir():
        parser.error(f"Verdict directory not found: {verdict_dir}")
    output_dir = (args.output_dir or verdict_dir / "plot").expanduser().resolve()
    if output_dir == verdict_dir or output_dir in verdict_dir.parents:
        parser.error("Output directory must not be the input directory or its ancestor")
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"Output path is not a directory: {output_dir}")
    if args.output_dir and output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Analysis directory is not empty: {output_dir}; use --overwrite")
    previous_plots = []
    previous_summary = output_dir / "analysis.json"
    if previous_summary.is_file():
        try:
            previous = read_json(previous_summary)
            if previous.get("operation") == "analyze_verifier":
                previous_plots = previous.get("outputs", {}).get("plots", [])
        except (OSError, ValueError):
            pass
    progress("ANALYZE_START", f"{verdict_dir} -> {output_dir}")
    json_files = []

    def walk_error(error: OSError) -> None:
        parser.error(str(error))

    for root, dirs, files in os.walk(verdict_dir, onerror=walk_error):
        dirs[:] = sorted(
            name for name in dirs
            if name not in {"plot", "plots", "work", "comparisons", "experiments", "__pycache__"}
            and not name.startswith(".") and (Path(root) / name).resolve() != output_dir
        )
        json_files.extend(Path(root) / name for name in sorted(files) if name.endswith(".json") and not name.startswith("."))
    progress("ANALYZE_LOAD", f"Loading {len(json_files)} verifier JSON candidate(s)")
    loaded = _parallel_load(json_files, args.concurrency, args.batch_size)
    verifier_artifacts = [
        item
        for item in loaded
        if item[1] is None or item[1].get("operation") == "verify"
        or "verdict" in item[1] or "decision" in item[1]
    ]

    manifest_paths = [path.resolve() for path in args.input_manifest]
    if args.manifest_dir is not None:
        manifest_root = args.manifest_dir.resolve()
        if not manifest_root.is_dir():
            parser.error(f"Manifest directory not found: {manifest_root}")
        manifest_paths.extend(path.resolve() for path in manifest_root.rglob("segments.json"))
    if not manifest_paths:
        for _, data, _ in verifier_artifacts:
            if not isinstance(data, dict):
                continue
            source = data.get("source")
            source_path = source.get("path") if isinstance(source, dict) else None
            if isinstance(source_path, str):
                candidate = Path(source_path).resolve().parent / "segments.json"
                if candidate.is_file():
                    manifest_paths.append(candidate)
    manifest_paths = sorted(set(manifest_paths))
    for path in manifest_paths:
        if not path.is_file():
            parser.error(f"Input manifest not found: {path}")

    rows = [row for path in manifest_paths for row in _manifest_rows(path)]
    expected_by_path: dict[str, list[int]] = defaultdict(list)
    expected_by_hash: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["audio_path"]:
            expected_by_path[str(Path(row["audio_path"]).resolve())].append(index)
        if row["clip_sha256"]:
            expected_by_hash[str(row["clip_sha256"])].append(index)

    known_prompts = _known_prompts()
    matched: set[int] = set()
    orphan_rows = []
    for artifact_path, data, load_error in verifier_artifacts:
        if data is None:
            row = _empty_row()
            row.update(
                {
                    "verifier_status": "fail",
                    "failure_stage": "parse",
                    "failure_code": load_error or "invalid_json",
                    "accepted": False,
                    "is_valid": False,
                    "is_pass": False,
                    "is_reject": False,
                    "is_missing": False,
                    "raw_response_available": False,
                    "sample_scope": "orphan",
                    "verdict_file": str(artifact_path.resolve()),
                }
            )
            orphan_rows.append(row)
            continue

        try:
            values = _artifact_values(artifact_path, data, known_prompts)
        except Exception:
            source = data.get("source") if isinstance(data.get("source"), dict) else {}
            values = _empty_row()
            values.update(
                {
                    "audio_path": source.get("path", ""),
                    "clip_sha256": source.get("sha256", ""),
                    "verifier_status": "fail",
                    "failure_stage": "artifact",
                    "failure_code": "artifact_read_failed",
                    "accepted": False,
                    "is_valid": False,
                    "is_pass": False,
                    "is_reject": False,
                    "is_missing": False,
                    "raw_response_available": False,
                    "verdict_file": str(artifact_path.resolve()),
                }
            )
        source_path = values["audio_path"]
        candidates = expected_by_path.get(str(Path(source_path).resolve()), []) if source_path else []
        if not candidates and values["clip_sha256"]:
            candidates = expected_by_hash.get(str(values["clip_sha256"]), [])
        available = [index for index in candidates if index not in matched]
        if rows and len(available) != 1:
            values["verifier_status"] = "fail"
            values["failure_stage"] = "manifest_match"
            values["failure_code"] = "ambiguous_source" if len(available) > 1 else "unmatched_source"
            values["final_verdict"] = ""
            values["accepted"] = False
            values["is_valid"] = False
            values["is_pass"] = False
            values["is_reject"] = False
            values["sample_scope"] = "orphan"
            orphan_rows.append(values)
            continue
        if rows:
            target_index = available[0]
            matched.add(target_index)
            manifest_values = rows[target_index]
            preserved = {
                key: manifest_values[key]
                for key in (
                    "sample_scope", "family", "sample_id", "turn_index", "speaker_id", "start_s", "end_s",
                    "duration_s", "start_sample", "end_sample", "overlap", "overlap_with_json",
                    "diarization_confidence", "source_recording_path", "source_recording_sha256",
                    "manifest_path", "manifest_sha256", "diarizer_model", "diarizer_parameters_json",
                )
            }
            manifest_values.update(values)
            manifest_values.update(preserved)
        else:
            values["sample_scope"] = "discovered"
            if source_path:
                values["family"] = infer_audio_family(Path(source_path))
                values["sample_id"] = values["clip_sha256"] or hashlib.sha256(source_path.encode("utf-8")).hexdigest()
            orphan_rows.append(values)

    rows.extend(orphan_rows)
    if not rows:
        parser.error("No diarization turns or verifier artifacts were found")
    rows.sort(
        key=lambda row: (
            str(row.get("family", "")),
            str(row.get("manifest_path", "")),
            _number(row.get("start_s", 0)),
            str(row.get("audio_path", "")),
            str(row.get("verdict_file", "")),
        )
    )
    successful_rows = [row for row in rows if row.get("verifier_status") == "success"]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / "all_samples.csv"
    successful_csv = output_dir / "successful_samples.csv"
    _write_csv(all_csv, rows)
    _write_csv(successful_csv, successful_rows)
    progress("ANALYZE_CSV", f"Wrote {len(rows)} total and {len(successful_rows)} successful row(s)")

    progress("ANALYZE_PLOTS", "Rendering coverage, decisions, defects and available measurements")
    plots = _make_plots(all_csv, successful_csv, output_dir)
    summary = _summary_from_csv(all_csv, successful_csv, plots, verdict_dir)
    details = _write_error_reports(all_csv, output_dir, verdict_dir)
    summary.update(details)
    for previous_plot in previous_plots:
        path = Path(previous_plot)
        if path.parent == output_dir and path.suffix == ".png" and str(path) not in plots:
            path.unlink(missing_ok=True)
    summary_path = output_dir / "analysis.json"
    write_json(summary_path, summary)
    progress(
        "ANALYZE_DONE",
        f"{summary['coverage']['valid']} valid, {summary['coverage']['invalid']} invalid, "
        f"{summary['coverage']['missing']} missing -> {output_dir}",
    )
    progress("ANALYZE_REPORT", f"{summary['decisions']['pass']} pass, {summary['decisions']['reject']} reject; {output_dir / 'report.md'}")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
