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
    persist_path,
    positive_int,
    progress,
    read_json,
    resolve_stored_path,
    safe_name,
    write_json,
)
from _verdicts import (
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
    "emotion",
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
    "cost_usd",
    "input_cost_usd",
    "output_cost_usd",
    "tokens_prompt",
    "tokens_output",
    "tokens_thinking",
    "tokens_total",
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
    paths: list[Path], concurrency: int = 1, batch_size: int = 1
) -> list[tuple[Path, dict[str, Any] | None, str | None]]:
    if not paths:
        return []
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    def load_batch(chunk: list[Path]) -> list[tuple[Path, dict[str, Any] | None, str | None]]:
        results = []
        for path in chunk:
            try:
                results.append((path, read_json(path), None))
            except Exception as exc:
                results.append((path, None, type(exc).__name__))
        return results

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
                "audio_path": persist_path(clip_path) if clip_path else "",
                "verifier_status": "missing",
                "accepted": False,
                "is_valid": False,
                "is_pass": False,
                "is_reject": False,
                "is_missing": True,
                "sample_scope": "expected",
                "family": family,
                "sample_id": turn.get("id", ""),
                "turn_index": index,
                "speaker_id": turn.get("speaker_id", ""),
                "start_s": start if start is not None else "",
                "end_s": end if end is not None else "",
                "duration_s": duration,
                "start_sample": turn.get("start_sample", ""),
                "end_sample": turn.get("end_sample", ""),
                "overlap": bool(turn.get("overlap")),
                "overlap_with_json": _json_cell(turn.get("overlap_with")),
                "diarization_confidence": turn.get("confidence", ""),
                "clip_sha256": turn.get("sha256", ""),
                "source_recording_path": persist_path(source["path"]) if source.get("path") else "",
                "source_recording_sha256": source.get("sha256", ""),
                "manifest_path": persist_path(manifest_path),
                "manifest_sha256": manifest_sha,
                "diarizer_model": manifest.get("model", ""),
                "diarizer_parameters_json": _json_cell(manifest.get("parameters")),
            }
        )
        result.append(row)
    return result


def _read_raw_response(artifact_path: Path, data: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    response = data.get("response") if isinstance(data.get("response"), dict) else {}
    response_path_value = response.get("path")
    if not isinstance(response_path_value, str) or not response_path_value:
        return "", response, None
    response_path = resolve_stored_path(response_path_value, base=artifact_path.parent)
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
    response = {**response, "resolved_path": persist_path(response_path), "actual_sha256": actual_sha}
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

    cost_info = verdict.get("_cost") if isinstance(verdict, dict) else None
    usage_info = verdict.get("_usage") if isinstance(verdict, dict) else None
    if not cost_info and isinstance(data, dict):
        cost_info = data.get("_cost") or data.get("cost")
    if not usage_info and isinstance(data, dict):
        usage_info = data.get("_usage") or data.get("usage")

    cost_usd = ""
    input_cost_usd = ""
    output_cost_usd = ""
    if isinstance(cost_info, dict) and "total_usd" in cost_info:
        try:
            cost_usd = f"{float(cost_info.get('total_usd') or 0.0):.6f}"
            input_cost_usd = f"{float(cost_info.get('input_usd') or 0.0):.6f}"
            output_cost_usd = f"{float(cost_info.get('output_usd') or 0.0):.6f}"
        except (ValueError, TypeError):
            pass

    tokens_prompt = ""
    tokens_output = ""
    tokens_thinking = ""
    tokens_total = ""
    if isinstance(usage_info, dict):
        try:
            tokens_prompt = str(usage_info.get("prompt_tokens") or "")
            tokens_output = str(usage_info.get("output_tokens") or "")
            tokens_thinking = str(usage_info.get("thinking_tokens") or "")
            tot = usage_info.get("total_tokens")
            if tot is None:
                p_int = int(tokens_prompt) if tokens_prompt else 0
                o_int = int(tokens_output) if tokens_output else 0
                t_int = int(tokens_thinking) if tokens_thinking else 0
                if p_int or o_int or t_int:
                    tot = p_int + o_int + t_int
            tokens_total = str(tot or "")
        except (ValueError, TypeError):
            pass

    values.update(
        {
            "audio_path": persist_path(source["path"]) if source.get("path") else "",
            "assistant_raw_response": raw,
            "raw_response_available": bool(response.get("path")) and raw_error is None,
            "response_kind": response_kind,
            "clip_sha256": source.get("sha256", ""),
            "verdict_file": persist_path(artifact_path),
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
            "usage_json": _json_cell(usage_info),
            "cost_json": _json_cell(cost_info),
            "cost_usd": cost_usd,
            "input_cost_usd": input_cost_usd,
            "output_cost_usd": output_cost_usd,
            "tokens_prompt": tokens_prompt,
            "tokens_output": tokens_output,
            "tokens_thinking": tokens_thinking,
            "tokens_total": tokens_total,
        }
    )

    if data.get("status") == "fail":
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        invalid_verdict = data.get("invalid_verdict") if isinstance(data.get("invalid_verdict"), dict) else {}
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
                "emotion": str(invalid_verdict.get("emotion") or "").strip() if invalid_verdict else "",
                "reason": str(invalid_verdict.get("reason") or "").strip() if invalid_verdict else "",
                "transcript": str(invalid_verdict.get("transcript") or "").strip() if invalid_verdict else "",
            }
        )
        return values

    if isinstance(verdict, dict):
        codes = verdict.get("failure_codes")
        code_list = codes if isinstance(codes, list) else []
        transcript = verdict.get("transcript")
        transcript = transcript if isinstance(transcript, str) else ""
        emotion = verdict.get("emotion")
        emotion = emotion.strip() if isinstance(emotion, str) else ""
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
                "emotion": emotion,
                "transcript": transcript,
                "transcript_chars": len(transcript),
                "transcript_words": len(transcript.split()),
                "parsed_response_json": _json_cell(verdict),
                "latency_s": verdict.get("_latency_s", ""),
                "confidence": verdict.get("confidence", ""),
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
            target_p = resolve_stored_path(p)
            try:
                rel = os.path.relpath(target_p, base)
            except ValueError:
                rel = persist_path(target_p)
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
                              "reason": row["reason"], "emotion": row.get("emotion", ""),
                              "failure_stage": row["failure_stage"],
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

    successful_rows = [row for row in rows if row.get("verifier_status") == "success"]
    emotion_rows = [row for row in successful_rows if row.get("emotion")]
    if emotion_rows:
        report.extend([
            "",
            "## Emotion distribution",
            "",
            "| Emotion | Pass samples | Reject samples | Total samples | Pass duration (s) | Total duration (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        all_emotions = Counter(row["emotion"].strip().lower() for row in emotion_rows if row["emotion"].strip())
        for e, total_count in all_emotions.most_common():
            p_cnt = sum(row["final_verdict"] == "pass" and row["emotion"].strip().lower() == e for row in emotion_rows)
            r_cnt = sum(row["final_verdict"] == "reject" and row["emotion"].strip().lower() == e for row in emotion_rows)
            p_dur = sum(_number(row["duration_s"]) for row in emotion_rows if row["final_verdict"] == "pass" and row["emotion"].strip().lower() == e)
            tot_dur = sum(_number(row["duration_s"]) for row in emotion_rows if row["emotion"].strip().lower() == e)
            report.append(f"| {cell(e)} | {p_cnt} | {r_cnt} | {total_count} | {p_dur:.1f} | {tot_dur:.1f} |")

    cost_rows = [row for row in rows if _number(row.get("cost_usd", 0.0)) > 0]
    if cost_rows:
        costs = [_number(r["cost_usd"]) for r in cost_rows]
        total_usd = sum(costs)
        in_usd = sum(_number(r.get("input_cost_usd", 0.0)) for r in cost_rows)
        out_usd = sum(_number(r.get("output_cost_usd", 0.0)) for r in cost_rows)
        mean_usd = total_usd / len(cost_rows)
        sorted_costs = sorted(costs)
        median_usd = sorted_costs[len(sorted_costs) // 2]
        tot_dur = sum(_number(r.get("duration_s", 0.0)) for r in cost_rows)
        rate_min = f"${(total_usd / tot_dur * 60.0):.4f} USD" if tot_dur > 0 else "N/A"
        tot_tokens = sum(int(r.get("tokens_total") or 0) for r in cost_rows)
        tot_prompt = sum(int(r.get("tokens_prompt") or 0) for r in cost_rows)
        tot_output = sum(int(r.get("tokens_output") or 0) for r in cost_rows)
        tot_thinking = sum(int(r.get("tokens_thinking") or 0) for r in cost_rows)

        report.extend([
            "",
            "## Cost analysis",
            "",
            f"- **Total cost:** ${total_usd:.4f} USD across {len(cost_rows)} priced sample(s)",
            f"- **Cost per sample:** mean ${mean_usd:.4f} | median ${median_usd:.4f} | min ${sorted_costs[0]:.4f} | max ${sorted_costs[-1]:.4f}",
            f"- **Cost breakdown:** input ${in_usd:.4f} ({in_usd / total_usd * 100:.1f}%) | output ${out_usd:.4f} ({out_usd / total_usd * 100:.1f}%)" if total_usd > 0 else "",
            f"- **Rate per audio minute:** {rate_min}",
        ])
        if tot_tokens > 0:
            report.append(f"- **Token usage:** {tot_tokens:,} total ({tot_prompt:,} prompt, {tot_output:,} output, {tot_thinking:,} thinking)")

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
    case_fields = ("model_id", "model", "kind", "category", "family", "audio_path", "verdict_file", "decision", "reason", "emotion", "transcript", "failure_stage", "assistant_raw_response")
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
        transcript_colors = {
            "pass_with_transcript": "#22c55e",
            "reject_without_transcript": "#ef4444",
            "transcript_schema_failure": "#a855f7",
            "other_failure": "#9ca3af",
        }
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        bars = axes[0].bar(
            transcript_order,
            [transcript_counts[k] for k in transcript_order],
            color=[transcript_colors[k] for k in transcript_order],
        )
        axes[0].bar_label(bars)
        axes[0].set_title("Transcript compliance by sample")
        axes[0].set_ylabel("Samples")
        axes[0].tick_params(axis="x", rotation=18)
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

        char_counts = [_number(row["transcript_chars"]) for row in successful_rows if row["final_verdict"] == "pass" and row["schema_profile"] == "acoustic_defect_v3"]
        if char_counts:
            axes[1].hist(char_counts, bins=min(20, max(5, len(char_counts) // 2)), color="#3b82f6", edgecolor="black")
            axes[1].set_title("Transcript length distribution (pass samples)")
            axes[1].set_xlabel("Characters")
            axes[1].set_ylabel("Samples")
            axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
        else:
            axes[1].axis("off")
        plots.append(_save_figure(plt, fig, output_dir / "transcripts.png"))

    emotion_rows = [row for row in successful_rows if row.get("emotion")]
    if emotion_rows:
        emotion_counts = Counter(row["emotion"].strip().lower() for row in emotion_rows if row["emotion"].strip())
        top_emotions = [e for e, _ in emotion_counts.most_common(12)]
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(top_emotions) * 0.45)))
        p_counts = [sum(row["final_verdict"] == "pass" and row["emotion"].strip().lower() == e for row in emotion_rows) for e in top_emotions]
        r_counts = [sum(row["final_verdict"] == "reject" and row["emotion"].strip().lower() == e for row in emotion_rows) for e in top_emotions]
        axes[0].barh(top_emotions, p_counts, label="pass", color="#22c55e")
        axes[0].barh(top_emotions, r_counts, left=p_counts, label="reject", color="#ef4444")
        axes[0].set_title("Emotion counts by decision")
        axes[0].set_xlabel("Samples")
        axes[0].legend()
        axes[0].grid(True, axis="x", linestyle="--", alpha=0.4)

        p_durs = [sum(_number(row["duration_s"]) for row in emotion_rows if row["final_verdict"] == "pass" and row["emotion"].strip().lower() == e) for e in top_emotions]
        r_durs = [sum(_number(row["duration_s"]) for row in emotion_rows if row["final_verdict"] == "reject" and row["emotion"].strip().lower() == e) for e in top_emotions]
        axes[1].barh(top_emotions, p_durs, label="pass", color="#22c55e")
        axes[1].barh(top_emotions, r_durs, left=p_durs, label="reject", color="#ef4444")
        axes[1].set_title("Emotion duration by decision")
        axes[1].set_xlabel("Seconds")
        axes[1].legend()
        axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)
        plots.append(_save_figure(plt, fig, output_dir / "emotions.png"))

    speakers = sorted({row["speaker_id"] for row in successful_rows if row["speaker_id"]})
    if len(speakers) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(speakers) * 0.4)))
        p_spk = [sum(row["final_verdict"] == "pass" and row["speaker_id"] == s for row in successful_rows) for s in speakers]
        r_spk = [sum(row["final_verdict"] == "reject" and row["speaker_id"] == s for row in successful_rows) for s in speakers]
        axes[0].barh(speakers, p_spk, label="pass", color="#22c55e")
        axes[0].barh(speakers, r_spk, left=p_spk, label="reject", color="#ef4444")
        axes[0].set_title("Decisions by speaker")
        axes[0].set_xlabel("Samples")
        axes[0].legend()
        axes[0].grid(True, axis="x", linestyle="--", alpha=0.4)

        pd_spk = [sum(_number(row["duration_s"]) for row in successful_rows if row["final_verdict"] == "pass" and row["speaker_id"] == s) for s in speakers]
        rd_spk = [sum(_number(row["duration_s"]) for row in successful_rows if row["final_verdict"] == "reject" and row["speaker_id"] == s) for s in speakers]
        axes[1].barh(speakers, pd_spk, label="pass", color="#22c55e")
        axes[1].barh(speakers, rd_spk, left=pd_spk, label="reject", color="#ef4444")
        axes[1].set_title("Duration by speaker")
        axes[1].set_xlabel("Seconds")
        axes[1].legend()
        axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)
        plots.append(_save_figure(plt, fig, output_dir / "by_speaker.png"))

    defect_counts = {code: sum(row[f"failure_{code}"].lower() == "true" for row in successful_rows) for code in FAILURE_CODES}
    observed_defects = {k: v for k, v in defect_counts.items() if v > 0}
    if observed_defects:
        fig, ax = plt.subplots(figsize=(10, max(4, len(observed_defects) * 0.4)))
        sorted_defects = sorted(observed_defects.items(), key=lambda item: item[1])
        bars = ax.barh([k for k, _ in sorted_defects], [v for _, v in sorted_defects], color="#ef4444")
        ax.bar_label(bars)
        ax.set_title("Reported acoustic and eligibility defect counts")
        ax.set_xlabel("Samples")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        plots.append(_save_figure(plt, fig, output_dir / "defects.png"))

    dimensions = [
        ("speaker_purity", "speaker purity", ("pure", "overlap", "multi_speaker")),
        ("word_completeness", "word completeness", ("complete", "clipped_start", "clipped_end", "clipped_both")),
        ("audio_quality", "audio quality", ("clean", "reverb", "noisy", "distorted", "compressed")),
    ]
    present_dims = [
        (col, title, classes) for col, title, classes in dimensions
        if any(row[col] for row in successful_rows)
    ]
    if present_dims:
        fig, axes = plt.subplots(1, len(present_dims), figsize=(5 * len(present_dims), 4.5))
        axes_list = [axes] if len(present_dims) == 1 else list(axes)
        for ax, (col, title, classes) in zip(axes_list, present_dims):
            counts = Counter(row[col] for row in successful_rows if row[col])
            bars = ax.bar(classes, [counts[c] for c in classes], color="#6366f1")
            ax.bar_label(bars)
            ax.set_title(f"Reported {title}")
            ax.set_ylabel("Samples")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plots.append(_save_figure(plt, fig, output_dir / "dimensions.png"))

    durations_all = [_number(row["duration_s"]) for row in successful_rows if _number(row["duration_s"]) > 0]
    confidences = [_number(row["confidence"]) for row in successful_rows if row["confidence"]]
    latencies = [_number(row["latency_s"]) for row in successful_rows if row["latency_s"]]
    panels = [p for p in (
        ("Duration (seconds)", durations_all, "#3b82f6"),
        ("Model confidence", confidences, "#10b981"),
        ("Inference latency (s)", latencies, "#f59e0b"),
    ) if p[1]]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4))
        axes_list = [axes] if len(panels) == 1 else list(axes)
        for ax, (title, values, color) in zip(axes_list, panels):
            ax.hist(values, bins=min(20, max(5, len(values) // 2)), color=color, edgecolor="black")
            ax.set_title(title)
            ax.set_ylabel("Samples")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
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

    cost_rows = [r for r in all_rows if _number(r.get("cost_usd", 0.0)) > 0]
    if cost_rows:
        costs = [_number(r["cost_usd"]) for r in cost_rows]
        input_costs = [_number(r.get("input_cost_usd", 0.0)) for r in cost_rows]
        output_costs = [_number(r.get("output_cost_usd", 0.0)) for r in cost_rows]
        durations = [_number(r.get("duration_s", 0.0)) for r in cost_rows]
        total_cost = sum(costs)
        mean_cost = total_cost / len(costs)
        sorted_costs = sorted(costs)
        n_c = len(sorted_costs)
        median_cost = (
            sorted_costs[n_c // 2]
            if n_c % 2 != 0
            else (sorted_costs[n_c // 2 - 1] + sorted_costs[n_c // 2]) / 2.0
        )
        total_dur = sum(durations)
        rate_per_min = (total_cost / total_dur * 60.0) if total_dur > 0 else 0.0

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # Panel 1: Cost distribution histogram
        ax_dist = axes[0, 0]
        bins = min(25, max(8, len(costs) // 2)) if len(set(costs)) > 1 else 10
        ax_dist.hist(costs, bins=bins, color="#0284c7", edgecolor="white", alpha=0.85)
        ax_dist.axvline(mean_cost, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean: ${mean_cost:.4f}")
        ax_dist.axvline(median_cost, color="#16a34a", linestyle="-.", linewidth=1.8, label=f"Median: ${median_cost:.4f}")
        ax_dist.set_title("Cost Distribution per Sample", fontsize=12, fontweight="bold")
        ax_dist.set_xlabel("Cost per sample ($ USD)")
        ax_dist.set_ylabel("Number of samples")
        ax_dist.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax_dist.legend(loc="upper right")

        # Panel 2: Cumulative expenditure progression
        ax_cum = axes[0, 1]
        cum_costs = []
        c_acc = 0.0
        for c in costs:
            c_acc += c
            cum_costs.append(c_acc)
        x_indices = list(range(1, len(cum_costs) + 1))
        ax_cum.plot(x_indices, cum_costs, color="#0284c7", linewidth=2.2)
        ax_cum.fill_between(x_indices, cum_costs, color="#38bdf8", alpha=0.25)
        ax_cum.scatter([x_indices[-1]], [cum_costs[-1]], color="#dc2626", s=50, zorder=5)
        ax_cum.annotate(
            f"Total: ${total_cost:.4f}",
            xy=(x_indices[-1], cum_costs[-1]),
            xytext=(-70, 10),
            textcoords="offset points",
            fontweight="bold",
            color="#dc2626",
            arrowprops=dict(arrowstyle="->", color="#dc2626"),
        )
        ax_cum.set_title("Cumulative Cost Progression", fontsize=12, fontweight="bold")
        ax_cum.set_xlabel("Sample index")
        ax_cum.set_ylabel("Cumulative cost ($ USD)")
        ax_cum.grid(True, linestyle="--", alpha=0.4)

        # Panel 3: Cost components breakdown (Input vs Output)
        ax_comp = axes[1, 0]
        tot_in = sum(input_costs)
        tot_out = sum(output_costs)
        comp_labels = ["Input Cost", "Output Cost"]
        comp_vals = [tot_in, tot_out]
        comp_colors = ["#3b82f6", "#8b5cf6"]
        if tot_in + tot_out > 0:
            wedges, texts, autotexts = ax_comp.pie(
                comp_vals,
                labels=comp_labels,
                colors=comp_colors,
                autopct=lambda pct: f"{pct:.1f}%\n(${pct * (tot_in + tot_out) / 100:.4f})",
                startangle=140,
                wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
            )
            for at in autotexts:
                at.set_fontsize(10)
                at.set_fontweight("bold")
            ax_comp.set_title("Cost Breakdown by Component", fontsize=12, fontweight="bold")
        else:
            ax_comp.text(0.5, 0.5, "No component cost breakdown", ha="center", va="center")

        # Panel 4: Cost vs Audio Duration & Summary Metrics
        ax_stat = axes[1, 1]
        has_dur = any(d > 0 for d in durations)
        if has_dur:
            scatter_colors = [
                "#22c55e" if r.get("final_verdict") == "pass" else "#ef4444"
                for r in cost_rows
            ]
            ax_stat.scatter(durations, costs, c=scatter_colors, alpha=0.7, edgecolors="none", s=35)
            ax_stat.set_xlabel("Audio duration (seconds)")
            ax_stat.set_ylabel("Cost per sample ($ USD)")
            ax_stat.set_title("Cost vs Duration & Summary", fontsize=12, fontweight="bold")
            ax_stat.grid(True, linestyle="--", alpha=0.4)
            pass_patch = Patch(color="#22c55e", label="pass")
            reject_patch = Patch(color="#ef4444", label="reject")
            ax_stat.legend(handles=[pass_patch, reject_patch], loc="lower right")
        else:
            ax_stat.axis("off")

        # Add text stats summary box
        tot_prompt = sum(int(r.get("tokens_prompt") or 0) for r in cost_rows)
        tot_output = sum(int(r.get("tokens_output") or 0) for r in cost_rows)
        tot_thinking = sum(int(r.get("tokens_thinking") or 0) for r in cost_rows)
        tot_tokens = sum(int(r.get("tokens_total") or 0) for r in cost_rows)

        summary_lines = [
            f"Total Cost: ${total_cost:.4f} USD",
            f"Priced Samples: {len(cost_rows):,}",
            f"Mean: ${mean_cost:.4f} / sample",
            f"Median: ${median_cost:.4f} / sample",
        ]
        if total_dur > 0:
            summary_lines.append(f"Rate: ${rate_per_min:.4f} / audio min")
        if tot_tokens > 0:
            summary_lines.append(f"Tokens: {tot_tokens:,} total")
            if tot_prompt > 0:
                summary_lines.append(f"  Prompt: {tot_prompt:,}")
            if tot_output > 0:
                summary_lines.append(f"  Output: {tot_output:,}")
            if tot_thinking > 0:
                summary_lines.append(f"  Thinking: {tot_thinking:,}")

        ax_stat.text(
            0.05,
            0.95,
            "\n".join(summary_lines),
            transform=ax_stat.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
        )

        plots.append(_save_figure(plt, fig, output_dir / "costs.png"))

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
            (
                row["verifier_backend"],
                row["verifier_model"],
                row["prompt_sha256"],
                row["schema_profile"],
            )
            for row in coverage_rows
        }
    )
    emotion_rows = [row for row in successful_coverage_rows if row.get("emotion")]
    emotions_summary = {
        "total_labeled": len(emotion_rows),
        "counts": dict(Counter(row["emotion"].strip().lower() for row in emotion_rows if row["emotion"].strip())),
        "pass_counts": dict(Counter(row["emotion"].strip().lower() for row in emotion_rows if row["final_verdict"] == "pass" and row["emotion"].strip())),
        "reject_counts": dict(Counter(row["emotion"].strip().lower() for row in emotion_rows if row["final_verdict"] == "reject" and row["emotion"].strip())),
        "duration_s": {
            e: round(sum(_number(row["duration_s"]) for row in emotion_rows if row["emotion"].strip().lower() == e), 6)
            for e in sorted({row["emotion"].strip().lower() for row in emotion_rows if row["emotion"].strip()})
        },
    }

    cost_rows = [r for r in coverage_rows if _number(r.get("cost_usd", 0.0)) > 0]
    cost_summary = None
    if cost_rows:
        costs = [_number(r["cost_usd"]) for r in cost_rows]
        input_costs = [_number(r.get("input_cost_usd", 0.0)) for r in cost_rows]
        output_costs = [_number(r.get("output_cost_usd", 0.0)) for r in cost_rows]
        durations = [_number(r.get("duration_s", 0.0)) for r in cost_rows]
        tot_dur = sum(durations)
        total_usd = sum(costs)
        sorted_costs = sorted(costs)
        n_c = len(sorted_costs)
        median_usd = (
            sorted_costs[n_c // 2]
            if n_c % 2 != 0
            else (sorted_costs[n_c // 2 - 1] + sorted_costs[n_c // 2]) / 2.0
        )
        p25_usd = sorted_costs[int(n_c * 0.25)]
        p75_usd = sorted_costs[int(n_c * 0.75)]
        p95_usd = sorted_costs[int(n_c * 0.95)] if n_c >= 20 else sorted_costs[-1]

        cost_summary = {
            "currency": "USD",
            "samples_priced": len(cost_rows),
            "total_usd": round(total_usd, 6),
            "input_usd": round(sum(input_costs), 6),
            "output_usd": round(sum(output_costs), 6),
            "mean_usd": round(total_usd / len(cost_rows), 6),
            "median_usd": round(median_usd, 6),
            "min_usd": round(sorted_costs[0], 6),
            "max_usd": round(sorted_costs[-1], 6),
            "p25_usd": round(p25_usd, 6),
            "p75_usd": round(p75_usd, 6),
            "p95_usd": round(p95_usd, 6),
            "cost_per_minute_usd": round(total_usd / tot_dur * 60.0, 6) if tot_dur > 0 else None,
            "tokens": {
                "prompt_tokens": sum(int(r.get("tokens_prompt") or 0) for r in cost_rows),
                "output_tokens": sum(int(r.get("tokens_output") or 0) for r in cost_rows),
                "thinking_tokens": sum(int(r.get("tokens_thinking") or 0) for r in cost_rows),
                "total_tokens": sum(int(r.get("tokens_total") or 0) for r in cost_rows),
            },
        }

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
        "emotions": emotions_summary,
        "costs": cost_summary,
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
            "all_samples_csv": {"path": persist_path(all_csv), "sha256": digest(all_csv), "rows": len(all_rows)},
            "successful_samples_csv": {"path": persist_path(successful_csv), "sha256": digest(successful_csv), "rows": len(successful_rows)},
            "plots": plots,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-id", "--input-dir", "--verdict-dir", dest="verdict_dir", type=Path, required=True, help="Saved model/effort run or one family; recursively read verifier JSON artifacts")
    parser.add_argument("-im", "--input-manifest", type=Path, action="append", default=[], help="Diarization segments.json to join (repeatable)")
    parser.add_argument("-md", "--manifest-dir", type=Path, help="Directory searched recursively for diarization segments.json files")
    parser.add_argument("-od", "--output-dir", type=Path, help="Analysis directory (default: <input-dir>/plot; regenerated on each run)")
    parser.add_argument("-w", "-ow", "--overwrite", action="store_true", help="Allow refreshing a nonempty custom --output-dir (default plot/ refreshes automatically)")
    parser.add_argument("-c", "--concurrency", type=positive_int, default=1, help="Number of worker threads for JSON loading")
    parser.add_argument("-b", "-bs", "--batch-size", type=positive_int, default=1, help="Number of JSON files loaded per worker task")
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
        json_files.extend(
            Path(root) / name
            for name in sorted(files)
            if name.endswith(".json") and not name.startswith(".") and name not in {"analysis.json", "report.json"}
        )
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
                candidate = resolve_stored_path(source_path).parent / "segments.json"
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
            expected_by_path[str(resolve_stored_path(row["audio_path"]))].append(index)
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
                    "verdict_file": persist_path(artifact_path),
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
                    "audio_path": persist_path(source["path"]) if source.get("path") else "",
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
                    "verdict_file": persist_path(artifact_path),
                }
            )
        source_path = values["audio_path"]
        candidates = expected_by_path.get(str(resolve_stored_path(source_path)), []) if source_path else []
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
                values["family"] = infer_audio_family(resolve_stored_path(source_path))
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
