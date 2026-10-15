"""Collect subagent labelling responses into verifier artifacts and a control report.

Reads the raw `<id>.txt` responses written by labelling subagents, parses them with
the production verifier parser and schema validator, writes standard verifier
artifacts for target clips (`<family>/<clip>_subagent.json` plus the raw response),
and scores the hidden controls against the pre-registered acceptance rules. Labels
are model labels: dev/teacher data only, never ground truth. No model is called.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "s4-agent"))
from _audio import VerifierResponseError, parse_verifier_response  # noqa: E402
from _common.files import ROOT, digest, persist_path, progress, resolve_stored_path, write_json  # noqa: E402
from _verdicts import _known_prompts, _validate_verdict  # noqa: E402

# Pre-registered acceptance rules (WORKLOG 2026-10-16).
MIN_VALID_RATE = 0.95
MIN_CLEAN_PASS = 0.80
MIN_DEFECT_REJECT = 0.70


def rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-id", "--input-dir", type=Path, required=True,
                        help="Labelling directory from prepare_subagent_label_batches (contains responses/)")
    parser.add_argument("--key-file", type=Path, required=True, help="Hidden answer key JSONL")
    parser.add_argument("-od", "--output-dir", type=Path, required=True, help="Verifier artifacts and reports")
    parser.add_argument("-pf", "-p", "--prompt-file", type=Path, default=ROOT / "prompts/full-tags-prompt.md",
                        help="Prompt the subagents followed (default: prompts/full-tags-prompt.md)")
    parser.add_argument("-m", "--model", default="gemini-3.8-flash-subagent",
                        help="Model label recorded in artifacts (default: gemini-3.8-flash-subagent)")
    parser.add_argument("--subagents-file", type=Path,
                        help="Optional JSON mapping batch number (or clip ID for retries) to subagent conversation ID")
    parser.add_argument("-w", "-ow", "--overwrite", action="store_true", help="Replace existing artifacts")
    args = parser.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8")
    known = _known_prompts()
    subagents = json.loads(args.subagents_file.read_text(encoding="utf-8")) if args.subagents_file else {}
    key = [json.loads(line) for line in args.key_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    responses_dir = args.input_dir / "responses"
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_dir} is not empty; pass --overwrite to replace it")

    rows, labels = [], []
    for index, item in enumerate(key):
        response_path = responses_dir / f"{item['id']}.txt"
        audio = resolve_stored_path(item["audio_path"])
        row = {**item, "error": None, "decision": None, "codes": []}
        if not response_path.is_file():
            row["error"] = "missing_response"
        elif digest(audio) != item["audio_sha256"]:
            row["error"] = "audio_hash_mismatch"
        else:
            raw = response_path.read_text(encoding="utf-8")
            try:
                verdict = parse_verifier_response(raw)
                verdict.pop("_raw_response", None)
                _, error = _validate_verdict(verdict, prompt, "subagent", known)
            except VerifierResponseError as exc:
                verdict, error = None, exc.code
            row["error"] = error
            if verdict is not None and error is None:
                row["decision"] = verdict["decision"]
                row["codes"] = verdict["failure_codes"]
                row["has_transcript"] = bool(verdict.get("transcript"))
            if item["role"] == "target":
                source = resolve_stored_path(item["source_path"])
                family = output_dir / item["source_dir"]
                raw_copy = family / f"{source.stem}_subagent.txt"
                family.mkdir(parents=True, exist_ok=True)
                raw_copy.write_text(raw, encoding="utf-8")
                artifact = {
                    "schema_version": 1,
                    "source": {"path": persist_path(source), "sha256": item["source_sha256"]},
                    "operation": "verify", "model": "subagent",
                    "parameters": {"model": args.model, "prompt_file": persist_path(args.prompt_file),
                                   "prompt": prompt, "label_audio": item["audio_path"],
                                   "label_audio_sha256": item["audio_sha256"], "label_id": item["id"],
                                   "batch": item["batch"],
                                   "subagent": subagents.get(item["id"], subagents.get(str(item["batch"])))},
                    "status": "success" if verdict is not None else "failed",
                    "response": {"path": persist_path(raw_copy), "format": "utf-8 text", "kind": "text",
                                 "bytes": raw_copy.stat().st_size, "sha256": digest(raw_copy)},
                    "verdict": verdict if verdict is not None else {"error": error},
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                write_json(family / f"{source.stem}_subagent.json", artifact)
                labels.append({"source_path": item["source_path"], "source_sha256": item["source_sha256"],
                               "recording": item["source_dir"], "label_id": item["id"], "error": error,
                               "decision": row["decision"], "failure_codes": row["codes"]})
        rows.append(row)
        progress("COLLECT", item["id"], current=index + 1, total=len(key))

    valid = [r for r in rows if r["error"] is None]
    controls = defaultdict(list)
    for r in rows:
        if r["role"] == "control":
            controls[r["control_type"]].append(r)
    per_control = {}
    for name, items in sorted(controls.items()):
        scored = [r for r in items if r["error"] is None]
        per_control[name] = {
            "n": len(items), "valid": len(scored),
            "correct_decision": sum(r["decision"] == r["expected_decision"] for r in scored),
            "expected_code_hit": sum(r["expected_code"] in r["codes"] for r in scored) if items[0]["expected_code"] else None,
            "decisions": dict(Counter(r["decision"] for r in scored)),
            "codes": dict(Counter(code for r in scored for code in r["codes"])),
        }
    clean = [r for r in controls.get("clean_anchor", []) if r["error"] is None]
    defects = [r for name, items in controls.items() if name not in {"clean_anchor", "digital_silence"}
               for r in items if r["error"] is None]
    silence_passed = [r["id"] for r in controls.get("digital_silence", []) if r["decision"] == "pass"]
    checks = {
        "valid_schema_rate": rate(len(valid), len(rows)),
        "clean_anchor_pass_rate": rate(sum(r["decision"] == "pass" for r in clean), len(clean)),
        "strong_defect_reject_rate": rate(sum(r["decision"] == "reject" for r in defects), len(defects)),
        "silence_controls_passed": silence_passed,
    }
    accepted = (
        (checks["valid_schema_rate"] or 0) >= MIN_VALID_RATE
        and (checks["clean_anchor_pass_rate"] or 0) >= MIN_CLEAN_PASS
        and (checks["strong_defect_reject_rate"] or 0) >= MIN_DEFECT_REJECT
        and not silence_passed
    )
    targets = [r for r in rows if r["role"] == "target"]
    by_recording = defaultdict(Counter)
    for r in targets:
        by_recording[r["source_dir"]][r["decision"] or f"error:{r['error']}"] += 1
    report = {
        "labels_usable_as_teacher": accepted,
        "rules": {"min_valid_rate": MIN_VALID_RATE, "min_clean_pass": MIN_CLEAN_PASS,
                  "min_defect_reject": MIN_DEFECT_REJECT, "silence_must_not_pass": True},
        "checks": checks, "controls": per_control,
        "targets": {"n": len(targets), "by_recording": {k: dict(v) for k, v in sorted(by_recording.items())},
                    "codes": dict(Counter(code for r in targets for code in r["codes"]))},
        "errors": dict(Counter(r["error"] for r in rows if r["error"])),
        "inputs": {"key_file": persist_path(args.key_file), "key_sha256": digest(args.key_file),
                   "prompt_file": persist_path(args.prompt_file), "prompt_sha256": digest(args.prompt_file),
                   "responses_dir": persist_path(responses_dir)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "subagent_label_report.json", report)
    with (output_dir / "control_rows.jsonl").open("w", encoding="utf-8") as stream:
        for r in rows:
            if r["role"] == "control":
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (output_dir / "labels.jsonl").open("w", encoding="utf-8") as stream:
        for label in labels:
            stream.write(json.dumps(label, ensure_ascii=False) + "\n")
    progress("DONE", f"teacher-usable={accepted} valid={checks['valid_schema_rate']} "
                     f"clean_pass={checks['clean_anchor_pass_rate']} defect_reject={checks['strong_defect_reject_rate']}")
    print(output_dir / "subagent_label_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
