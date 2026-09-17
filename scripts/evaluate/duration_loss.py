"""Audit audio duration loss and yield across diarization, silence, and post-merge filtering.

Computes:
1. Time lost from raw audio to raw diarization turns (non-speech / acoustic silence).
2. Time lost due to silence (raw non-speech, silence bridged in merge, unmerged gaps).
3. Time lost due to post-merge duration filtering (< 1.5s too-short and > 15.0s too-long).
4. Final valid clip duration, count, and yield rates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.duration_loss import (
    aggregate_audits,
    audit_item,
    render_csv,
    render_single_detail,
    render_table,
)
from _common.files import (
    LoggingArgumentParser,
    positive_int,
    progress,
    write_json,
)


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
