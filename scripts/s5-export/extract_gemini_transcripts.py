#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def extract_transcript(text: str) -> str | None:
    text = text.strip()

    # Trường hợp toàn bộ file là JSON.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            value = data.get("transcript") or data.get("transcripts")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except json.JSONDecodeError:
        pass

    # Trường hợp file văn bản có dòng "transcript: ...".
    match = re.search(
        r"""(?im)^\s*(?:[-*]\s*)?(?:\*\*)?["']?transcripts?["']?
            (?:\*\*)?\s*[:：]\s*(.+?)\s*$""",
        text,
        flags=re.VERBOSE,
    )
    if match:
        value = match.group(1).strip().rstrip(",").strip("\"'` ")
        if value and value.lower() not in {"null", "none", "n/a", "unknown"}:
            return value

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract transcripts from *_gemini.txt files into a CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.input.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Input directory does not exist: {root}")

    files = sorted(root.rglob("*_gemini.txt"))
    rows = []
    skipped = []

    for file in files:
        try:
            content = file.read_text(encoding="utf-8-sig")
            transcript = extract_transcript(content)
        except (OSError, UnicodeError) as exc:
            skipped.append((file, str(exc)))
            continue

        if transcript is None:
            skipped.append((file, "transcript field not found"))
            continue

        rows.append({
            "id": file.name.removesuffix("_gemini.txt"),
            "path": file.relative_to(root).as_posix(),
            "transcripts": transcript,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stt", "id", "path", "transcripts"],
        )
        writer.writeheader()

        for stt, row in enumerate(rows, start=1):
            writer.writerow({"stt": stt, **row})

    print(f"TXT files found: {len(files)}", file=sys.stderr)
    print(f"CSV rows written: {len(rows)}", file=sys.stderr)
    print(f"Skipped: {len(skipped)}", file=sys.stderr)

    for file, reason in skipped[:20]:
        print(f"  SKIP {file.relative_to(root)}: {reason}", file=sys.stderr)

    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more", file=sys.stderr)

    print(args.output.resolve())


if __name__ == "__main__":
    main()
