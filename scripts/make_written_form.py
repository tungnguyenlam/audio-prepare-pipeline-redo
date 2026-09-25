#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from pathlib import Path


# Khớp một từ/số ngay trước dấu [...], kể cả từ có dấu tiếng Việt.
# Ví dụ: VinUni[vin-iu-ni], top[tóp], 100[một_trăm]
ATTACHED_PRONUNCIATION = re.compile(
    r"(?<=[^\W_])\[[^\[\]\n]*\]",
    flags=re.UNICODE,
)


def remove_pronunciation(text: str) -> str:
    return ATTACHED_PRONUNCIATION.sub("", text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove attached IPA/ViePhoneme annotations from transcript CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV from extract_transcripts-spoken-form.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".data/written-form-thu-am-studio-transcript.csv"),
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input CSV does not exist: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with (
        args.input.open("r", encoding="utf-8-sig", newline="") as source,
        args.output.open("w", encoding="utf-8-sig", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        required = {"id", "path", "transcripts"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            parser.error(f"Missing CSV columns: {', '.join(sorted(missing))}")

        writer = csv.DictWriter(
            destination,
            fieldnames=["stt", "id", "path", "transcripts"],
        )
        writer.writeheader()

        for row in reader:
            count += 1
            writer.writerow({
                "stt": count,
                "id": row["id"],
                "path": row["path"],
                "transcripts": remove_pronunciation(row["transcripts"]),
            })

    print(f"Rows written: {count}", file=sys.stderr)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
