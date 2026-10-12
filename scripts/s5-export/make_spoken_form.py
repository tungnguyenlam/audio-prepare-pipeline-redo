#!/usr/bin/env python3
"""Create spoken-form transcripts with verified audio paths."""

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


# Skip existing bracket payloads so conversion is safe to repeat. Match complete
# attached tokens, including M&A, numeric dates/fractions and apostrophes.
ANNOTATED_WORD = re.compile(
    r"\[[^\[\]\n]*\]|"
    r"(?<!\w)[^\W_]+(?:[-_’'&/+][^\W_]+|(?<=\d)[.,:](?=\d)[^\W_]+)*"
    r"(?P<pronunciation>\[[^\[\]\n]+\])",
    re.UNICODE,
)
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def to_spoken_form(text: str) -> str:
    return ANNOTATED_WORD.sub(lambda match: match["pronunciation"] or match[0], text)


def index_audio(root: Path) -> dict[str, list[Path]]:
    index = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            index[path.stem].append(path.resolve())
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "--input", dest="input", type=Path, required=True, help="CSV with transcripts and either id or audio-path")
    parser.add_argument("--audio-root", type=Path, required=True, help="Search recursively for audio")
    parser.add_argument(
        "--output-file", "--output", dest="output", type=Path, default=Path(".data/spoken-form-thu-am-studio-transcript.csv")
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input CSV does not exist: {args.input}")
    if not args.audio_root.is_dir():
        parser.error(f"Audio directory does not exist: {args.audio_root}")

    audio_index = index_audio(args.audio_root)
    print(f"Indexed {sum(map(len, audio_index.values()))} audio files.", file=sys.stderr)
    rows = []
    problems = []

    with args.input.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        key = "id" if "id" in columns else "audio-path"
        missing = {key, "transcripts"} - columns
        if missing:
            parser.error(f"Missing CSV columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(key) is None for key in (key, "transcripts")):
                problems.append(f"CSV line {line_number}: malformed CSV row")
                continue
            sample_id = row[key].strip()
            if key == "audio-path":
                sample_id = Path(sample_id).stem
            matches = audio_index.get(sample_id, [])
            if len(matches) != 1:
                explanation = "no audio" if not matches else f"multiple audio files: {matches}"
                problems.append(f"CSV line {line_number}, id={sample_id}: {explanation}")
                continue
            rows.append({
                "stt": len(rows) + 1,
                "audio-path": matches[0].name,
                "transcripts": to_spoken_form(row["transcripts"]),
            })

    if problems:
        print(f"Could not convert {len(problems)} rows:", file=sys.stderr)
        for problem in problems[:30]:
            print(f"  {problem}", file=sys.stderr)
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more", file=sys.stderr)
        raise SystemExit("No output written. Check --audio-root and duplicate audio basenames.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Read and validate everything before atomically replacing the output;
    # this also permits converting the input CSV in place.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="",
            dir=args.output.parent, prefix=f".{args.output.name}.", delete=False,
        ) as destination:
            temporary = Path(destination.name)
            writer = csv.DictWriter(destination, fieldnames=["stt", "audio-path", "transcripts"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Rows written: {len(rows)}", file=sys.stderr)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
