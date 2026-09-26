"""Unwrap pronunciations, square filler tags, and randomly remove CSV pause markers."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import ROOT, LoggingArgumentParser, progress
from _transcript_emotions import EMOTIONS

# Capture flat or singly nested groups; payload validation rejects nested groups.
BRACKETS = re.compile(r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]')
LEGACY_CONSONANT = re.compile(r'<(?:s|x|sh|zh|ph|ch|g|v|c|p|b|đ|th|t|d|k|f|l|r|z|h|m|n)>')

# Skip remaining square annotations; convert only standalone angle-tag tokens.
FILLER_TAG = re.compile(BRACKETS.pattern + r'|<(?P<label>[a-zđ]+(?:[_-][a-zđ]+)*)>')


def square_filler_tags(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        label = match.group('label')
        if label is None:
            return match.group()
        start, end = match.span()
        # A consonant belongs to ViePhoneme only when joined to another block.
        joined_left = (start >= 2 and text[start - 1] == '-'
                       and (text[start - 2].isalpha() or text[start - 2] == '>'))
        joined_right = (end + 1 < len(text) and text[end] == '-'
                        and (text[end + 1].isalpha() or text[end + 1] == '<'))
        if LEGACY_CONSONANT.fullmatch(match.group()) and (joined_left or joined_right):
            return match.group()
        count += 1
        return f'[{label}]'

    return FILLER_TAG.sub(replace, text), count


def unwrap_pronunciation_brackets(text: str, preserved: set[str]) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        payload = match.group()[1:-1]
        if payload.strip().casefold() in preserved:
            return match.group()
        # Keep slash delimiters, stress/length marks and internal IPA spaces.
        if (re.fullmatch(r'/[^/\[\]<>_\-\r\n\t]+/', payload)
                and payload[1:-1].strip()):
            count += 1
            return payload
        # Historical spoken CSVs also use forms such as [<s>-trét].
        letters = LEGACY_CONSONANT.sub('s', payload)
        if (not any(c.isalpha() for c in letters)
                or not all(c in '_-' or (c.isalpha() and 'LATIN' in unicodedata.name(c, ''))
                           or unicodedata.category(c).startswith('M') for c in letters)):
            return match.group()
        count += 1
        return payload

    return BRACKETS.sub(replace, text), count


def main() -> int:
    parser = LoggingArgumentParser(description=__doc__)
    parser.add_argument('--input-file', '-if', type=Path, required=True, help='Spoken transcript CSV')
    parser.add_argument('--output-file', '-of', type=Path, required=True, help='New CSV under .data/')
    parser.add_argument('--transcript-column', help='Default: detect transcript or transcripts; require a choice if both exist')
    parser.add_argument('--remove-tilde-ratio', type=float, default=0.8, help='Fraction of all ~ occurrences to remove (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--preserve-tag', action='append', default=[], help='Additional bracket label to preserve; repeat as needed')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output after successful cleanup')
    args = parser.parse_args()
    source = args.input_file.expanduser().resolve()
    destination = args.output_file.expanduser().resolve()
    if not math.isfinite(args.remove_tilde_ratio) or not 0 <= args.remove_tilde_ratio <= 1:
        parser.error('--remove-tilde-ratio must be between 0 and 1')
    if not source.is_file():
        parser.error('Input CSV does not exist')
    if source == destination or (destination.exists() and source.samefile(destination)):
        parser.error('Input and output must differ')
    if not destination.is_relative_to((ROOT / '.data').resolve()) or destination.suffix.lower() != '.csv':
        parser.error('--output-file must be a .csv path under .data/')
    if destination.exists() and not args.overwrite:
        parser.error('Output already exists; use another path or --overwrite')
    try:
        with source.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            candidates = [i for i, name in enumerate(header)
                          if name == args.transcript_column] if args.transcript_column else [
                              i for i, name in enumerate(header) if name in ('transcript', 'transcripts')]
            if len(candidates) != 1:
                raise ValueError('Specify --transcript-column matching exactly one CSV column')
            column = candidates[0]
            rows = []
            for number, row in enumerate(reader, 2):
                if len(row) != len(header):
                    raise ValueError(f'CSV record {number}: expected {len(header)} cells, got {len(row)}')
                rows.append(row)

        preserved = set(EMOTIONS) | {tag.strip().casefold() for tag in args.preserve_tag}
        total_tildes = sum(row[column].count('~') for row in rows)
        remove_count = math.floor(total_tildes * args.remove_tilde_ratio)
        selected = set(random.Random(args.seed).sample(range(total_tildes), remove_count))
        occurrence = 0
        unwrapped = changed = converted_tags = 0
        for row in rows:
            original = row[column]
            text, count = unwrap_pronunciation_brackets(original, preserved)
            unwrapped += count
            text, count = square_filler_tags(text)
            converted_tags += count
            pieces = []
            for char in text:
                if char == '~':
                    if occurrence not in selected:
                        pieces.append(char)
                    occurrence += 1
                else:
                    pieces.append(char)
            row[column] = ''.join(pieces)
            changed += row[column] != original

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='transcript-cleanup-', dir=destination.parent) as temporary:
            staged = Path(temporary) / 'cleaned_transcripts.csv'
            with staged.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
            if args.overwrite:
                os.replace(staged, destination)
            else:
                os.link(staged, destination)
        progress('TRANSCRIPT_CLEANUP', f'{len(rows)} rows; {changed} changed; {unwrapped} ViePhoneme/IPA brackets removed; '
                 f'{converted_tags} filler/sound tags converted; '
                 f'{remove_count}/{total_tildes} tildes removed; seed={args.seed}')
        print(destination)
        return 0
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
