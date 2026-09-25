"""Export verified audio with paired spoken/written transcript CSVs in one ZIP."""
from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import ROOT, LoggingArgumentParser, progress
from _verifier_export_inventory import collect_run

EMOTIONS = frozenset('''neutral calm excited happy amused playful proud warm tender grateful
relieved hopeful angry frustrated annoyed impatient anxious fearful panicked disgusted sad
 disappointed hurt worried apologetic embarrassed tired bored nostalgic surprised shocked
 amazed curious confused hesitant skeptical confident determined serious pleading sarcastic
 contemptuous'''.split())


def split_transcript_forms(text: str) -> tuple[str, str]:
    """Split current word[pronunciation] grammar; preserve payloads and non-neutral tags."""
    spoken: list[str] = []
    written: list[str] = []
    i = 0

    def fail(position: int, message: str) -> None:
        raise ValueError(f'character {position + 1}: {message}')

    def boundary(position: int) -> bool:
        char = text[position]
        if char in '.,:/':
            # Decimal/grouped numbers, dates and times are one written token.
            return not (position > 0 and position + 1 < len(text)
                        and text[position - 1].isdigit() and text[position + 1].isdigit())
        if char in "'’":
            return not (position > 0 and position + 1 < len(text)
                        and text[position - 1].isalpha() and text[position + 1].isalpha())
        return char.isspace() or char in '[]<>!?~*;"“”‘(){}«»'

    def bracket(position: int) -> tuple[str, int]:
        end = text.find(']', position + 1)
        if end < 0:
            fail(position, 'unclosed square bracket')
        payload = text[position + 1:end]
        if not payload.strip() or any(char in '[]<>' for char in payload):
            fail(position, 'empty or nested annotation')
        return payload, end + 1

    while i < len(text):
        char = text[i]
        if char == '[':
            label, end = bracket(i)
            if label not in EMOTIONS or (i and not text[i - 1].isspace()):
                fail(i, 'unsupported standalone bracket; expected a separated emotion or word[pronunciation]')
            if end < len(text) and not text[end].isspace() and text[end] not in '.,!?~*;:':
                fail(end, 'emotion must be separated from speech')
            if label != 'neutral':
                spoken.append(text[i:end])
                written.append(text[i:end])
            else:
                # Remove only whitespace made redundant by deleting this tag.
                following = end
                while following < len(text) and text[following].isspace():
                    following += 1
                for output in (spoken, written):
                    if following == len(text) or text[following] in '.,!?~*;:':
                        while output and output[-1].isspace():
                            output.pop()
                if not spoken or spoken[-1].isspace() or following == len(text) or text[following] in '.,!?~*;:':
                    end = following
            i = end
        elif char == '<':
            end = text.find('>', i + 1)
            if end < 0 or not re.fullmatch(r'[a-z]+(?:[_-][a-z]+)*', text[i + 1:end]):
                fail(i, 'invalid sound/filler tag')
            spoken.append(text[i:end + 1])
            written.append(text[i:end + 1])
            i = end + 1
        elif char in ']>':
            fail(i, 'unmatched closing bracket')
        elif boundary(i):
            spoken.append(char)
            written.append(char)
            i += 1
        else:
            start = i
            while i < len(text) and not boundary(i):
                i += 1
            word = text[start:i]
            written.append(word)
            if i < len(text) and text[i] == '[':
                if start >= 2 and text[start - 1] in '.:/' and text[start - 2].isalnum():
                    fail(start, 'ambiguous compound written token; annotate each word separately')
                payload, end = bracket(i)
                if '/' in payload:
                    if not (payload.startswith('/') and payload.endswith('/') and len(payload) > 2
                            and payload[1:-1].strip() and '/' not in payload[1:-1]
                            and not any(c in '_-\r\n\t' for c in payload)):
                        fail(i, 'IPA must be enclosed in /.../ without mixed ViePhoneme markers')
                elif not all(c in '_-' or (c.isalpha() and 'LATIN' in unicodedata.name(c, ''))
                             or unicodedata.category(c).startswith('M') for c in payload):
                    fail(i, 'unsupported ViePhoneme payload; expected Latin letters joined by _ or -')
                if not any(c.isalpha() for c in payload):
                    fail(i, 'pronunciation has no letters')
                spoken.append(payload)
                i = end
            else:
                spoken.append(word)
    forms = ''.join(spoken).strip(), ''.join(written).strip()
    if not all(forms):
        raise ValueError('empty transcript after removing [neutral]')
    return forms


def write_tts_zip(destination: Path, records: list[dict], overwrite: bool) -> None:
    """Hash the exact bytes streamed into ZIP64, then publish the finished archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tts-export-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'dataset.zip'
        with zipfile.ZipFile(staged, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for index, row in enumerate(records, 1):
                checksum = hashlib.sha256()
                with row['_source'].open('rb') as source, archive.open(row['audio_path'], 'w', force_zip64=True) as target:
                    while chunk := source.read(1024 * 1024):
                        checksum.update(chunk)
                        target.write(chunk)
                if checksum.hexdigest() != row['source_sha256']:
                    raise ValueError(f"{row['clip_id']}: audio changed during export")
                if index == 1 or index == len(records) or index % max(1, len(records) // 10) == 0:
                    progress('TTS_EXPORT_AUDIO', row['audio_path'], current=index, total=len(records))
            for form in ('spoken', 'written'):
                with archive.open(f'{form}.csv', 'w', force_zip64=True) as binary:
                    with io.TextIOWrapper(binary, encoding='utf-8', newline='') as stream:
                        writer = csv.writer(stream)
                        writer.writerow(('audio_path', 'transcript'))
                        writer.writerows((row['audio_path'], row[form]) for row in records)
        if overwrite:
            os.replace(staged, destination)
        else:
            # Atomic publication without clobbering a concurrently created output.
            os.link(staged, destination)


def main() -> int:
    parser = LoggingArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', '-id', type=Path, required=True, help='Verifier directory searched recursively')
    parser.add_argument('--input-manifest', '-im', type=Path, action='append', required=True,
                        help='Complete indexed audio or exported segments inventory; repeat for disjoint inventories')
    parser.add_argument('--output-file', '-of', type=Path, required=True, help='Exact ZIP destination under .data/')
    parser.add_argument('--configuration', help='Select a unique settings hash or prefix for competing verdicts')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing ZIP only after export completes')
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    destination = args.output_file.resolve()
    if not directory.is_dir():
        parser.error('Input directory does not exist')
    if not destination.is_relative_to((ROOT / '.data').resolve()) or destination.suffix.lower() != '.zip':
        parser.error('--output-file must be a .zip path under .data/')
    if destination.is_relative_to(directory) or destination in {p.resolve() for p in args.input_manifest}:
        parser.error('Output must be outside the verifier run and differ from input manifests')
    if destination.exists() and not args.overwrite:
        parser.error('Output already exists; choose another path or use --overwrite')
    try:
        progress('TTS_EXPORT_SCAN', 'Checking inventory, verdicts and source integrity')
        rows, summary = collect_run(directory, args.input_manifest, args.configuration,
                                    progress_tag='TTS_EXPORT_SELECT')
        if summary['processing_status'] != 'complete':
            issues = [f"{row['clip_id']}: {row['verifier_status']}; {row['reason']}" for row in rows
                      if row['verifier_status'] not in ('pass', 'reject') or row.get('audio_issue')]
            raise ValueError(f"Incomplete verifier run: coverage={summary['coverage']}; outcomes={summary['counts']}"
                             + ('\n' + '\n'.join(issues[:20]) if issues else ''))
        if destination in {row['_source'] for row in rows if '_source' in row}:
            raise ValueError('Output must differ from every source audio')
        records = [row for row in rows if row['verifier_status'] == 'pass']
        progress('TTS_EXPORT_SELECT', f"Selected {len(records)} pass; excluded {summary['counts'].get('reject', 0)} reject")
        if not records:
            raise ValueError('No passed clips to export')
        ids = set()
        progress('TTS_EXPORT_PARSE', f'Splitting {len(records)} transcripts')
        for row in records:
            if row['clip_id'] in ids:
                raise ValueError(f"Duplicate clip ID: {row['clip_id']}")
            ids.add(row['clip_id'])
            if not row['transcript']:
                raise ValueError(f"{row['clip_id']}: profile {row['schema_profile']} has no transcript; cannot export TTS")
            try:
                row['spoken'], row['written'] = split_transcript_forms(row['transcript'])
            except ValueError as exc:
                raise ValueError(f"{row['clip_id']}: {exc}") from None
        write_tts_zip(destination, records, args.overwrite)
        progress('TTS_EXPORT_DONE', f'{len(records)} audio files and paired CSV rows -> {destination.name}')
        print(destination)
        return 0
    except (ValueError, OSError, KeyError, TypeError, zipfile.LargeZipFile) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
