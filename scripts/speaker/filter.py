"""Filter scored speaker segments by similarity threshold, duration, and overlap."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, identity, probe, progress, read_json, write_json
from _common.segments import normalize_turns, source_path


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True, help='Manifest with scored segments')
    p.add_argument('--output-manifest', type=Path, required=True, help='Filtered output manifest')
    p.add_argument('--threshold', type=float, default=0.6, help='Minimum similarity threshold')
    p.add_argument('--min-duration-s', type=float, default=1.5, help='Minimum segment duration in seconds')
    p.add_argument('--exclude-overlap', action=argparse.BooleanOptionalAction, default=True, help='Exclude turns overlapping other speakers')
    p.add_argument('--input-file', type=Path, help='Optional source audio override')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    manifest = read_json(args.input_manifest)
    source = source_path(manifest, args.input_manifest, args.input_file)
    raw_turns = manifest.get('turns', [])
    progress('FILTER_START', f'Filtering {len(raw_turns)} turns (threshold>={args.threshold}, min_dur>={args.min_duration_s}s)')

    kept_turns = []
    for turn in raw_turns:
        dur = turn.get('end_s', 0.0) - turn.get('start_s', 0.0)
        sim = turn.get('similarity', -1.0)
        overlaps = turn.get('overlaps_other_speaker', turn.get('overlap', False))
        if sim < args.threshold:
            continue
        if dur < args.min_duration_s:
            continue
        if args.exclude_overlap and overlaps:
            continue
        kept_turns.append(turn)

    normalized = normalize_turns(kept_turns, source)
    dest = args.output_manifest.resolve()
    metadata = {
        'schema_version': 1,
        'source': identity(source),
        'timestamp_origin': 'diarized_input',
        'operation': 'speaker_filter',
        'parameters': {
            'threshold': args.threshold,
            'min_duration_s': args.min_duration_s,
            'exclude_overlap': args.exclude_overlap,
            'input_manifest': identity(args.input_manifest),
        },
        'turns': normalized,
        'speaker_ids': sorted({t['speaker_id'] for t in normalized}),
        'source_sample_rate': probe(source)['sample_rate'],
        'clips_valid': False,
        'complete': True,
    }
    if dest.exists() and not args.overwrite:
        old = read_json(dest)
        if all(old.get(k) == v for k, v in metadata.items() if k != 'turns'):
            print(dest)
            return 0
        p.error(f'Conflicting output: {dest}; use --overwrite')
    write_json(dest, metadata)
    progress('FILTER_DONE', f'Kept {len(normalized)} of {len(raw_turns)} turns -> {dest.name}')
    print(dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
