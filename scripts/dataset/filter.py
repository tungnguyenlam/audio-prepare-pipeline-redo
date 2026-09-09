"""Filter an existing file manifest by tags and duration; never processes audio."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import identity, read_json, write_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--output-manifest', type=Path, required=True)
    p.add_argument('--tag', action='append', default=[], help='Require every supplied tag')
    p.add_argument('--exclude-tag', action='append', default=[])
    p.add_argument('--min-duration', type=float, default=0)
    p.add_argument('--max-duration', type=float)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.input_manifest.resolve() == args.output_manifest.resolve():
        p.error('Input and output manifests must differ')
    if not math.isfinite(args.min_duration) or args.min_duration < 0 or (args.max_duration is not None and (not math.isfinite(args.max_duration) or args.max_duration < args.min_duration)):
        p.error('Require finite 0 <= min-duration <= max-duration')
    metadata = {'schema_version': 1, 'operation': 'filter', 'source': identity(args.input_manifest),
                'parameters': {'tags': sorted(set(args.tag)), 'exclude_tags': sorted(set(args.exclude_tag)),
                               'min_duration': args.min_duration, 'max_duration': args.max_duration}}
    source = read_json(args.input_manifest)
    entries = [entry for entry in source['entries'] if set(args.tag) <= set(entry.get('tags', []))
               and not set(args.exclude_tag) & set(entry.get('tags', []))
               and entry['duration_s'] >= args.min_duration
               and (args.max_duration is None or entry['duration_s'] <= args.max_duration)]
    for entry in entries:
        audio = Path(entry['path'])
        entry['path'] = str(audio if audio.is_absolute() else (args.input_manifest.parent / audio).resolve())
    output = {**metadata, 'entries': entries, 'complete': source.get('complete', True)}
    if args.output_manifest.exists() and not args.overwrite:
        if read_json(args.output_manifest) != output:
            p.error('Conflicting output; use --overwrite')
    else:
        write_json(args.output_manifest, output)
    print(args.output_manifest.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
