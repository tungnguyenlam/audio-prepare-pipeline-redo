"""Filter an existing file manifest by tags and duration; never processes audio."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, identity, persist_path, progress, read_json, resolve_stored_path, write_json


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-im', '--input-manifest', type=Path, required=True, help='Path to input manifest JSON')
    p.add_argument('-om', '--output-manifest', type=Path, required=True, help='Path to output filtered manifest JSON')
    p.add_argument('-t', '--tag', action='append', default=[], help='Require every supplied tag')
    p.add_argument('-xt', '--exclude-tag', action='append', default=[], help='Exclude any entry matching these tags')
    p.add_argument('-min', '--min-duration', type=float, default=0, help='Minimum duration in seconds')
    p.add_argument('-max', '--max-duration', type=float, help='Maximum duration in seconds')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true', help='Overwrite existing output manifest')
    p.add_argument('-c', '--concurrency', type=int, default=1, help='Number of concurrent workers for filtering entries. Set > 1 to enable concurrent execution')
    p.add_argument('-b', '-bs', '--batch-size', type=int, default=1, help='Batch size for processing manifest entries')
    args = p.parse_args()
    if args.concurrency < 1:
        p.error('--concurrency must be at least 1')
    if args.batch_size < 1:
        p.error('--batch-size must be at least 1')
    if args.input_manifest.resolve() == args.output_manifest.resolve():
        p.error('Input and output manifests must differ')
    if not math.isfinite(args.min_duration) or args.min_duration < 0 or (args.max_duration is not None and (not math.isfinite(args.max_duration) or args.max_duration < args.min_duration)):
        p.error('Require finite 0 <= min-duration <= max-duration')
    metadata = {'schema_version': 1, 'operation': 'filter', 'source': identity(args.input_manifest),
                'parameters': {'tags': sorted(set(args.tag)), 'exclude_tags': sorted(set(args.exclude_tag)),
                               'min_duration': args.min_duration, 'max_duration': args.max_duration}}
    source = read_json(args.input_manifest)
    raw_entries = source.get('entries', [])
    total = len(raw_entries)
    progress('FILTER_START', f'Filtering {total} entries from {args.input_manifest.name}')

    req_tags = set(args.tag)
    excl_tags = set(args.exclude_tag)

    def _filter_batch(batch_items):
        kept = []
        for entry in batch_items:
            e_tags = set(entry.get('tags', []))
            if not req_tags <= e_tags:
                continue
            if excl_tags & e_tags:
                continue
            if entry.get('duration_s', 0.0) < args.min_duration:
                continue
            if args.max_duration is not None and entry.get('duration_s', 0.0) > args.max_duration:
                continue
            e_copy = dict(entry)
            audio = resolve_stored_path(e_copy['path'], base=args.input_manifest.parent)
            e_copy['path'] = persist_path(audio)
            kept.append(e_copy)
        return kept

    batches = [raw_entries[i:i + args.batch_size] for i in range(0, total, args.batch_size)]
    entries = []
    if args.concurrency > 1 and len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
            for b_kept in pool.map(_filter_batch, batches):
                entries.extend(b_kept)
    else:
        for b in batches:
            entries.extend(_filter_batch(b))
    output = {**metadata, 'entries': entries, 'complete': source.get('complete', True)}
    if args.output_manifest.exists() and not args.overwrite:
        if read_json(args.output_manifest) != output:
            p.error('Conflicting output; use --overwrite')
    else:
        write_json(args.output_manifest, output)
    progress('FILTER_DONE', f'Retained {len(entries)} of {total} entries -> {args.output_manifest.name}')
    print(args.output_manifest.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
