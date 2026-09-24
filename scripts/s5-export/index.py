"""Snapshot a directory into a file-based audio manifest; no registry service."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import ROOT, identity, inputs, parser, persist_path, probe, progress, read_json, write_json


def main() -> int:
    p = parser(__doc__, 's5-export', segments=True)
    p.add_argument('-om', '--output-manifest', type=Path, help='Explicit destination path for output manifest JSON')
    p.add_argument('-t', '--tag', action='append', default=[], help='Tag to associate with indexed entries in manifest (repeatable)')
    args = p.parse_args()
    sources = inputs(args)
    destination = (args.output_manifest or args.output_dir / 'manifest.json').resolve()
    if destination in {src.resolve() for src, _ in sources}:
        p.error('Cannot overwrite an input')
    total = len(sources)
    progress('INDEX_START', f'Indexing {total} audio files')

    def _index_item(src, relative):
        return {'relative_path': relative.as_posix() if isinstance(relative, Path) else str(relative),
                **identity(src), **probe(src), 'tags': sorted(set(args.tag))}

    def _index_batch(batch_items):
        b_entries = []
        b_failures = []
        for src, relative in batch_items:
            try:
                b_entries.append(_index_item(src, relative))
            except Exception as exc:
                b_failures.append({'path': persist_path(src), 'error': str(exc)})
        return b_entries, b_failures

    batches = [sources[i:i + args.batch_size] for i in range(0, total, args.batch_size)]
    entries, failures = [], []
    if args.concurrency > 1 and len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
            for b_entries, b_failures in pool.map(_index_batch, batches):
                entries.extend(b_entries)
                failures.extend(b_failures)
    else:
        for b in batches:
            b_entries, b_failures = _index_batch(b)
            entries.extend(b_entries)
            failures.extend(b_failures)
    manifest = {'schema_version': 1, 'operation': 'index', 'entries': entries, 'failures': failures,
                'complete': not failures}
    if destination.exists() and not args.overwrite:
        if read_json(destination) != manifest:
            p.error('Conflicting manifest; use --overwrite')
    else:
        write_json(destination, manifest)
    progress('INDEX_DONE', f'{len(entries)} succeeded; {len(failures)} failed -> {destination.name}')
    print(destination)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
