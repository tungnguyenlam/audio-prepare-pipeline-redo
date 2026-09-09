"""Snapshot a directory into a file-based audio manifest; no registry service."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import ROOT, identity, inputs, parser, probe, read_json, write_json


def main() -> int:
    p = parser(__doc__, 'dataset', segments=True)
    p.add_argument('--output-manifest', type=Path)
    p.add_argument('--tag', action='append', default=[])
    args = p.parse_args()
    sources = inputs(args)
    destination = (args.output_manifest or args.output_dir / 'manifest.json').resolve()
    if destination in {src.resolve() for src, _ in sources}:
        p.error('Cannot overwrite an input')
    entries, failures = [], []
    for src, relative in sources:
        try:
            entries.append({'path': str(src.resolve()), 'relative_path': str(relative),
                            **identity(src), **probe(src), 'tags': sorted(set(args.tag))})
        except Exception as exc:
            failures.append({'path': str(src), 'error': str(exc)})
            print(f'FAILED {src}: {exc}', file=sys.stderr)
    manifest = {'schema_version': 1, 'operation': 'index', 'entries': entries, 'failures': failures,
                'complete': not failures}
    if destination.exists() and not args.overwrite:
        if read_json(destination) != manifest:
            p.error('Conflicting manifest; use --overwrite')
    else:
        write_json(destination, manifest)
    print(destination)
    print(f'{len(entries)} succeeded; {len(failures)} failed', file=sys.stderr)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
