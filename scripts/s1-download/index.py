"""Rebuild index.jsonl for every directory under --input-dir that holds download sidecars."""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, progress
from youtube import INDEX_NAME, write_index


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-id', '--input-dir', type=Path, default=ROOT / '.data/s1-download',
                   help='Root to scan recursively; each directory with download sidecars gets its own index')
    args = p.parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f'Input directory not found: {args.input_dir}')
    directories = sorted({path.parent for path in args.input_dir.rglob('*.json') if path.name != INDEX_NAME})
    written = 0
    for directory in directories:
        dest = write_index(directory)
        if dest is None:
            continue
        written += 1
        progress('INDEX', f'{dest.relative_to(args.input_dir) if dest.is_relative_to(args.input_dir) else dest}')
        print(dest, flush=True)
    progress('COMPLETE', f'Wrote {written} index file(s) under {args.input_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
