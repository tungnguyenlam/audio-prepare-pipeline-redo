"""Print audio properties as JSON Lines (read-only)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import inputs, parser, probe, progress


def main() -> int:
    p = parser(__doc__, 'info', segments=True)
    args = p.parse_args()
    items = inputs(args)
    total = len(items)
    progress('INFO_START', f'Probing {total} audio file(s)')
    failed = 0
    for idx, (src, _) in enumerate(items, 1):
        try:
            print(json.dumps({'path': str(src), **probe(src)}))
            progress('PROBED', f'{src.name}', current=idx, total=total)
        except Exception as exc:
            failed += 1
            progress('PROBE_FAIL', f'{src.name}: {exc}', current=idx, total=total)
    progress('INFO_DONE', f'{total - failed} probed; {failed} failed')
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
