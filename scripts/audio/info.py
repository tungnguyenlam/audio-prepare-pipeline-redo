"""Print audio properties as JSON Lines (read-only)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import inputs, parser, probe


def main() -> int:
    p = parser(__doc__, 'info', segments=True)
    args = p.parse_args()
    failed = 0
    for src, _ in inputs(args):
        try:
            print(json.dumps({'path': str(src), **probe(src)}))
        except Exception as exc:
            failed += 1
            print(f'FAILED {src}: {exc}', file=sys.stderr)
    print(f'{failed} failed', file=sys.stderr)
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
