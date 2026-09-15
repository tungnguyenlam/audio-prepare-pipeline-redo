"""Print audio properties as JSON Lines (read-only)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import inputs, parser, persist_path, probe, progress


def main() -> int:
    import concurrent.futures
    import threading

    p = parser(__doc__, 'info', segments=True)
    args = p.parse_args()
    items = inputs(args)
    total = len(items)
    progress('INFO_START', f'Probing {total} audio file(s) (concurrency={args.concurrency}, batch_size={args.batch_size})')
    failed = 0
    completed = 0
    lock = threading.Lock()

    batches = [items[i:i + args.batch_size] for i in range(0, total, args.batch_size)]

    def probe_item(item_idx: int, src: Path) -> tuple[bool, str, str]:
        try:
            line = json.dumps({'path': persist_path(src), **probe(src)})
            return True, line, src.name
        except Exception as exc:
            return False, '', f'{src.name}: {exc}'

    if args.concurrency <= 1:
        for batch_chunk in batches:
            for src, _ in batch_chunk:
                completed += 1
                success, line, msg = probe_item(completed, src)
                if success:
                    print(line, flush=True)
                    progress('PROBED', msg, current=completed, total=total)
                else:
                    failed += 1
                    progress('PROBE_FAIL', msg, current=completed, total=total)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for batch_chunk in batches:
                futures = []
                for src, _ in batch_chunk:
                    with lock:
                        completed += 1
                        cur_idx = completed
                    futures.append(pool.submit(probe_item, cur_idx, src))
                for future in concurrent.futures.as_completed(futures):
                    success, line, msg = future.result()
                    with lock:
                        if success:
                            print(line, flush=True)
                            progress('PROBED', msg, current=completed, total=total)
                        else:
                            failed += 1
                            progress('PROBE_FAIL', msg, current=completed, total=total)

    progress('INFO_DONE', f'{total - failed} probed; {failed} failed')
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
