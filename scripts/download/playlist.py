"""Download all entries in a playlist, sequentially."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import progress
from youtube import arguments, download


def main(description: str = __doc__) -> int:
    args = arguments(description, bulk=True).parse_args()
    from yt_dlp import YoutubeDL
    options = {'extract_flat': 'in_playlist', 'quiet': True, 'ignoreerrors': True}
    if args.cookie_file:
        options['cookiefile'] = str(args.cookie_file.resolve())
    with contextlib.redirect_stdout(sys.stderr), YoutubeDL(options) as ydl:
        listing = ydl.extract_info(args.url, download=False)
        if not listing or 'entries' not in listing:
            raise ValueError('URL did not resolve to a playlist or channel')
        entries = list(listing['entries'])
    total = len(entries)
    progress('PLAYLIST', f'Found {total} items to process')
    indexed_entries = list(enumerate(entries, 1))
    batches = [indexed_entries[i:i + args.batch_size] for i in range(0, len(indexed_entries), args.batch_size)]
    import threading
    lock = threading.Lock()

    def _process_item(idx, entry):
        if not entry:
            raise ValueError('Unavailable playlist entry')
        title = entry.get('title') or entry.get('id') or 'video'
        url = entry.get('webpage_url') or entry.get('url')
        if not url or not url.startswith('http'):
            url = 'https://www.youtube.com/watch?v=' + entry['id']
        with lock:
            progress('ITEM_START', f'{title}', current=idx, total=total)
        with contextlib.redirect_stdout(sys.stderr):
            dest = download(url, args)
        with lock:
            progress('ITEM_DONE', f'{dest.name}', current=idx, total=total)
            print(dest, flush=True)

    def _process_batch(batch_items):
        batch_failed = 0
        for idx, entry in batch_items:
            try:
                _process_item(idx, entry)
            except Exception as exc:
                batch_failed += 1
                with lock:
                    progress('ITEM_FAIL', f'{exc}', current=idx, total=total)
        return batch_failed

    failed = 0
    if args.concurrency > 1 and len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
            for b_failed in pool.map(_process_batch, batches):
                failed += b_failed
    else:
        for b in batches:
            failed += _process_batch(b)

    progress('PLAYLIST_COMPLETE', f'{total - failed} succeeded; {failed} failed')
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
