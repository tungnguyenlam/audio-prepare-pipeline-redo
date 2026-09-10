"""Download all entries in a playlist, sequentially."""
from __future__ import annotations

import contextlib
import sys
import urllib.parse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import progress
from youtube import arguments, download


def normalize_playlist_url(url: str) -> str:
    """If a video URL contains a playlist parameter (list=...), infer the playlist URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if 'list' in query and query['list']:
            return f'https://www.youtube.com/playlist?list={query["list"][0]}'
    except Exception:
        pass
    return url


def main(description: str = __doc__) -> int:
    args = arguments(description, bulk=True).parse_args()
    from yt_dlp import YoutubeDL
    options = {'extract_flat': 'in_playlist', 'quiet': True, 'ignoreerrors': True}
    if args.cookie_file:
        options['cookiefile'] = str(args.cookie_file.resolve())
    if args.limit is not None:
        options['playlistend'] = args.limit
    target_url = normalize_playlist_url(args.url)
    if target_url != args.url:
        progress('INFER', f'Inferred playlist URL: {target_url}')
    with contextlib.redirect_stdout(sys.stderr), YoutubeDL(options) as ydl:
        listing = ydl.extract_info(target_url, download=False)
        if not listing or 'entries' not in listing:
            raise ValueError('URL did not resolve to a playlist or channel')
        raw_entries = [e for e in listing['entries'] if e is not None]
    entries = raw_entries[:args.limit] if args.limit is not None else raw_entries
    total = len(entries)
    if args.limit is not None and len(raw_entries) != total:
        progress('PLAYLIST', f'Found {len(raw_entries)} items; limited to first {total}')
    else:
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
