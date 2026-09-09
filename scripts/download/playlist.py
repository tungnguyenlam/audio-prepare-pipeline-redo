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
    failed = 0
    for idx, entry in enumerate(entries, 1):
        try:
            if not entry:
                raise ValueError('Unavailable playlist entry')
            title = entry.get('title') or entry.get('id') or 'video'
            url = entry.get('webpage_url') or entry.get('url')
            if not url or not url.startswith('http'):
                url = 'https://www.youtube.com/watch?v=' + entry['id']
            progress('ITEM_START', f'{title}', current=idx, total=total)
            with contextlib.redirect_stdout(sys.stderr):
                dest = download(url, args)
            progress('ITEM_DONE', f'{dest.name}', current=idx, total=total)
            print(dest, flush=True)
        except Exception as exc:
            failed += 1
            progress('ITEM_FAIL', f'{exc}', current=idx, total=total)
    progress('PLAYLIST_COMPLETE', f'{total - failed} succeeded; {failed} failed')
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
