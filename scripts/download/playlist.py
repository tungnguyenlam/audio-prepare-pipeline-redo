"""Download all entries in a playlist, sequentially."""
from __future__ import annotations

import contextlib
import sys
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
    failed = 0
    for entry in entries:
        try:
            if not entry:
                raise ValueError('Unavailable playlist entry')
            url = entry.get('webpage_url') or entry.get('url')
            if not url or not url.startswith('http'):
                url = 'https://www.youtube.com/watch?v=' + entry['id']
            with contextlib.redirect_stdout(sys.stderr):
                dest = download(url, args)
            print(dest, flush=True)
        except Exception as exc:
            failed += 1
            print(f'FAILED: {exc}', file=sys.stderr)
    print(f'{len(entries) - failed} succeeded; {failed} failed', file=sys.stderr)
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
