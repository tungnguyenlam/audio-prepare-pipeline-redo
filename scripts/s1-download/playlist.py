"""Download all entries in a playlist, sequentially; accepts --url or --url-file."""
from __future__ import annotations

import contextlib
import sys
import threading
import urllib.parse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import progress
from youtube import (RateLimitAbort, arguments, download, raise_if_rate_limited,
                     resolve_url_or_url_file, with_throttle_retry, ydl_options,
                     youtube_session)


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


def _is_channel_target(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split('/') if part]
    return bool(parts) and (
        parts[0].startswith('@') or parts[0].startswith('UC')
        or parts[0] in {'browse', 'c', 'channel', 'user'})


def _first_text(info: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _fallback_name(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if query.get('list'):
        return f'playlist-{query["list"][0]}'
    parts = [urllib.parse.unquote(part) for part in parsed.path.split('/') if part]
    if parts:
        if parts[0].startswith('@'):
            return parts[0][1:]
        if parts[0] in {'browse', 'c', 'channel', 'user'} and len(parts) > 1:
            return parts[1].lstrip('@')
    return 'youtube'


def resolved_listing_name(listing: dict, url: str) -> str:
    """Return the remote playlist/channel name used as the output group."""
    if _is_channel_target(url):
        name = _first_text(listing, ('channel', 'uploader', 'title'))
    else:
        name = _first_text(listing, ('title', 'playlist_title', 'channel', 'uploader'))
    return name or _fallback_name(url)


def list_entries(url: str, args, *, limit: int | None = None) -> tuple[str, int, list[dict], str]:
    """Resolve a playlist-like yt-dlp target without downloading its media."""
    from yt_dlp import YoutubeDL
    target_url = normalize_playlist_url(url)

    def _list():
        options = ydl_options(args, extra={
            'extract_flat': 'in_playlist',
            'ignoreerrors': True,
            'noplaylist': False,
        })
        if limit is not None:
            options['playlistend'] = limit
        with youtube_session(args), contextlib.redirect_stdout(sys.stderr), YoutubeDL(options) as ydl:
            listing = ydl.extract_info(target_url, download=False)
        if not listing or 'entries' not in listing:
            raise ValueError('URL did not resolve to a playlist, channel tab, or search')
        return listing

    listing = with_throttle_retry(args, _list, label=target_url)
    raw_entries = list(listing['entries'])
    entries = [entry for entry in raw_entries if entry is not None]
    if limit is not None:
        entries = entries[:limit]
    return target_url, len(raw_entries), entries, resolved_listing_name(listing, target_url)


def entry_url(entry: dict) -> str:
    url = entry.get('webpage_url') or entry.get('url')
    if url and str(url).startswith('http'):
        return str(url)
    if not entry.get('id'):
        raise ValueError('Playlist entry has neither a URL nor video ID')
    return 'https://www.youtube.com/watch?v=' + str(entry['id'])


def download_playlist(url: str, args) -> tuple[int, int, bool]:
    """Download one playlist or channel URL. Returns (succeeded, failed, aborted)."""
    try:
        target_url, raw_count, entries, output_group = list_entries(
            url, args, limit=args.limit)
    except RateLimitAbort as exc:
        progress('RATE_LIMITED', f'{exc}')
        return 0, 0, True
    except Exception as exc:
        progress('ERROR', f'{exc}')
        return 0, 1, False
    if target_url != url:
        progress('INFER', f'Inferred playlist URL: {target_url}')
    total = len(entries)
    if args.limit is not None and raw_count != total:
        progress('PLAYLIST', f'Found {raw_count} items; limited to first {total}; group: {output_group}')
    else:
        progress('PLAYLIST', f'Found {total} items to process; group: {output_group}')
    indexed_entries = list(enumerate(entries, 1))
    batches = [indexed_entries[i:i + args.batch_size] for i in range(0, len(indexed_entries), args.batch_size)]
    lock = threading.Lock()
    succeeded = 0
    failed = 0

    def _process_item(idx, entry):
        nonlocal succeeded
        raise_if_rate_limited()
        if not entry:
            raise ValueError('Unavailable playlist entry')
        title = entry.get('title') or entry.get('id') or 'video'
        item_url = entry_url(entry)
        with lock:
            progress('ITEM_START', f'{title}', current=idx, total=total)
        with contextlib.redirect_stdout(sys.stderr):
            dest = download(item_url, args, source_info=entry, output_group=output_group)
        with lock:
            succeeded += 1
            progress('ITEM_DONE', f'{dest.name}', current=idx, total=total)
            print(dest, flush=True)

    def _process_batch(batch_items):
        nonlocal failed
        batch_failed = 0
        for idx, entry in batch_items:
            try:
                _process_item(idx, entry)
            except RateLimitAbort as exc:
                batch_failed += 1
                with lock:
                    failed += 1
                    progress('RATE_LIMITED', f'{exc}', current=idx, total=total)
                raise
            except Exception as exc:
                batch_failed += 1
                with lock:
                    failed += 1
                    progress('ITEM_FAIL', f'{exc}', current=idx, total=total)
        return batch_failed

    aborted = False
    try:
        if args.concurrency > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
                for _b_failed in pool.map(_process_batch, batches):
                    pass
        else:
            for b in batches:
                _process_batch(b)
    except RateLimitAbort as exc:
        aborted = True
        progress('RATE_LIMITED', f'{exc}')

    skipped = total - succeeded - failed
    if aborted:
        progress('PLAYLIST_COMPLETE',
                 f'{succeeded} succeeded; {failed} failed; {skipped} skipped after rate limit')
    else:
        progress('PLAYLIST_COMPLETE', f'{succeeded} succeeded; {failed} failed')
    return succeeded, failed, aborted


def main(description: str = __doc__) -> int:
    parser = arguments(description, bulk=True)
    args = parser.parse_args()
    urls = resolve_url_or_url_file(parser, args)
    if args.url_file:
        progress('URL_FILE', f'Found {len(urls)} URL(s) to process from {args.url_file.name}')
        if not urls:
            progress('STATUS', '0 succeeded; 0 failed')
            return 0

    succeeded = 0
    failed = 0
    aborted = False
    for index, url in enumerate(urls, 1):
        if len(urls) > 1:
            progress('PLAYLIST_START', url, current=index, total=len(urls))
        item_succeeded, item_failed, aborted = download_playlist(url, args)
        succeeded += item_succeeded
        failed += item_failed
        if aborted:
            remaining = len(urls) - index
            if remaining:
                progress('STATUS', f'{remaining} URL(s) skipped after rate limit')
            break
    if len(urls) > 1:
        progress('STATUS', f'{succeeded} videos succeeded; {failed} failed')
    return int(failed > 0 or aborted)


if __name__ == '__main__':
    raise SystemExit(main())
