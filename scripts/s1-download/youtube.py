"""Download YouTube video(s) as mono WAV; accepts --url or --url-file; never expand playlists."""
from __future__ import annotations

import argparse
import contextlib
import copy
import re
import threading
import time
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (LoggingArgumentParser, ROOT, completed, convert,
                           family_audio_name, family_name, positive_int,
                           progress, publish, request, safe_name)


_RATE_LIMIT_MARKERS = (
    'http error 429',
    'too many requests',
    'sign in to confirm',
    'not a bot',
    "confirm you're not a bot",
    'confirm you’re not a bot',
    'confirm you are not a bot',
    'rate-limit exceeded',
    'rate limit exceeded',
)
_YOUTUBE_LOCK = threading.Lock()
_THROTTLE_LOCK = threading.Lock()
_NEXT_YOUTUBE_TIME = 0.0
_CONSECUTIVE_THROTTLED_ITEMS = 0
_RATE_LIMIT_ABORT = threading.Event()


class RateLimitAbort(RuntimeError):
    """YouTube is rate-limiting this process; remaining items should stop."""


def non_negative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be non-negative')
    return result


def non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be non-negative')
    return result


def parse_rate_limit(value: str) -> int | None:
    text = value.strip().upper().replace(' ', '')
    if text in {'0', 'NONE', 'OFF'}:
        return None
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT])I?B?', text)
    if not match:
        raise argparse.ArgumentTypeError('expected a byte rate such as 2M or 500K, or 0 to disable')
    number = float(match.group(1))
    unit = match.group(2)
    multipliers = {'K': 1000, 'M': 1_000_000, 'G': 1_000_000_000, 'T': 1_000_000_000_000}
    result = int(number * multipliers.get(unit, 1))
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def add_download_arguments(p: LoggingArgumentParser) -> LoggingArgumentParser:
    """Add options shared by single-video, playlist, channel, and crawl commands."""
    p.add_argument('--sample-rate', type=positive_int, default=48000, help='Target sample rate in Hz for converted WAV')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Output root; bulk playlist/channel downloads add a resolved collection-name subdirectory')
    p.add_argument('--work-dir', type=Path, default=ROOT / '.data/s1-download/work', help='Working directory for temporary files')
    p.add_argument('--cookie-file', type=Path,
                   help='Optional cookies.txt for yt-dlp. Prefer a throwaway account; a personal login can be banned')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output files and sidecars')
    p.add_argument('--concurrency', type=positive_int, default=1,
                   help='Parallel convert/publish workers. YouTube HTTP stays serialized. Keep 1 for large crawls')
    p.add_argument('--batch-size', type=positive_int, default=1, help='Number of items to batch per worker task')
    p.add_argument('--sleep-requests', type=non_negative_float, default=1.5,
                   help='Seconds to sleep between yt-dlp metadata requests, including playlist/channel paging')
    p.add_argument('--sleep-interval', type=non_negative_float, default=5.0,
                   help='Minimum seconds to sleep before each media download; 0 disables download pacing')
    p.add_argument('--max-sleep-interval', type=non_negative_float, default=15.0,
                   help='Maximum seconds to sleep before each media download; randomized with --sleep-interval')
    p.add_argument('--rate-limit', type=parse_rate_limit, default=2_000_000, metavar='RATE',
                   help='Maximum media download rate (e.g. 500K, 2M). 0 disables. default: 2M')
    p.add_argument('--throttle-retries', type=non_negative_int, default=3,
                   help='Extra attempts per item after a YouTube rate-limit or bot-check error')
    p.add_argument('--throttle-abort-after', type=non_negative_int, default=3,
                   help='Stop the run after this many consecutive items that stay rate-limited; 0 never aborts')
    p.add_argument('--throttle-backoff-s', type=non_negative_float, default=300.0,
                   help='Initial cooldown in seconds after a rate-limit error; doubles each retry up to 8x')
    p.set_defaults(_operation='download', _default_base=ROOT / '.data/s1-download')
    return p


def arguments(description: str, bulk: bool = False) -> LoggingArgumentParser:
    p = LoggingArgumentParser(description=description)
    if not bulk:
        p.add_argument('--url', default=None, help='YouTube video URL or video ID')
        p.add_argument('--url-file', type=Path, default=None,
                       help='Path to text file containing YouTube URLs or video IDs (one per line)')
        p.add_argument('--output-file', type=Path,
                       help='Explicit destination WAV file path (requires single video download via --url)')
    else:
        p.add_argument('--url', required=True, help='YouTube video, playlist, or channel URL')
        p.add_argument('--limit', '--max-items', dest='limit', type=positive_int, default=None,
                       help='Maximum number of videos to download from the playlist or channel')
    return add_download_arguments(p)


def _retry_sleep(attempt: int) -> float:
    return min(30.0, 1.0 * (2 ** attempt))


def ydl_options(args, extra: dict | None = None) -> dict:
    """Build yt-dlp options with pacing, retries, and optional cookies."""
    if args.max_sleep_interval and args.sleep_interval and args.max_sleep_interval < args.sleep_interval:
        raise ValueError('--max-sleep-interval must be greater than or equal to --sleep-interval')
    options = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': False,
        'retries': 3,
        'fragment_retries': 3,
        'extractor_retries': 3,
        'retry_sleep_functions': {
            'http': _retry_sleep,
            'fragment': _retry_sleep,
            'extractor': _retry_sleep,
        },
    }
    if args.cookie_file:
        options['cookiefile'] = str(args.cookie_file.resolve())
    if args.sleep_requests:
        options['sleep_interval_requests'] = args.sleep_requests
    if args.sleep_interval:
        options['sleep_interval'] = args.sleep_interval
        options['max_sleep_interval'] = max(args.sleep_interval, args.max_sleep_interval or args.sleep_interval)
    if args.rate_limit:
        options['ratelimit'] = args.rate_limit
    if extra:
        options.update(extra)
    return options


def is_rate_limited(exc: BaseException) -> bool:
    text = str(exc).lower()
    cause = exc.__cause__
    while cause is not None:
        text += ' ' + str(cause).lower()
        cause = cause.__cause__
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def raise_if_rate_limited() -> None:
    if _RATE_LIMIT_ABORT.is_set():
        raise RateLimitAbort('YouTube rate limiting aborted this run; remaining items were skipped')


def _sleep_or_abort(delay: float) -> None:
    deadline = time.monotonic() + delay
    while True:
        raise_if_rate_limited()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def youtube_session(args):
    """Serialize yt-dlp access and keep a gap between YoutubeDL instances."""
    gap = float(getattr(args, 'sleep_requests', 0) or 0)

    @contextlib.contextmanager
    def _session():
        global _NEXT_YOUTUBE_TIME
        with _YOUTUBE_LOCK:
            raise_if_rate_limited()
            wait = _NEXT_YOUTUBE_TIME - time.monotonic()
            if wait >= 1:
                progress('PACE', f'Sleeping {wait:.1f}s before the next YouTube request')
            if wait > 0:
                _sleep_or_abort(wait)
            try:
                yield
            finally:
                _NEXT_YOUTUBE_TIME = time.monotonic() + gap
    return _session()


def _record_success() -> None:
    global _CONSECUTIVE_THROTTLED_ITEMS
    with _THROTTLE_LOCK:
        _CONSECUTIVE_THROTTLED_ITEMS = 0


def _give_up_throttled_item(args, exc: BaseException, label: str) -> None:
    global _CONSECUTIVE_THROTTLED_ITEMS
    with _THROTTLE_LOCK:
        _CONSECUTIVE_THROTTLED_ITEMS += 1
        consecutive = _CONSECUTIVE_THROTTLED_ITEMS
        abort_after = args.throttle_abort_after
        if abort_after and consecutive >= abort_after:
            _RATE_LIMIT_ABORT.set()
            raise RateLimitAbort(
                f'{label}: still rate-limited after retries ({exc}); '
                f'{consecutive} consecutive throttled items, aborting run') from exc
    raise exc


def with_throttle_retry(args, fn, label: str):
    """Call fn, cooling down and retrying on YouTube rate-limit errors."""
    raise_if_rate_limited()
    attempts = args.throttle_retries + 1
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        raise_if_rate_limited()
        try:
            result = fn()
            _record_success()
            return result
        except RateLimitAbort:
            raise
        except Exception as exc:
            if not is_rate_limited(exc):
                raise
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            delay = min(args.throttle_backoff_s * 8, args.throttle_backoff_s * (2 ** attempt))
            progress('RATE_LIMITED',
                     f'{label}: {exc}; cooling down {delay:.0f}s '
                     f'(attempt {attempt + 1}/{attempts})')
            with _YOUTUBE_LOCK:
                _sleep_or_abort(delay)
    assert last_exc is not None
    _give_up_throttled_item(args, last_exc, label)
    raise last_exc


def _fetch_metadata(url: str, args) -> dict:
    from yt_dlp import YoutubeDL
    progress('METADATA', f'Fetching info for {url}')
    options = ydl_options(args, extra={'noplaylist': True, 'extract_flat': 'in_playlist'})
    with youtube_session(args), YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info or info.get('_type') in {'playlist', 'multi_video'}:
        raise ValueError('Expected a single video URL; use playlist.py, channel.py, or crawl.py for bulk downloads')
    return info


def _extract_media(url: str, args, info: dict, work: Path):
    from yt_dlp import YoutubeDL
    options = ydl_options(args, extra={'noplaylist': True, 'outtmpl': str(work / 'source.%(ext)s')})
    with YoutubeDL(options) as ydl:
        if info.get('formats'):
            try:
                downloaded = ydl.process_ie_result(copy.deepcopy(info), download=True)
            except Exception as exc:
                if is_rate_limited(exc):
                    raise
                downloaded = ydl.extract_info(url, download=True)
        else:
            downloaded = ydl.extract_info(url, download=True)
        if not downloaded:
            raise ValueError(f'YouTube download returned no result: {url}')
        src = Path(ydl.prepare_filename(downloaded))
    if not src.is_file():
        matches = sorted(path for path in work.glob('source.*') if path.is_file())
        if not matches:
            raise FileNotFoundError(f'yt-dlp did not write a media file for {url}')
        src = matches[0]
    return src


def download(url: str, args, source_info: dict | None = None,
             *, output_group: str | None = None) -> Path:
    raise_if_rate_limited()
    info = source_info
    if info is None:
        info = with_throttle_retry(args, lambda: _fetch_metadata(url, args), label=url)
    if not info or not info.get('id'):
        raise ValueError(f'YouTube metadata did not contain a video ID: {url}')
    video_id = str(info['id'])
    title = info.get('title') or 'video'
    source = {'video_id': video_id, 'title': title, 'url': info.get('webpage_url') or url}
    metadata = request(source, 'download', {'sample_rate': args.sample_rate, 'channels': 1}, 'youtube')
    explicit = getattr(args, 'output_file', None)
    family = family_name(video_id, title)
    filename = family_audio_name(video_id, title, args.sample_rate)
    if explicit is not None:
        dest = explicit.resolve()
    elif output_group is not None:
        base = args.output_dir.resolve() if args.output_dir is not None else ROOT / '.data/s1-download'
        dest = (base / safe_name(output_group, default='youtube') / filename).resolve()
    elif args.output_dir is not None:
        dest = (args.output_dir.resolve() / filename).resolve()
    else:
        dest = (ROOT / '.data/s1-download' / family / filename).resolve()
    if completed(dest, metadata, args.overwrite):
        progress('CACHED', f'Already completed: {dest.name}')
        return dest

    def _download_and_publish():
        args.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            progress('DOWNLOAD', f'Downloading: {title}')
            with youtube_session(args):
                src = _extract_media(url, args, info, Path(work))
            staged = Path(work) / 'output.wav'
            progress('CONVERT', f'Converting {src.name} -> {args.sample_rate}Hz mono WAV')
            convert(src, staged, args.sample_rate, 1)
            publish(staged, dest, metadata)
        return dest

    dest = with_throttle_retry(args, _download_and_publish, label=title)
    progress('COMPLETE', f'Saved to {dest.name}')
    return dest


def main() -> int:
    parser = arguments(__doc__)
    args = parser.parse_args()

    if not args.url and not args.url_file:
        parser.error('Supply --url or --url-file')
    if args.url and args.url_file:
        parser.error('Cannot specify both --url and --url-file')
    if args.url_file:
        if getattr(args, 'output_file', None) is not None:
            parser.error('--output-file cannot be used with --url-file; use --output-dir')
        if not args.url_file.is_file():
            parser.error(f'URL file not found: {args.url_file}')

    if args.url:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                dest = download(args.url, args)
            print(dest)
            progress('STATUS', '1 succeeded; 0 failed')
            return 0
        except RateLimitAbort as exc:
            progress('RATE_LIMITED', f'{exc}')
            progress('STATUS', '0 succeeded; 1 failed')
            return 1
        except Exception as exc:
            progress('ERROR', f'{exc}')
            progress('STATUS', '0 succeeded; 1 failed')
            return 1

    with args.url_file.open(encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    total = len(urls)
    progress('URL_FILE', f'Found {total} URL(s) to process from {args.url_file.name}')
    if total == 0:
        progress('STATUS', '0 succeeded; 0 failed')
        return 0

    indexed_urls = list(enumerate(urls, 1))
    batches = [indexed_urls[i:i + args.batch_size] for i in range(0, total, args.batch_size)]
    lock = threading.Lock()
    succeeded = 0
    failed = 0

    def _process_item(idx: int, target_url: str):
        nonlocal succeeded
        raise_if_rate_limited()
        if not target_url.startswith('http') and len(target_url) == 11:
            target_url = f'https://www.youtube.com/watch?v={target_url}'
        with lock:
            progress('ITEM_START', f'{target_url}', current=idx, total=total)
        with contextlib.redirect_stdout(sys.stderr):
            dest = download(target_url, args)
        with lock:
            succeeded += 1
            progress('ITEM_DONE', f'{dest.name}', current=idx, total=total)
            print(dest, flush=True)

    def _process_batch(batch_items):
        nonlocal failed
        batch_failed = 0
        for idx, target_url in batch_items:
            try:
                _process_item(idx, target_url)
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
        progress('STATUS', f'{succeeded} succeeded; {failed} failed; {skipped} skipped after rate limit')
    else:
        progress('STATUS', f'{succeeded} succeeded; {failed} failed')
    return int(failed > 0 or aborted)


if __name__ == '__main__':
    raise SystemExit(main())
