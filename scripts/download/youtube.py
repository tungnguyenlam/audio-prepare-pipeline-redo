"""Download YouTube video(s) as mono WAV; accepts --url or --url-file; never expand playlists."""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (LoggingArgumentParser, ROOT, completed, convert,
                           family_audio_name, family_name, positive_int,
                           progress, publish, read_json, request, safe_name)


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
    p.add_argument('--sample-rate', type=positive_int, default=48000, help='Target sample rate in Hz for converted WAV')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Output directory (default: dynamic per audio family under .data/download/<family>)')
    p.add_argument('--work-dir', type=Path, default=ROOT / '.data/download/work', help='Working directory for temporary files')
    p.add_argument('--cookie-file', type=Path, help='Optional cookies.txt file for yt-dlp authentication')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output files and sidecars')
    p.add_argument('--concurrency', type=positive_int, default=1, help='Number of concurrent workers for downloads. Set > 1 to enable concurrent execution')
    p.add_argument('--batch-size', type=positive_int, default=1, help='Number of items to batch per worker task')
    p.set_defaults(_operation='download', _default_base=ROOT / '.data/download')
    return p


def download(url: str, args) -> Path:
    from yt_dlp import YoutubeDL
    options = {'noplaylist': True, 'format': 'bestaudio/best', 'quiet': True, 'no_warnings': False}
    if args.cookie_file:
        options['cookiefile'] = str(args.cookie_file.resolve())
    progress('METADATA', f'Fetching info for {url}')
    with YoutubeDL({**options, 'extract_flat': 'in_playlist'}) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info or info.get('_type') in {'playlist', 'multi_video'}:
        raise ValueError('Expected a single video URL; use playlist.py or channel.py for bulk downloads')
    video_id = str(info['id'])
    title = info.get('title') or 'video'
    source = {'video_id': video_id, 'title': title, 'url': info.get('webpage_url') or url}
    metadata = request(source, 'download', {'sample_rate': args.sample_rate, 'channels': 1}, 'youtube')
    explicit = getattr(args, 'output_file', None)
    family = family_name(video_id, title)
    filename = family_audio_name(video_id, title, args.sample_rate)
    if explicit is not None:
        dest = explicit.resolve()
    elif args.output_dir is not None:
        dest = (args.output_dir.resolve() / filename).resolve()
    else:
        dest = (ROOT / '.data/download' / family / filename).resolve()
    if completed(dest, metadata, args.overwrite):
        progress('CACHED', f'Already completed: {dest.name}')
        return dest
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
        options['outtmpl'] = str(Path(work) / 'source.%(ext)s')
        progress('DOWNLOAD', f'Downloading: {title}')
        with YoutubeDL(options) as ydl:
            downloaded = ydl.extract_info(url, download=True)
            src = Path(ydl.prepare_filename(downloaded))
        staged = Path(work) / 'output.wav'
        progress('CONVERT', f'Converting {src.name} -> {args.sample_rate}Hz mono WAV')
        convert(src, staged, args.sample_rate, 1)
        publish(staged, dest, metadata)
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
    import threading
    lock = threading.Lock()

    def _process_item(idx: int, target_url: str):
        if not target_url.startswith('http') and len(target_url) == 11:
            target_url = f'https://www.youtube.com/watch?v={target_url}'
        with lock:
            progress('ITEM_START', f'{target_url}', current=idx, total=total)
        with contextlib.redirect_stdout(sys.stderr):
            dest = download(target_url, args)
        with lock:
            progress('ITEM_DONE', f'{dest.name}', current=idx, total=total)
            print(dest, flush=True)

    def _process_batch(batch_items):
        batch_failed = 0
        for idx, target_url in batch_items:
            try:
                _process_item(idx, target_url)
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

    progress('STATUS', f'{total - failed} succeeded; {failed} failed')
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
