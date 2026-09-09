"""Download one YouTube video as mono WAV; never expand playlists."""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, completed, convert, positive_int, progress, publish, read_json, request, safe_name


def arguments(description: str, bulk: bool = False) -> LoggingArgumentParser:
    p = LoggingArgumentParser(description=description)
    p.add_argument('--url', required=True)
    p.add_argument('--sample-rate', type=positive_int, default=16000)
    p.add_argument('--output-dir', type=Path, default=ROOT / '.data/download/out')
    if not bulk:
        p.add_argument('--output-file', type=Path)
    p.add_argument('--work-dir', type=Path, default=ROOT / '.data/download/work')
    p.add_argument('--cookie-file', type=Path)
    p.add_argument('--overwrite', action='store_true')
    return p


def download(url: str, args) -> Path:
    from yt_dlp import YoutubeDL
    options = {'noplaylist': True, 'format': 'bestaudio/best', 'quiet': True, 'no_warnings': False}
    if args.cookie_file:
        options['cookiefile'] = str(args.cookie_file.resolve())
    progress('METADATA', f'Fetching info for {url}')
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info or info.get('_type') in {'playlist', 'multi_video'}:
        raise ValueError('Expected a single video URL; use playlist.py or channel.py for bulk downloads')
    video_id = str(info['id'])
    title = info.get('title') or 'video'
    source = {'video_id': video_id, 'title': title, 'url': info.get('webpage_url') or url}
    metadata = request(source, 'download', {'sample_rate': args.sample_rate, 'channels': 1}, 'youtube')
    explicit = getattr(args, 'output_file', None)
    dest = (explicit or args.output_dir / f'{safe_name(title, 80, default="video")}-{args.sample_rate}.wav').resolve()
    if not explicit and (dest.exists() or dest.with_suffix('.json').exists()):
        try:
            existing_id = read_json(dest.with_suffix('.json')).get('source', {}).get('video_id')
        except (ValueError, OSError):
            existing_id = None
        if existing_id and existing_id != video_id:
            dest = dest.with_name(f'{dest.stem}-{safe_name(video_id)}.wav')
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
    args = arguments(__doc__).parse_args()
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


if __name__ == '__main__':
    raise SystemExit(main())
