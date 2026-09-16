"""Enroll or update a target speaker profile from reference WAV clips."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, progress, safe_name


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-n', '--name', required=True, help='Target speaker profile name')
    p.add_argument('--clip', dest='clips', action='append', default=[], type=Path, help='Reference WAV clip file (repeatable)')
    p.add_argument('--clip-dir', type=Path, help='Directory of reference WAV clips')
    p.add_argument('-pd', '--profiles-dir', type=Path, default=ROOT / '.data' / 'speaker_profiles', help='Root profiles directory')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true', help='Replace existing profile completely')
    p.add_argument('--add', action='store_true', help='Append reference clips to existing profile')
    p.add_argument('--channel-id', help='Optional source channel ID for provenance')
    p.add_argument('--channel-name', help='Optional source channel name for provenance')
    p.add_argument('--channel-url', help='Optional source channel URL for provenance')
    p.add_argument('-c', '--concurrency', type=int, default=1, help='Number of concurrent workers for copying clips. Set > 1 to enable concurrent execution')
    p.add_argument('-b', '-bs', '--batch-size', type=int, default=1, help='Number of clips to batch per copying task')
    args = p.parse_args()
    if args.concurrency < 1:
        p.error('--concurrency must be at least 1')
    if args.batch_size < 1:
        p.error('--batch-size must be at least 1')

    clip_paths: list[Path] = list(args.clips)
    if args.clip_dir:
        clip_paths.extend(sorted(args.clip_dir.glob("*.wav")))
    if not clip_paths:
        p.error('At least one clip is required (use --clip or --clip-dir)')

    for c in clip_paths:
        if not c.is_file():
            p.error(f'Clip does not exist: {c}')

    slug = safe_name(args.name)
    if not slug:
        p.error(f'Invalid speaker name: {args.name!r}')

    profile_dir = args.profiles_dir.resolve() / slug
    manifest_path = profile_dir / 'profile.json'
    clips_dir = profile_dir / 'clips'

    if profile_dir.exists() and not (args.overwrite or args.add):
        p.error(f'Profile already exists at {profile_dir}; use --overwrite or --add')

    now = datetime.now(timezone.utc).isoformat()
    existing_clips = []
    created_at = now

    if args.add and manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        existing_clips = list(old.get('clips', []))
        created_at = old.get('created_at', now)
    elif profile_dir.exists() and args.overwrite:
        shutil.rmtree(profile_dir)

    clips_dir.mkdir(parents=True, exist_ok=True)
    new_clip_names = list(existing_clips)

    copy_tasks = []
    idx = len(existing_clips)
    for src_clip in clip_paths:
        suffix = src_clip.suffix or '.wav'
        dest_name = f'clip_{idx:02d}{suffix}'
        while (clips_dir / dest_name).exists():
            idx += 1
            dest_name = f'clip_{idx:02d}{suffix}'
        copy_tasks.append((src_clip, clips_dir / dest_name))
        new_clip_names.append(dest_name)
        idx += 1

    def _copy_batch(batch_items):
        for src, dst in batch_items:
            shutil.copy2(src, dst)

    batches = [copy_tasks[i:i + args.batch_size] for i in range(0, len(copy_tasks), args.batch_size)]
    if args.concurrency > 1 and len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
            list(pool.map(_copy_batch, batches))
    else:
        for b in batches:
            _copy_batch(b)

    manifest = {
        'schema_version': '2.0',
        'name': args.name.strip(),
        'created_at': created_at,
        'updated_at': now,
        'clips': new_clip_names,
        'channel_id': args.channel_id,
        'channel_name': args.channel_name,
        'channel_url': args.channel_url,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    progress('ENROLL_DONE', f'Enrolled {len(new_clip_names)} clips for speaker {args.name!r} at {profile_dir}')
    print(manifest_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
