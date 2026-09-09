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
    p.add_argument('--name', required=True, help='Speaker profile name')
    p.add_argument('--clip', dest='clips', action='append', default=[], type=Path, help='Reference clip file (repeatable)')
    p.add_argument('--clip-dir', type=Path, help='Directory of reference WAV clips')
    p.add_argument('--profiles-dir', type=Path, default=ROOT / '.data' / 'speaker_profiles', help='Root profiles directory')
    p.add_argument('--overwrite', action='store_true', help='Replace existing profile')
    p.add_argument('--add', action='store_true', help='Append clips to existing profile')
    p.add_argument('--channel-id', help='Optional source channel ID')
    p.add_argument('--channel-name', help='Optional source channel name')
    p.add_argument('--channel-url', help='Optional source channel URL')
    args = p.parse_args()

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

    idx = len(existing_clips)
    for src_clip in clip_paths:
        suffix = src_clip.suffix or '.wav'
        dest_name = f'clip_{idx:02d}{suffix}'
        while (clips_dir / dest_name).exists():
            idx += 1
            dest_name = f'clip_{idx:02d}{suffix}'
        shutil.copy2(src_clip, clips_dir / dest_name)
        new_clip_names.append(dest_name)
        idx += 1

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
