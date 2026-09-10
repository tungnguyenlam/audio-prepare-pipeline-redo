"""File handoff for independent turn-editing commands."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, identity, infer_audio_family, probe, progress, read_json, write_json
from _common.segments import normalize_turns, source_path


def arguments(description: str) -> LoggingArgumentParser:
    p = LoggingArgumentParser(description=description)
    p.add_argument('--input-manifest', type=Path, required=True, help='Path to input segments.json manifest')
    p.add_argument('--output-manifest', type=Path, help='Output manifest (default: dynamic per family under .data/purity/<stage>/<family>/segments.json)')
    p.add_argument('--input-file', type=Path, help='Optional source audio file override')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output manifest and invalidate cached sidecars')
    p.add_argument('--concurrency', type=int, default=1, help='Number of concurrent workers for processing turns. Set > 1 to enable concurrent execution')
    p.add_argument('--batch-size', type=int, default=1, help='Number of turns to batch per processing unit')
    return p


def load(args) -> tuple[dict, Path]:
    if getattr(args, 'output_manifest', None) is None:
        stage = Path(sys.argv[0]).stem if sys.argv else 'purity'
        family = infer_audio_family(args.input_manifest)
        args.output_manifest = (ROOT / '.data/purity' / stage / family / 'segments.json').resolve()
    if args.input_manifest.resolve() == args.output_manifest.resolve():
        raise ValueError('Input and output manifests must differ')
    manifest = read_json(args.input_manifest)
    source = source_path(manifest, args.input_manifest, args.input_file)
    if args.output_manifest.resolve() == source:
        raise ValueError('Cannot overwrite source audio')
    manifest['turns'] = normalize_turns(manifest['turns'], source)
    for turn in manifest['turns']:
        turn.setdefault('confidence', None)
    progress('LOAD_MANIFEST', f'Loaded {len(manifest["turns"])} turns from {args.input_manifest.name} (audio: {source.name})')
    return manifest, source


def save(args, manifest: dict, source: Path, turns: list[dict], operation: str, parameters: dict, **details) -> None:
    metadata = {'schema_version': 1, 'source': identity(source), 'timestamp_origin': 'diarized_input',
                'model': manifest.get('model'), 'operation': operation,
                'parameters': {**parameters, 'input_manifest': identity(args.input_manifest)}}
    destination = args.output_manifest.resolve()
    if destination.exists() and not args.overwrite:
        old = read_json(destination)
        if all(old.get(k) == v for k, v in metadata.items()) and old.get('complete'):
            print(destination)
            return
        raise ValueError(f'Conflicting output: {destination}; use --overwrite')
    turns = normalize_turns(turns, source)
    for turn in turns:
        turn.pop('clip_frames', None)
    progress('SAVE_MANIFEST', f'Saving {len(turns)} turns ({operation}) -> {destination.name}')
    write_json(destination, {**metadata, 'turns': turns,
               'speaker_ids': sorted({t['speaker_id'] for t in turns}),
               'source_sample_rate': probe(source)['sample_rate'],
               'clips_valid': False, 'complete': True, **details})
    progress('MANIFEST_DONE', f'Wrote {destination.name}')
    print(destination)
