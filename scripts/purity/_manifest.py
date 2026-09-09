"""File handoff for independent turn-editing commands."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, identity, probe, progress, read_json, write_json
from _common.segments import normalize_turns, source_path


def arguments(description: str) -> LoggingArgumentParser:
    p = LoggingArgumentParser(description=description)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--output-manifest', type=Path, required=True)
    p.add_argument('--input-file', type=Path)
    p.add_argument('--overwrite', action='store_true')
    return p


def load(args) -> tuple[dict, Path]:
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
