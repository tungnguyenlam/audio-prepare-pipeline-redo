"""Render supplied turns into WAV clips without running a diarization model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import ROOT, identity, positive_int, read_json, request
from _common.segments import export, manifest_complete, source_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--input-file', type=Path)
    p.add_argument('--output-dir', type=Path, default=ROOT / '.data/export_segments/out')
    p.add_argument('--work-dir', type=Path, default=ROOT / '.data/export_segments/work')
    p.add_argument('--sample-rate', type=positive_int)
    p.add_argument('--channels', type=int, choices=(1, 2), default=1)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    manifest = read_json(args.input_manifest)
    source = source_path(manifest, args.input_manifest, args.input_file)
    destination = args.output_dir.resolve() / 'segments.json'
    if destination == args.input_manifest.resolve() or source.is_relative_to(destination.parent):
        p.error('Output directory must be separate from the input manifest and source')
    wanted = request(identity(source), 'export_segments', {'input_manifest': identity(args.input_manifest),
                     'sample_rate': args.sample_rate, 'channels': args.channels}, manifest.get('model'))
    if not manifest_complete(destination, wanted, args.overwrite):
        export({**wanted, 'turns': manifest['turns']}, source, destination, args.work_dir, args.sample_rate, args.channels)
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
