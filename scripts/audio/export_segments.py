"""Render supplied turns into WAV clips without running a diarization model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, identity, infer_audio_family, positive_int, read_json, request
from _common.segments import export, manifest_complete, source_path


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True,
                   help='Path to input segments.json manifest')
    p.add_argument('--input-file', type=Path, default=None,
                   help='Optional source audio file override (default: None)')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Output directory (default: dynamic per audio family under .data/audio/clips/<family>)')
    p.add_argument('--work-dir', type=Path, default=ROOT / '.data/export_segments/work',
                   help='Working directory for temporary files (default: .data/export_segments/work)')
    p.add_argument('--sample-rate', type=positive_int, default=None,
                   help='Target audio sample rate in Hz (default: preserve source)')
    p.add_argument('--channels', type=int, choices=(1, 2), default=1,
                   help='Target audio channels (1=mono, 2=stereo) (default: 1)')
    p.add_argument('--min-duration-s', type=float, default=2.0,
                   help='Minimum turn duration in seconds to keep and export (default: 2.0)')
    p.add_argument('--max-duration-s', type=float, default=15.0,
                   help='Maximum turn duration in seconds to keep and export (default: 15.0)')
    p.add_argument('--overwrite', action='store_true', default=False,
                   help='Overwrite existing output clips and manifest (default: False)')
    p.add_argument('--concurrency', type=positive_int, default=1,
                   help='Number of concurrent workers for parallel clip rendering (default: 1)')
    p.add_argument('--batch-size', type=positive_int, default=1,
                   help='Number of clips rendered per batch chunk (default: 1)')
    args = p.parse_args()
    import math
    if args.min_duration_s is not None and (not math.isfinite(args.min_duration_s) or args.min_duration_s < 0):
        p.error('--min-duration-s must be finite and non-negative')
    if args.max_duration_s is not None and (not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0):
        p.error('--max-duration-s must be finite and positive')
    if args.min_duration_s is not None and args.max_duration_s is not None and args.min_duration_s > args.max_duration_s:
        p.error('--min-duration-s cannot exceed --max-duration-s')
    manifest = read_json(args.input_manifest)
    source = source_path(manifest, args.input_manifest, args.input_file)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (ROOT / '.data/audio/clips' / infer_audio_family(args.input_manifest)).resolve()
    destination = output_dir / 'segments.json'
    if destination == args.input_manifest.resolve() or source.is_relative_to(destination.parent):
        p.error('Output directory must be separate from the input manifest and source')
    wanted = request(identity(source), 'export_segments', {'input_manifest': identity(args.input_manifest),
                     'sample_rate': args.sample_rate, 'channels': args.channels,
                     'min_duration_s': args.min_duration_s, 'max_duration_s': args.max_duration_s}, manifest.get('model'))
    if not manifest_complete(destination, wanted, args.overwrite):
        export({**wanted, 'turns': manifest['turns']}, source, destination, args.work_dir, args.sample_rate, args.channels,
               args.min_duration_s, args.max_duration_s, concurrency=args.concurrency, batch_size=args.batch_size)
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
