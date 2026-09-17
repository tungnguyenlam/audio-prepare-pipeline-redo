"""Render supplied turns into WAV clips without running a diarization model.

Oversized turns use the shared recursive Silero VAD policy; when no report is
provided, the policy creates or reuses its cached report automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, identity, infer_audio_family, positive_int, read_json, request
from _common.segments import ensure_plots, export, manifest_complete, source_path


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-im', '--input-manifest', type=Path, required=True,
                   help='Path to input segments.json manifest')
    p.add_argument('-i', '-if', '--input-file', type=Path, default=None,
                   help='Source audio override on the same timeline; bypasses the recorded source hash (default: None)')
    p.add_argument('-od', '--output-dir', type=Path, default=None,
                   help='Output directory (default: dynamic per audio family under .data/audio/clips/<family>)')
    p.add_argument('-wd', '--work-dir', type=Path, default=ROOT / '.data/export_segments/work',
                   help='Working directory for temporary files (default: .data/export_segments/work)')
    p.add_argument('-sr', '--sample-rate', type=positive_int, default=None,
                   help='Target audio sample rate in Hz (default: preserve source)')
    p.add_argument('-ch', '--channels', type=int, choices=(1, 2), default=1,
                   help='Target audio channels (1=mono, 2=stereo) (default: 1)')
    p.add_argument('-min', '--min-duration-s', type=float, default=1.0,
                   help='Minimum turn duration in seconds to keep and export (default: 1.0)')
    p.add_argument('-max', '--max-duration-s', type=float, default=15.0,
                   help='Maximum turn duration in seconds to keep and export (default: 15.0)')
    p.add_argument('--long-segment-strategy', '--overlong-strategy',
                   dest='long_segment_strategy', choices=('vad', 'drop'), default='vad',
                   help='How to handle turns longer than --max-duration-s: vad recursively cuts them; drop discards them')
    p.add_argument('--vad-report', type=Path,
                   help='Completed evaluate/silero_jit report for VAD cuts; auto-generated when omitted')
    p.add_argument('--vad-device', default='auto',
                   help='Probability track in the VAD report; auto prefers cuda:0 and falls back to cpu')
    p.add_argument('--vad-cut-threshold', '--vad-threshold',
                   dest='vad_cut_threshold', type=float, default=0.1,
                   help='Only cut at VAD probabilities strictly below this value (default: 0.1)')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true', default=False,
                   help='Rebuild clips and manifest, removing obsolete clips tracked by the previous manifest (default: False)')
    p.add_argument('-c', '--concurrency', type=positive_int, default=1,
                   help='Number of concurrent workers for parallel clip rendering (default: 1)')
    p.add_argument('-b', '-bs', '--batch-size', type=positive_int, default=1,
                   help='Number of clips rendered per batch chunk (default: 1)')
    args = p.parse_args()
    import math
    if args.min_duration_s is not None and (not math.isfinite(args.min_duration_s) or args.min_duration_s < 0):
        p.error('--min-duration-s must be finite and non-negative')
    if args.max_duration_s is not None and (not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0):
        p.error('--max-duration-s must be finite and positive')
    if (not math.isfinite(args.vad_cut_threshold) or
            not 0 <= args.vad_cut_threshold <= 1):
        p.error('--vad-cut-threshold must be finite and between 0 and 1')
    if args.min_duration_s is not None and args.max_duration_s is not None and args.min_duration_s > args.max_duration_s:
        p.error('--min-duration-s cannot exceed --max-duration-s')
    if args.long_segment_strategy == 'drop' and args.vad_report is not None:
        p.error('--vad-report requires --long-segment-strategy vad')
    if args.vad_report is not None and not args.vad_report.is_file():
        p.error(f'VAD report does not exist: {args.vad_report}')
    manifest = read_json(args.input_manifest)
    source = source_path(manifest, args.input_manifest, args.input_file)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (ROOT / '.data/audio/clips' / infer_audio_family(args.input_manifest)).resolve()
    destination = output_dir / 'segments.json'
    if destination == args.input_manifest.resolve() or source.is_relative_to(destination.parent):
        p.error('Output directory must be separate from the input manifest and source')
    wanted = request(identity(source), 'export_segments', {'input_manifest': identity(args.input_manifest),
                     'sample_rate': args.sample_rate, 'channels': args.channels,
                     'min_duration_s': args.min_duration_s, 'max_duration_s': args.max_duration_s,
                     'long_segment_strategy': args.long_segment_strategy,
                     'vad_device': args.vad_device,
                     'vad_cut_threshold': args.vad_cut_threshold,
                     'vad_report': identity(args.vad_report) if args.vad_report else None}, manifest.get('model'))
    if not manifest_complete(destination, wanted, args.overwrite):
        export({**wanted, 'turns': manifest['turns']}, source, destination, args.work_dir, args.sample_rate, args.channels,
               args.min_duration_s, args.max_duration_s, concurrency=args.concurrency, batch_size=args.batch_size,
               long_segment_strategy=args.long_segment_strategy, vad_report=args.vad_report,
               vad_device=args.vad_device, vad_cut_threshold=args.vad_cut_threshold)
        ensure_plots(destination, overwrite=True)
    else:
        # A prior export may predate automatic plots, or an interrupted plot
        # render may have left only part of the sibling set behind.
        ensure_plots(destination, overwrite=False)
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
