"""Plot speaker-turn timelines and duration summaries from a diarization manifest."""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.diarize_plots import write_family_plots, write_plots
from _common.files import LoggingArgumentParser, infer_audio_family, positive_int, progress, safe_name


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-manifest', type=Path, help='Path to one input segments.json manifest')
    inputs.add_argument('--input-dir', type=Path,
                        help='Directory recursively containing segments.json manifests to aggregate')
    p.add_argument('--reference-manifest', type=Path, help='Optional reference segments.json for comparison')
    p.add_argument('--output-file', type=Path,
                   help='Single-manifest Gantt image path; duration and cutoff plots are written beside it')
    p.add_argument('--output-dir', type=Path,
                   help='Output root for family aggregate plots (default: <input-dir>/plot for one family)')
    p.add_argument('--title', help='Custom plot title (default: auto-generated)')
    p.add_argument('--bin-width', type=float, default=0.25, help='Bin width in seconds for segment duration histogram (default: 0.25)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output files if present')
    p.add_argument('--concurrency', type=positive_int, default=1, help='Number of worker threads for parallel manifest loading (default: 1)')
    p.add_argument('--batch-size', type=positive_int, default=1, help='Batch size for chunked turn processing during plotting (default: 1)')
    args = p.parse_args()

    if not math.isfinite(args.bin_width) or args.bin_width <= 0:
        p.error('--bin-width must be a positive number')

    if args.input_manifest is not None:
        if args.output_file is None:
            p.error('--output-file is required with --input-manifest')
        if args.output_dir is not None:
            p.error('--output-dir is only valid with --input-dir')
        dest = args.output_file.resolve()
        progress('PLOT_START', f'Rendering diarization plots: {args.input_manifest.name}')
    else:
        if args.output_file is not None:
            p.error('--output-file is only valid with --input-manifest')
        if args.reference_manifest is not None:
            p.error('--reference-manifest is only valid with --input-manifest')
        input_dir = args.input_dir.resolve()
        if not input_dir.is_dir():
            p.error(f'Input directory not found: {input_dir}')
        manifests = sorted(input_dir.rglob('segments.json'))
        if not manifests:
            p.error(f'No segments.json manifests found below: {input_dir}')
        families: dict[str, list[Path]] = {}
        for manifest in manifests:
            family = safe_name(infer_audio_family(manifest))
            families.setdefault(family, []).append(manifest)

        output_root = args.output_dir.resolve() if args.output_dir is not None else input_dir
        progress('PLOT_START', f'Rendering aggregate diarization plots for {len(manifests)} manifest(s) in {len(families)} famil(ies)')
        paths: list[Path] = []
        try:
            for family, family_manifests in families.items():
                # A single family keeps the historical default <input-dir>/plot.
                # An explicit output root always receives a family subdirectory;
                # multiple families also need separate destinations by default.
                if args.output_dir is None and len(families) == 1:
                    family_root = output_root
                else:
                    family_root = output_root / family
                paths.extend(write_family_plots(
                    family_manifests,
                    family_root / 'plot',
                    root_dir=input_dir,
                    title=args.title,
                    overwrite=args.overwrite,
                    concurrency=args.concurrency,
                    batch_size=args.batch_size,
                    bin_width=args.bin_width,
                ))
        except (FileExistsError, ValueError) as exc:
            p.error(str(exc))
        if not paths:
            p.error('No plot stages could be rendered from the discovered manifests')
        print(paths[0])
        return 0

    try:
        paths = write_plots(
            args.input_manifest,
            dest,
            reference_manifest=args.reference_manifest,
            title=args.title,
            overwrite=args.overwrite,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            bin_width=args.bin_width,
        )
    except FileExistsError as exc:
        p.error(str(exc))
    print(paths[0])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
