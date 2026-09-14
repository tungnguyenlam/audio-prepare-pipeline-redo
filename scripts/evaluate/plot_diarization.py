"""Plot speaker-turn timelines and duration summaries from a diarization manifest."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.diarize_plots import write_plots
from _common.files import LoggingArgumentParser, positive_int, progress


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True, help='Path to input segments.json manifest')
    p.add_argument('--reference-manifest', type=Path, help='Optional reference segments.json for comparison')
    p.add_argument('--output-file', type=Path, required=True,
                   help='Gantt image path (.png, .svg, .pdf); duration and cutoff plots are written beside it')
    p.add_argument('--title', help='Custom plot title (default: auto-generated)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output files if present')
    p.add_argument('--concurrency', type=positive_int, default=1, help='Number of worker threads for parallel manifest loading (default: 1)')
    p.add_argument('--batch-size', type=positive_int, default=1, help='Batch size for chunked turn processing during plotting (default: 1)')
    args = p.parse_args()

    dest = args.output_file.resolve()
    progress('PLOT_START', f'Rendering diarization plots: {args.input_manifest.name}')
    try:
        paths = write_plots(
            args.input_manifest,
            dest,
            reference_manifest=args.reference_manifest,
            title=args.title,
            overwrite=args.overwrite,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
        )
    except FileExistsError as exc:
        p.error(str(exc))
    print(paths[0])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
