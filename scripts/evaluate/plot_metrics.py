"""Generate comparison plots from separation or diarization evaluation JSON files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, positive_int, progress, read_json
from concurrent.futures import ThreadPoolExecutor


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--metrics-file', dest='files', action='append', required=True, type=Path, help='Evaluation metrics JSON file (repeatable)')
    p.add_argument('--output-file', type=Path, required=True, help='Output image path (.png, .svg, .pdf)')
    p.add_argument('--title', help='Custom plot title (default: auto-generated from metric type)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output file if present')
    p.add_argument('--concurrency', type=positive_int, default=1, help='Number of worker threads for parallel metrics file loading (default: 1)')
    p.add_argument('--batch-size', type=positive_int, default=1, help='Batch size for chunked metrics file processing (default: 1)')
    args = p.parse_args()

    dest = args.output_file.resolve()
    if dest.exists() and not args.overwrite:
        p.error(f'Destination exists: {dest}; use --overwrite')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if args.concurrency > 1:
        batches = [args.files[i:i + args.batch_size] for i in range(0, len(args.files), args.batch_size)]
        def load_batch(batch_files):
            return [(f.stem, read_json(f)) for f in batch_files]
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as ex:
            records = [rec for batch_res in ex.map(load_batch, batches) for rec in batch_res]
    else:
        records = []
        for f in args.files:
            data = read_json(f)
            records.append((f.stem, data))

    # Detect separation metrics vs diarization metrics
    is_sep = any('si_sdr_db' in r[1].get('metrics', {}) for r in records)
    is_diar = any('der' in r[1].get('metrics', {}) or 'der_pct' in r[1].get('metrics', {}) for r in records)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r[0] for r in records]

    if is_sep:
        values = [r[1].get('metrics', {}).get('si_sdr_db', 0.0) for r in records]
        bars = ax.bar(names, values, color='#3b82f6', width=0.5)
        ax.set_ylabel('SI-SDR (dB)')
        ax.set_title(args.title or 'Separation Performance (SI-SDR dB)')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom' if yval >= 0 else 'top')
    elif is_diar:
        der_vals = [r[1].get('metrics', {}).get('der_pct', r[1].get('metrics', {}).get('der', 0.0) * 100) for r in records]
        bars = ax.bar(names, der_vals, color='#ef4444', width=0.5)
        ax.set_ylabel('Diarization Error Rate (%)')
        ax.set_title(args.title or 'Diarization Performance (DER %)')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')
    else:
        p.error('Unrecognized metric format in input files')

    progress('PLOT_START', f'Plotting comparison metrics for {len(args.files)} files')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(dest, dpi=200)
    plt.close(fig)
    progress('PLOT_DONE', f'Saved plot to {dest.name}')
    print(dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
