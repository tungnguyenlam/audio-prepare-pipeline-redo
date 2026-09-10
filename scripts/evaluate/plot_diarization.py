"""Plot a timeline / Gantt chart of speaker turns from a diarization manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, positive_int, progress, read_json
from concurrent.futures import ThreadPoolExecutor


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True, help='Path to input segments.json manifest')
    p.add_argument('--reference-manifest', type=Path, help='Optional reference segments.json for comparison')
    p.add_argument('--output-file', type=Path, required=True, help='Output image path (.png, .svg, .pdf)')
    p.add_argument('--title', help='Custom plot title (default: auto-generated)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output file if present')
    p.add_argument('--concurrency', type=positive_int, default=1, help='Number of worker threads for parallel manifest loading (default: 1)')
    p.add_argument('--batch-size', type=positive_int, default=1, help='Batch size for chunked turn processing during plotting (default: 1)')
    args = p.parse_args()

    dest = args.output_file.resolve()
    if dest.exists() and not args.overwrite:
        p.error(f'Destination exists: {dest}; use --overwrite')

    progress('PLOT_START', f'Rendering Gantt chart: {args.input_manifest.name}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if args.reference_manifest:
        if args.concurrency > 1:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_hyp = ex.submit(read_json, args.input_manifest)
                fut_ref = ex.submit(read_json, args.reference_manifest)
                data = fut_hyp.result()
                ref_data = fut_ref.result()
        else:
            data = read_json(args.input_manifest)
            ref_data = read_json(args.reference_manifest)
        turns = data.get('turns', [])
        ref_turns = ref_data.get('turns', [])
        fig, (ax_ref, ax_hyp) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        _plot_turns(ax_ref, ref_turns, 'Reference Diarization', batch_size=args.batch_size)
        _plot_turns(ax_hyp, turns, 'Hypothesis Diarization', batch_size=args.batch_size)
        if args.title:
            fig.suptitle(args.title, fontsize=14)
    else:
        data = read_json(args.input_manifest)
        turns = data.get('turns', [])
        fig, ax = plt.subplots(figsize=(14, max(3.0, 1.2 + 0.5 * len({t['speaker_id'] for t in turns}))))
        _plot_turns(ax, turns, args.title or f"Diarization: {data.get('source', {}).get('path', args.input_manifest.stem)}", batch_size=args.batch_size)

    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(dest, dpi=200)
    plt.close(fig)
    progress('PLOT_DONE', f'Saved plot to {dest.name}')
    print(dest)
    return 0


def _plot_turns(ax, turns: list[dict], title: str, batch_size: int = 1):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    speakers = sorted({t['speaker_id'] for t in turns})
    if not speakers:
        speakers = ['SPEAKER_00']
    spk_to_y = {spk: idx for idx, spk in enumerate(reversed(speakers))}
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(len(speakers))]

    batch_size = max(1, batch_size)
    for i in range(0, len(turns), batch_size):
        chunk = turns[i:i + batch_size]
        for t in chunk:
            dur = t['end_s'] - t['start_s']
            spk = t['speaker_id']
            y = spk_to_y[spk]
            color = colors[speakers.index(spk) % len(colors)]
            overlap = t.get('overlap', False) or t.get('overlaps_other_speaker', False)
            edgecolor = '#ef4444' if overlap else 'none'
            linewidth = 1.5 if overlap else 0
            ax.barh(y=y, width=dur, left=t['start_s'], height=0.6, color=color,
                    edgecolor=edgecolor, linewidth=linewidth, alpha=0.85)

    has_overlap = any(t.get('overlap', False) or t.get('overlaps_other_speaker', False) for t in turns)
    if has_overlap:
        patch = mpatches.Patch(facecolor='#fca5a5', edgecolor='#ef4444', linewidth=1.5, label='Overlap')
        ax.legend(handles=[patch], loc='upper right', framealpha=0.85)

    ax.set_yticks([spk_to_y[s] for s in speakers])
    ax.set_yticklabels(speakers)
    ax.set_xlabel('Time (seconds)')
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)


if __name__ == '__main__':
    raise SystemExit(main())
