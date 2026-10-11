"""Select the frozen ViCut-Bench anchor pool with a speaker-disjoint split.

Reads measure_anchors output and keeps unflagged 3-12 s clips with a transcript
(optionally only from --dataset).
Fills --stop-count final stop-coda (-p/-t/-c/-ch) clips and the rest open/nasal
controls, alternating datasets so each class is balanced where supply allows and
capping clips per speaker. Whole speakers are then assigned to the test split until
each dataset reaches --test-fraction of its clips.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, progress, write_json


def take(pools: dict[str, list[dict]], count: int, per_speaker: dict, cap: int) -> list[dict]:
    """Round-robin over datasets, skipping speakers already at the cap."""
    chosen, queues = [], {name: list(rows) for name, rows in pools.items()}
    while len(chosen) < count and any(queues.values()):
        for name in sorted(queues):
            while queues[name] and len(chosen) < count:
                row = queues[name].pop()
                if per_speaker.get(row['speaker'], 0) < cap:
                    per_speaker[row['speaker']] = per_speaker.get(row['speaker'], 0) + 1
                    chosen.append(row)
                    break
    return chosen


def main() -> int:
    anchors = Path('.data/vicut_bench/anchors')
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-im', '--input-manifest', type=Path, default=anchors / 'measured_candidates.jsonl')
    p.add_argument('-om', '--output-manifest', type=Path, default=anchors / 'vicut_bench_anchors_300.jsonl')
    p.add_argument('--dataset', action='append', help='Restrict to this dataset; repeatable (default: all)')
    p.add_argument('--target-count', type=int, default=300)
    p.add_argument('--stop-count', type=int, default=100)
    p.add_argument('--max-per-speaker', type=int, default=10)
    p.add_argument('--test-fraction', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-w', '--overwrite', action='store_true')
    args = p.parse_args()
    if args.output_manifest.exists() and not args.overwrite:
        p.error('Output exists; choose another --output-manifest or use --overwrite')

    rows = [json.loads(line) for line in args.input_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    eligible = [r for r in rows if r.get('in_3_12s_window') and r.get('text') and r.get('screen_flags') == []
                and (not args.dataset or r['dataset'] in args.dataset)]
    rng = random.Random(args.seed)
    pools = {}
    for row in sorted(eligible, key=lambda r: r['id']):
        pools.setdefault(row['is_stop_coda'], {}).setdefault(row['dataset'], []).append(row)
    for by_dataset in pools.values():
        for pool in by_dataset.values():
            rng.shuffle(pool)
    per_speaker = {}
    selected = take(pools.get(True, {}), args.stop_count, per_speaker, args.max_per_speaker)
    selected += take(pools.get(False, {}), args.target_count - len(selected), per_speaker, args.max_per_speaker)
    if len(selected) < args.target_count:
        progress('SHORTFALL', f'{len(selected)}/{args.target_count} anchors available after screening and speaker cap')

    test_speakers = set()
    for dataset in sorted({r['dataset'] for r in selected}):
        subset = [r for r in selected if r['dataset'] == dataset]
        speakers = sorted({r['speaker'] for r in subset})
        rng.shuffle(speakers)
        for speaker in speakers:
            if sum(r['speaker'] in test_speakers for r in subset) >= args.test_fraction * len(subset):
                break
            test_speakers.add(speaker)
    selected = [{**r, 'split': 'test' if r['speaker'] in test_speakers else 'train'}
                for r in sorted(selected, key=lambda r: r['id'])]

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open('w', encoding='utf-8') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    def counts(key):
        values = {}
        for row in selected:
            values[str(row[key])] = values.get(str(row[key]), 0) + 1
        return values

    durations = [r['duration_s'] for r in selected]
    summary = {
        'input_manifest': str(args.input_manifest), 'eligible_unflagged': len(eligible),
        'total_selected': len(selected), 'by_dataset': counts('dataset'),
        'stop_coda': counts('is_stop_coda'), 'by_split': counts('split'),
        'speakers': {split: len({r['speaker'] for r in selected if r['split'] == split})
                     for split in ('train', 'test')},
        'max_clips_per_speaker': max(per_speaker.values(), default=0),
        'duration_s': {'min': min(durations, default=0), 'max': max(durations, default=0),
                       'mean': round(statistics.fmean(durations), 3) if durations else 0,
                       'total': round(sum(durations), 2)},
        'parameters': {k: v for k, v in vars(args).items() if k not in {'input_manifest', 'output_manifest', 'overwrite'}},
    }
    write_json(args.output_manifest.with_name('anchors_summary.json'), summary)
    progress('SELECTED', json.dumps({k: summary[k] for k in ('total_selected', 'by_dataset', 'stop_coda', 'by_split', 'speakers')}))
    print(args.output_manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
