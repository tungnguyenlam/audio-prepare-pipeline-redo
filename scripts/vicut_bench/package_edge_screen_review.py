"""Package a blind human listening batch that checks the anchor edge screen (GOAL §6).

Samples frozen anchors (half stop coda, half open, one per speaker within each class), VIVOS clips the
screen flagged (one per flag combination, largest combination first) and a random
sample of flagged PhoAudiobook clips, shuffles them with --seed and writes to
--output-dir:

  REVIEW.md     question, what to listen for, decision, mpv commands
  review.csv    review_id,abs_path,sha256,duration_s,listen_for,your_label,your_note
  playlist.m3u  same absolute paths in the same order
  key.jsonl     review_id -> group and screen measurements (never shown to the listener)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, digest, progress

SECONDS_PER_LABEL = 10
KEY_FIELDS = ('id', 'dataset', 'speaker', 'final_word', 'is_stop_coda', 'screen_flags', 'split',
              'lead_silence_s', 'trail_silence_s', 'lead_zero_s', 'trail_zero_s', 'snr_db')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def one_per_speaker(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    rows = sorted(rows, key=lambda r: r['id'])
    rng.shuffle(rows)
    chosen, speakers = [], set()
    for row in rows:
        if row['speaker'] not in speakers:
            speakers.add(row['speaker'])
            chosen.append(row)
        if len(chosen) == count:
            break
    return chosen


def one_per_flag_combination(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Round-robin over flag combinations, largest combination first."""
    groups: dict[tuple, list[dict]] = {}
    for row in sorted(rows, key=lambda r: r['id']):
        groups.setdefault(tuple(sorted(row['screen_flags'])), []).append(row)
    queues = [groups[k] for k in sorted(groups, key=lambda k: (-len(groups[k]), k))]
    for queue in queues:
        rng.shuffle(queue)
    chosen = []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop())
    return chosen


def review_markdown(out_dir: Path, rows: list[dict], seed: int, minutes: float) -> str:
    first = rows[0]['abs_path']
    return f"""# Edge-screen listening batch

**Question.** Does each clip contain one complete, clean utterance from start to end?
**Decision it controls.** Whether the 300 frozen ViCut-Bench anchors can serve as clean
twins for defect generation, and whether the automatic edge screen and the exclusion of
PhoAudiobook were right.

**Size.** {len(rows)} clips, about {minutes:.0f} min including labelling. Rows are shuffled
with seed `{seed}`. Which clips are anchors and which were flagged is hidden on purpose.

## What to listen for

- **pass**: speech starts and ends completely (the first and last word are whole, the
  final consonant or vowel is not cut), only one speaker, no music, buzz or other noise
  you would not accept in TTS training data.
- **fail**: any of the above is wrong. Write the reason in `your_note`, using one or more
  of `start_cut`, `end_cut`, `other_voice`, `noise`, `other: <words>`.
- **unsure**: you cannot decide after two listens. Say why in `your_note`.

`listen_for` gives the expected last word of each clip from its transcript.

## Result that would change the plan

- Two or more frozen anchors labelled `fail`: anchor selection is revisited before any
  defects are generated.
- Most screen-flagged clips sounding clean: the screen is re-tuned, and PhoAudiobook may
  be reconsidered as a source.

## How to listen

```bash
mpv --playlist={out_dir / 'playlist.m3u'}
mpv --keep-open=no {first}        # replay one clip (path from review.csv)
mpv --start=-1 {first}            # hear only the last second
```

Fill `your_label` (`pass` / `fail` / `unsure`) and optionally `your_note` in
`{out_dir / 'review.csv'}`. Do not edit, reorder or delete other columns or rows.
"""


def main() -> int:
    anchors = Path('.data/vicut_bench/anchors')
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-im', '--input-manifest', type=Path, default=anchors / 'measured_candidates.jsonl')
    p.add_argument('--anchor-manifest', type=Path, default=anchors / 'vicut_bench_anchors_300.jsonl')
    p.add_argument('--audio-root', type=Path, default=anchors, help='Directory that manifest rel_path values are relative to')
    p.add_argument('-od', '--output-dir', type=Path, default=Path('.data/goal/review/edge_screen_anchors'))
    p.add_argument('--anchor-count', type=int, default=24, help='Frozen anchors, split evenly between stop coda and open')
    p.add_argument('--vivos-flagged-count', type=int, default=10)
    p.add_argument('--pho-flagged-count', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-w', '--overwrite', action='store_true')
    args = p.parse_args()
    if (args.output_dir / 'review.csv').exists() and not args.overwrite:
        p.error(f'{args.output_dir}/review.csv exists (it may hold labels); use --overwrite to replace it')
    total = args.anchor_count + args.vivos_flagged_count + args.pho_flagged_count
    if total > 40:
        p.error(f'{total} clips exceeds the 40-clip batch limit in GOAL §6')

    rng = random.Random(args.seed)
    frozen = read_jsonl(args.anchor_manifest)
    frozen_ids = {r['id'] for r in frozen}
    candidates = read_jsonl(args.input_manifest)
    stop_n = args.anchor_count // 2
    groups = {
        'frozen_anchor': one_per_speaker([r for r in frozen if r['is_stop_coda']], stop_n, rng)
        + one_per_speaker([r for r in frozen if not r['is_stop_coda']], args.anchor_count - stop_n, rng),
        'vivos_flagged': one_per_flag_combination(
            [r for r in candidates if r['dataset'] == 'vivos' and r['screen_flags'] and r['id'] not in frozen_ids],
            args.vivos_flagged_count, rng),
        'phoaudiobook_flagged': rng.sample(
            sorted((r for r in candidates if r['dataset'] == 'phoaudiobook' and r['screen_flags']), key=lambda r: r['id']),
            args.pho_flagged_count),
    }
    rows = []
    for group, members in groups.items():
        if len(members) < {'frozen_anchor': args.anchor_count, 'vivos_flagged': args.vivos_flagged_count,
                           'phoaudiobook_flagged': args.pho_flagged_count}[group]:
            progress('SHORTFALL', f'{group}: {len(members)} clips available')
        for row in members:
            path = (args.audio_root / row['rel_path']).resolve()
            rows.append({'group': group, 'abs_path': str(path), 'sha256': digest(path),
                         'duration_s': row['duration_s'], **{k: row.get(k) for k in KEY_FIELDS}})
    rng.shuffle(rows)
    for index, row in enumerate(rows, 1):
        row['review_id'] = f'edge{index:02d}'

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.output_dir.resolve()
    with (out_dir / 'review.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['review_id', 'abs_path', 'sha256', 'duration_s', 'listen_for', 'your_label', 'your_note'])
        for row in rows:
            writer.writerow([row['review_id'], row['abs_path'], row['sha256'], f"{row['duration_s']:.2f}",
                             f"start and end complete; last word '{row['final_word']}'", '', ''])
    (out_dir / 'playlist.m3u').write_text('#EXTM3U\n' + ''.join(r['abs_path'] + '\n' for r in rows), encoding='utf-8')
    with (out_dir / 'key.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps({'review_id': row['review_id'], **{k: v for k, v in row.items() if k != 'review_id'}},
                                    ensure_ascii=False) + '\n')
    minutes = sum(r['duration_s'] + SECONDS_PER_LABEL for r in rows) / 60
    (out_dir / 'REVIEW.md').write_text(review_markdown(out_dir, rows, args.seed, minutes), encoding='utf-8')
    progress('PACKAGED', json.dumps({g: len(m) for g, m in groups.items()} | {'minutes': round(minutes, 1)}))
    print(out_dir / 'REVIEW.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
