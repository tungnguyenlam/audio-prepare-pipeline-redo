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


def review_html(out_dir: Path, rows: list[dict], seed: int, minutes: float) -> str:
    import html
    cards = []
    for r in rows:
        rel_audio = html.escape(str(Path(r['abs_path']).relative_to(out_dir.parent.parent.parent)))
        # Path relative to repo root: .data/...
        repo_rel = f"../../../../{html.escape(str(Path(r['abs_path']).relative_to(out_dir.resolve().parents[3])))}"
        rid = html.escape(r['review_id'])
        listen_for = html.escape(r['listen_for'])
        dur = f"{r['duration_s']:.2f}"
        cards.append(f"""
      <tr id="row-{rid}">
        <td class="id-col">{rid}</td>
        <td class="dur-col">{dur}s</td>
        <td class="inst-col">{listen_for}</td>
        <td class="audio-col">
          <audio id="audio-{rid}" src="{repo_rel}" preload="none" controls></audio>
          <button type="button" class="btn-sub" onclick="playLastSec('{rid}')">Last 1s</button>
        </td>
        <td class="lbl-col">
          <select id="lbl-{rid}" onchange="updateRow('{rid}')">
            <option value="">-- select --</option>
            <option value="pass">pass</option>
            <option value="fail">fail</option>
            <option value="unsure">unsure</option>
          </select>
        </td>
        <td class="note-col">
          <input type="text" id="note-{rid}" placeholder="start_cut, end_cut, etc." oninput="updateRow('{rid}')">
        </td>
      </tr>""")

    tbody = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Edge-Screen Review ({len(rows)} clips)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #121212; color: #e0e0e0; }}
    h1 {{ font-size: 1.4rem; color: #fff; margin-bottom: 4px; }}
    .meta {{ font-size: 0.9rem; color: #888; margin-bottom: 16px; }}
    .panel {{ background: #1e1e1e; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #333; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #2a2a2a; text-align: left; vertical-align: middle; }}
    th {{ background: #252525; color: #aaa; font-weight: 600; position: sticky; top: 0; }}
    tr:hover {{ background: #1a1a1a; }}
    tr.done {{ background: #172417; }}
    audio {{ height: 32px; vertical-align: middle; width: 220px; }}
    button.btn-sub {{ background: #333; color: #ccc; border: 1px solid #555; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; margin-left: 4px; }}
    button.btn-sub:hover {{ background: #444; color: #fff; }}
    button.btn-primary {{ background: #2e7d32; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: 600; margin-right: 8px; }}
    button.btn-primary:hover {{ background: #388e3c; }}
    select, input[type="text"] {{ background: #222; color: #fff; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; }}
    input[type="text"] {{ width: 95%; }}
    .id-col {{ font-family: monospace; font-weight: bold; width: 60px; }}
    .dur-col {{ width: 50px; color: #888; }}
    .lbl-col {{ width: 100px; }}
  </style>
</head>
<body>
  <h1>Edge-Screen Human Review</h1>
  <div class="meta">
    {len(rows)} clips (~{minutes:.0f} min). Shuffled with seed {seed}. Labels saved to browser storage and downloadable as <code>review.csv</code>.
  </div>

  <div class="panel">
    <strong>Instructions:</strong>
    <ul>
      <li><strong>pass:</strong> clean speech, whole first/last word, single speaker, no noise/bleed.</li>
      <li><strong>fail:</strong> start cut, end cut, extra voice, or noise (note reason in Note column).</li>
      <li><strong>unsure:</strong> uncertain after 2 listens.</li>
    </ul>
    <div style="margin-top: 12px;">
      <button class="btn-primary" onclick="exportCSV()">Download review.csv</button>
      <button class="btn-sub" onclick="copyCSV()">Copy CSV to Clipboard</button>
      <span id="save-status" style="margin-left: 10px; color: #888; font-size: 0.85rem;"></span>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Dur</th>
        <th>Listen For</th>
        <th>Audio</th>
        <th>Your Label</th>
        <th>Your Note</th>
      </tr>
    </thead>
    <tbody>
      {tbody}
    </tbody>
  </table>

  <script>
    const rowsData = {json.dumps([{'review_id': r['review_id'], 'abs_path': r['abs_path'], 'sha256': r['sha256'], 'duration_s': f"{r['duration_s']:.2f}", 'listen_for': r['listen_for']} for r in rows])};

    function playLastSec(id) {{
      const a = document.getElementById('audio-' + id);
      if (!a) return;
      if (isNaN(a.duration) || a.duration === 0) {{
        a.addEventListener('loadedmetadata', () => {{
          a.currentTime = Math.max(0, a.duration - 1.0);
          a.play();
        }}, {{ once: true }});
        a.load();
      }} else {{
        a.currentTime = Math.max(0, a.duration - 1.0);
        a.play();
      }}
    }}

    function updateRow(id) {{
      const lbl = document.getElementById('lbl-' + id).value;
      const note = document.getElementById('note-' + id).value;
      const tr = document.getElementById('row-' + id);
      if (lbl) {{
        tr.classList.add('done');
      }} else {{
        tr.classList.remove('done');
      }}
      localStorage.setItem('review_' + id, JSON.stringify({{ lbl, note }}));
      document.getElementById('save-status').innerText = 'Saved locally at ' + new Date().toLocaleTimeString();
    }}

    function loadSaved() {{
      rowsData.forEach(r => {{
        const saved = localStorage.getItem('review_' + r.review_id);
        if (saved) {{
          try {{
            const data = JSON.parse(saved);
            if (data.lbl) document.getElementById('lbl-' + r.review_id).value = data.lbl;
            if (data.note) document.getElementById('note-' + r.review_id).value = data.note;
            if (data.lbl) document.getElementById('row-' + r.review_id).classList.add('done');
          }} catch(e) {{}}
        }}
      }});
    }}

    function buildCSV() {{
      const lines = ['review_id,abs_path,sha256,duration_s,listen_for,your_label,your_note'];
      rowsData.forEach(r => {{
        const lbl = (document.getElementById('lbl-' + r.review_id).value || '').replace(/"/g, '""');
        const note = (document.getElementById('note-' + r.review_id).value || '').replace(/"/g, '""');
        const inst = r.listen_for.replace(/"/g, '""');
        lines.push(`"${{r.review_id}}","${{r.abs_path}}","${{r.sha256}}","${{r.duration_s}}","${{inst}}","${{lbl}}","${{note}}"`);
      }});
      return lines.join('\\n') + '\\n';
    }}

    function exportCSV() {{
      const csv = buildCSV();
      const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'review.csv';
      a.click();
      URL.revokeObjectURL(url);
    }}

    function copyCSV() {{
      const csv = buildCSV();
      navigator.clipboard.writeText(csv).then(() => {{
        alert('CSV copied to clipboard! Paste it into .data/goal/review/edge_screen_anchors/review.csv');
      }});
    }}

    window.addEventListener('DOMContentLoaded', loadSaved);
  </script>
</body>
</html>
"""


def review_markdown(out_dir: Path, rows: list[dict], seed: int, minutes: float) -> str:
    first = rows[0]['abs_path']
    rel_html = out_dir / 'review.html'
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

### Option A: Web Browser via SSH Port-Forward (Recommended)
1. In another terminal on your local machine, run:
   ```bash
   ssh -L 8080:localhost:8080 <your-ssh-host>
   ```
2. On this remote machine, serve the repository root:
   ```bash
   python3 -m http.server 8080
   ```
3. Open in your local browser:
   `http://localhost:8080/{rel_html}`
4. Fill in labels, test "Last 1s" buttons, and click **Download review.csv** or **Copy CSV to Clipboard**, then save to `{out_dir / 'review.csv'}`.

### Option B: Local mpv
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
        row['listen_for'] = f"start and end complete; last word '{row['final_word']}'"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.output_dir.resolve()
    with (out_dir / 'review.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['review_id', 'abs_path', 'sha256', 'duration_s', 'listen_for', 'your_label', 'your_note'])
        for row in rows:
            writer.writerow([row['review_id'], row['abs_path'], row['sha256'], f"{row['duration_s']:.2f}",
                             row['listen_for'], '', ''])
    (out_dir / 'playlist.m3u').write_text('#EXTM3U\n' + ''.join(r['abs_path'] + '\n' for r in rows), encoding='utf-8')
    with (out_dir / 'key.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps({'review_id': row['review_id'], **{k: v for k, v in row.items() if k != 'review_id'}},
                                    ensure_ascii=False) + '\n')
    minutes = sum(r['duration_s'] + SECONDS_PER_LABEL for r in rows) / 60
    (out_dir / 'REVIEW.md').write_text(review_markdown(out_dir, rows, args.seed, minutes), encoding='utf-8')
    (out_dir / 'review.html').write_text(review_html(out_dir, rows, args.seed, minutes), encoding='utf-8')
    progress('PACKAGED', json.dumps({g: len(m) for g, m in groups.items()} | {'minutes': round(minutes, 1)}))
    print(out_dir / 'REVIEW.md')
    print(out_dir / 'review.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
