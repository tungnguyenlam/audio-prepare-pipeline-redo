"""Measure speech onset/offset and screen ViCut-Bench anchor candidates.

Silero VAD (shared automatic reports under .data/vad/auto/) locates the speech
region. A 5 ms RMS envelope (1 ms hop) of the 100 Hz high-passed audio, which
removes mains hum present in some VIVOS speakers, then places each edge where
sustained energy rises above the noise floor (5th percentile) + --edge-db,
searched within --search-s of the Silero edge. Edge precision is about ±3 ms.
Exact digital-zero padding (PhoAudiobook adds 50 ms per side) is excluded from the
floor and margins and reported as lead/trail_zero_s.

Screening flags are measurements, not verdicts; listen to flagged clips:
  no_vad_speech      Silero found no frame >= --vad-threshold
  low_snr            99th - 5th percentile envelope < --min-snr-db
  tight_start/end    less than --min-margin-s of non-zero silence before onset/after
                     offset (speech may already be clipped by the source segmentation)
  leading/trailing_activity
                     sustained energy more than --gap-s before onset/after offset
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, persist_path, progress
from _common.vad import ensure_vad_report, load_vad_report

FRAME_S = 0.005
HIGHPASS_HZ = 100
HOP_S = 0.001
MIN_RUN_S = 0.02


def envelope_db(audio, rate: int):
    import numpy as np

    frame, hop = round(FRAME_S * rate), round(HOP_S * rate)
    padded = np.pad(audio, (frame // 2, frame - frame // 2 - 1))
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    return 20 * np.log10(np.maximum(np.sqrt(np.mean(windows ** 2, axis=1)), 1e-10))


def sustained_runs(mask) -> list[tuple[int, int]]:
    """Return [start, end) envelope-frame runs of at least MIN_RUN_S."""
    import numpy as np

    change = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return [(int(s), int(e)) for s, e in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))
            if (e - s) * HOP_S >= MIN_RUN_S]


def measure(path: Path, args) -> dict:
    import numpy as np
    import soundfile as sf

    from scipy.signal import butter, sosfiltfilt

    audio, rate = sf.read(path, dtype='float64', always_2d=True)
    audio = audio.mean(axis=1)
    duration = len(audio) / rate
    nonzero = np.flatnonzero(audio)
    report = ensure_vad_report(path, vad_device=args.device)
    candidates, _ = load_vad_report(path, report, args.device)
    speech = [center_s for _, probability, _, center_s in candidates if probability >= args.vad_threshold]
    result = {'vad_report': persist_path(report), 'duration_s': round(duration, 4)}
    if not speech or not len(nonzero):
        return {**result, 'screen_flags': ['no_vad_speech']}

    # Envelope frames stay on the source timeline; only the non-zero core is analyzed.
    start, end = nonzero[0] / rate, (nonzero[-1] + 1) / rate
    filtered = sosfiltfilt(butter(4, HIGHPASS_HZ, 'highpass', fs=rate, output='sos'), audio)
    env = envelope_db(filtered, rate)
    env[(np.arange(len(env)) * HOP_S < start) | (np.arange(len(env)) * HOP_S >= end)] = -200.0
    core = env[env > -200.0]
    floor_db, peak_db = float(np.percentile(core, 5)), float(np.percentile(core, 99))
    runs = sustained_runs(env >= floor_db + args.edge_db)
    vad_on, vad_off = speech[0], speech[-1]
    near_on = [r for r in runs if r[1] * HOP_S > vad_on - args.search_s and r[0] * HOP_S < vad_on + args.search_s]
    near_off = [r for r in runs if r[1] * HOP_S > vad_off - args.search_s and r[0] * HOP_S < vad_off + args.search_s]
    onset = near_on[0][0] * HOP_S if near_on else vad_on
    offset = min(near_off[-1][1] * HOP_S, end) if near_off else vad_off

    flags = []
    if peak_db - floor_db < args.min_snr_db:
        flags.append('low_snr')
    if onset - start < args.min_margin_s:
        flags.append('tight_start')
    if end - offset < args.min_margin_s:
        flags.append('tight_end')
    if any(e * HOP_S < onset - args.gap_s for _, e in runs):
        flags.append('leading_activity')
    if any(s * HOP_S > offset + args.gap_s for s, _ in runs):
        flags.append('trailing_activity')
    if not near_on or not near_off:
        flags.append('no_energy_edge')
    return {**result,
            'speech_onset_s': round(onset, 4), 'speech_offset_s': round(offset, 4),
            'lead_silence_s': round(onset - start, 4), 'trail_silence_s': round(end - offset, 4),
            'lead_zero_s': round(start, 4), 'trail_zero_s': round(duration - end, 4),
            'vad_onset_s': round(vad_on, 4), 'vad_offset_s': round(vad_off, 4),
            'floor_db': round(floor_db, 2), 'peak_db': round(peak_db, 2),
            'snr_db': round(peak_db - floor_db, 2), 'screen_flags': flags}


def main() -> int:
    anchors = Path('.data/vicut_bench/anchors')
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-im', '--input-manifest', type=Path, default=anchors / 'all_candidates.jsonl',
                   help='Anchor JSONL with rel_path relative to the manifest directory')
    p.add_argument('-om', '--output-manifest', type=Path, default=anchors / 'measured_candidates.jsonl')
    p.add_argument('-d', '--device', default='auto', help='Silero device: auto, cpu, or cuda:N (also ROCm)')
    p.add_argument('--vad-threshold', type=float, default=0.5)
    p.add_argument('--edge-db', type=float, default=10.0, help='Edge level above the clip noise floor')
    p.add_argument('--search-s', type=float, default=0.2, help='Energy-edge search radius around the Silero edge')
    p.add_argument('--min-margin-s', type=float, default=0.05)
    p.add_argument('--gap-s', type=float, default=0.1)
    p.add_argument('--min-snr-db', type=float, default=30.0)
    p.add_argument('--all', action='store_true', help='Also measure rows outside the 3-12 s window')
    p.add_argument('-w', '--overwrite', action='store_true')
    args = p.parse_args()
    if args.output_manifest.exists() and not args.overwrite:
        p.error('Output exists; choose another --output-manifest or use --overwrite')

    rows = [json.loads(line) for line in args.input_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not args.all:
        rows = [row for row in rows if row.get('in_3_12s_window')]
    base = args.input_manifest.parent
    measured = []
    for index, row in enumerate(rows, 1):
        edges = measure(base / row['rel_path'], args)
        measured.append({**row, **edges})
        progress('MEASURE', f"{row['id']} {','.join(edges['screen_flags']) or 'ok'}", current=index, total=len(rows))

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open('w', encoding='utf-8') as stream:
        for row in measured:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    for dataset in sorted({row['dataset'] for row in measured}):
        subset = [row for row in measured if row['dataset'] == dataset]
        counts = {}
        for row in subset:
            for flag in row['screen_flags']:
                counts[flag] = counts.get(flag, 0) + 1
        clean = sum(not row['screen_flags'] for row in subset)
        progress('SUMMARY', f'{dataset}: {clean}/{len(subset)} unflagged; {counts}')
    print(args.output_manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
