"""Materialize and catalog clean anchor audio for ViCut-Bench (Phase 1).

Extracts PhoAudiobook test clips from Parquet and catalogs VIVOS test clips,
characterizes their duration and phonological coda (-p/-t/-c/-ch vs open/nasal),
and generates the frozen canonical 300-anchor pool.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import shutil
import sys
import wave

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


STOP_CODAS = ('p', 't', 'c', 'ch')


def get_final_word(text: str) -> str:
    words = re.findall(r'\b\w+\b', text.strip().lower())
    return words[-1] if words else ''


def is_stop_coda(word: str) -> bool:
    return word.endswith(STOP_CODAS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--phoaudiobook-parquet',
        type=Path,
        default=Path('.data/s1-download/phoaudiobook/test-00000-of-00001.parquet'),
        help='Path to PhoAudiobook test parquet',
    )
    parser.add_argument(
        '--vivos-test-dir',
        type=Path,
        default=Path('.data/s1-download/vivos/vivos/test'),
        help='Path to VIVOS test directory',
    )
    parser.add_argument(
        '-od', '--output-dir',
        type=Path,
        default=Path('.data/vicut_bench/anchors'),
        help='Output directory for materialized anchors',
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=300,
        help='Canonical anchor pool size (default: 300)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible anchor selection (default: 42)',
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    out_dir = args.output_dir
    pho_out = out_dir / 'phoaudiobook'
    vivos_out = out_dir / 'vivos'
    pho_out.mkdir(parents=True, exist_ok=True)
    vivos_out.mkdir(parents=True, exist_ok=True)

    all_anchors = []

    # 1. Process PhoAudiobook test set
    if args.phoaudiobook_parquet.exists():
        print(f'Reading PhoAudiobook from {args.phoaudiobook_parquet}...', file=sys.stderr)
        tab = pq.read_table(args.phoaudiobook_parquet)
        for i in range(len(tab)):
            row = tab.slice(i, 1).to_pydict()
            speaker = str(row['speaker'][0])
            text = str(row['text'][0]).strip()
            audio_bytes = row['audio'][0]['bytes']
            
            clean_spk = re.sub(r'[^\w\-]', '_', speaker)
            wav_name = f'pho_test_{i:04d}_{clean_spk}.wav'
            wav_path = pho_out / wav_name

            if not wav_path.exists():
                with open(wav_path, 'wb') as f:
                    f.write(audio_bytes)

            data, sr = sf.read(io.BytesIO(audio_bytes))
            dur = float(len(data) / sr)

            final_w = get_final_word(text)
            has_stop = is_stop_coda(final_w)

            record = {
                'id': f'pho_{i:04d}',
                'dataset': 'phoaudiobook',
                'speaker': speaker,
                'rel_path': str(wav_path.relative_to(out_dir)),
                'duration_s': round(dur, 4),
                'sample_rate': sr,
                'text': text,
                'final_word': final_w,
                'is_stop_coda': has_stop,
                'in_3_12s_window': bool(3.0 <= dur <= 12.0),
            }
            all_anchors.append(record)

            sidecar = wav_path.with_suffix('.json')
            if not sidecar.exists():
                with open(sidecar, 'w', encoding='utf-8') as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)

    # 2. Process VIVOS test set
    prompts_file = args.vivos_test_dir / 'prompts.txt'
    waves_dir = args.vivos_test_dir / 'waves'
    if prompts_file.exists() and waves_dir.exists():
        print(f'Reading VIVOS test from {args.vivos_test_dir}...', file=sys.stderr)
        prompts = {}
        for line in prompts_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                prompts[parts[0]] = parts[1]

        for p in sorted(waves_dir.rglob('*.wav')):
            cid = p.stem
            speaker = p.parent.name
            dest_wav = vivos_out / f'vivos_{speaker}_{cid}.wav'
            if not dest_wav.exists():
                try:
                    dest_wav.hardlink_to(p)
                except OSError:
                    shutil.copy2(p, dest_wav)

            with wave.open(str(p), 'rb') as wf:
                sr = wf.getframerate()
                dur = float(wf.getnframes() / sr)

            text = prompts.get(cid, '').strip()
            final_w = get_final_word(text)
            has_stop = is_stop_coda(final_w)

            record = {
                'id': f'vivos_{cid}',
                'dataset': 'vivos',
                'speaker': speaker,
                'rel_path': str(dest_wav.relative_to(out_dir)),
                'duration_s': round(dur, 4),
                'sample_rate': sr,
                'text': text,
                'final_word': final_w,
                'is_stop_coda': has_stop,
                'in_3_12s_window': bool(3.0 <= dur <= 12.0),
            }
            all_anchors.append(record)

            sidecar = dest_wav.with_suffix('.json')
            if not sidecar.exists():
                with open(sidecar, 'w', encoding='utf-8') as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)

    print(f'Total candidates indexed: {len(all_anchors)}', file=sys.stderr)

    # Filter to eligible candidates in 3-12s window
    candidates = [a for a in all_anchors if a['in_3_12s_window'] and a['text']]
    print(f'Eligible candidates in [3.0, 12.0] s window: {len(candidates)}', file=sys.stderr)

    pho_candidates = [a for a in candidates if a['dataset'] == 'phoaudiobook']
    vivos_candidates = [a for a in candidates if a['dataset'] == 'vivos']

    print(f'  PhoAudiobook eligible: {len(pho_candidates)}', file=sys.stderr)
    print(f'  VIVOS eligible: {len(vivos_candidates)}', file=sys.stderr)

    # Separate by stop coda
    pho_stops = [a for a in pho_candidates if a['is_stop_coda']]
    pho_opens = [a for a in pho_candidates if not a['is_stop_coda']]
    vivos_stops = [a for a in vivos_candidates if a['is_stop_coda']]
    vivos_opens = [a for a in vivos_candidates if not a['is_stop_coda']]

    # Selection target: 300 anchors
    # Target 100 stops (40 PhoAudiobook, 60 VIVOS)
    # Target 200 opens (100 PhoAudiobook, 100 VIVOS)
    np.random.shuffle(pho_stops)
    np.random.shuffle(vivos_stops)
    np.random.shuffle(pho_opens)
    np.random.shuffle(vivos_opens)

    n_pho_stops = min(40, len(pho_stops))
    n_vivos_stops = min(60, len(vivos_stops))
    sel_stops = pho_stops[:n_pho_stops] + vivos_stops[:n_vivos_stops]

    remaining_needed = args.target_count - len(sel_stops)
    n_pho_opens = min(remaining_needed // 2, len(pho_opens))
    n_vivos_opens = remaining_needed - n_pho_opens
    sel_opens = pho_opens[:n_pho_opens] + vivos_opens[:n_vivos_opens]

    selected_300 = sel_stops + sel_opens
    np.random.shuffle(selected_300)

    selected_ids = {a['id'] for a in selected_300}
    for a in all_anchors:
        a['selected_in_300'] = a['id'] in selected_ids

    # Write all candidates manifest
    all_manifest = out_dir / 'all_candidates.jsonl'
    with open(all_manifest, 'w', encoding='utf-8') as f:
        for a in all_anchors:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')

    # Write canonical 300 manifest
    manifest_300 = out_dir / 'vicut_bench_anchors_300.jsonl'
    with open(manifest_300, 'w', encoding='utf-8') as f:
        for a in selected_300:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')

    # Summary statistics
    speakers_300 = {a['speaker'] for a in selected_300}
    durations_300 = [a['duration_s'] for a in selected_300]
    stops_300 = [a for a in selected_300 if a['is_stop_coda']]
    opens_300 = [a for a in selected_300 if not a['is_stop_coda']]
    pho_300 = [a for a in selected_300 if a['dataset'] == 'phoaudiobook']
    vivos_300 = [a for a in selected_300 if a['dataset'] == 'vivos']

    summary = {
        'total_selected': len(selected_300),
        'by_dataset': {
            'phoaudiobook': len(pho_300),
            'vivos': len(vivos_300),
        },
        'by_phonological_coda': {
            'stop_consonants': len(stops_300),
            'open_syllables': len(opens_300),
        },
        'distinct_speakers': len(speakers_300),
        'duration_stats': {
            'min_s': float(np.min(durations_300)),
            'max_s': float(np.max(durations_300)),
            'mean_s': float(np.mean(durations_300)),
            'median_s': float(np.median(durations_300)),
            'total_duration_s': float(np.sum(durations_300)),
            'total_duration_min': float(np.sum(durations_300) / 60.0),
        },
    }

    summary_file = out_dir / 'anchors_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('=' * 60, file=sys.stderr)
    print(f'Successfully materialized {len(selected_300)} clean anchors!', file=sys.stderr)
    print(f'  Dataset breakdown: PhoAudiobook={len(pho_300)}, VIVOS={len(vivos_300)}', file=sys.stderr)
    print(f'  Coda breakdown: Stop consonants={len(stops_300)}, Open controls={len(opens_300)}', file=sys.stderr)
    print(f'  Distinct speakers: {len(speakers_300)}', file=sys.stderr)
    print(f'  Total duration: {summary["duration_stats"]["total_duration_min"]:.2f} min ({summary["duration_stats"]["mean_s"]:.2f} s avg)', file=sys.stderr)
    print(f'  Canonical manifest: {manifest_300}', file=sys.stderr)
    print(f'  Summary report: {summary_file}', file=sys.stderr)
    print('=' * 60, file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
