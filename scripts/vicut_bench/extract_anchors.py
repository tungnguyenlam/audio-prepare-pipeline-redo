"""Materialize and catalog candidate clean anchor audio for ViCut-Bench (Phase 1).

Extracts PhoAudiobook test clips from Parquet and hardlinks VIVOS clips (a seeded
sample of up to --vivos-per-speaker clips per speaker from each split directory),
records duration and the final-word coda (-p/-t/-c/-ch vs open/nasal), and writes
all_candidates.jsonl. Edge measurement and selection are measure_anchors and
select_anchors.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import random
import re
import shutil
import sys
import wave

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
        '--vivos-dir',
        type=Path,
        action='append',
        help='VIVOS split directory with prompts.txt and waves/; repeatable '
             '(default: .data/s1-download/vivos/vivos/test and .../train)',
    )
    parser.add_argument('--vivos-per-speaker', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '-od', '--output-dir',
        type=Path,
        default=Path('.data/vicut_bench/anchors'),
        help='Output directory for materialized anchors',
    )
    args = parser.parse_args()

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

    # 2. Process VIVOS splits
    vivos_root = Path('.data/s1-download/vivos/vivos')
    for vivos_dir in args.vivos_dir or [vivos_root / 'test', vivos_root / 'train']:
        prompts_file = vivos_dir / 'prompts.txt'
        waves_dir = vivos_dir / 'waves'
        if not (prompts_file.exists() and waves_dir.exists()):
            print(f'Skipping missing VIVOS split {vivos_dir}', file=sys.stderr)
            continue
        print(f'Reading VIVOS from {vivos_dir}...', file=sys.stderr)
        prompts = {}
        for line in prompts_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                prompts[parts[0]] = parts[1]

        rng = random.Random(f'{args.seed}:{vivos_dir.name}')
        sampled = []
        for speaker_dir in sorted(d for d in waves_dir.iterdir() if d.is_dir()):
            wavs = sorted(speaker_dir.glob('*.wav'))
            sampled += sorted(rng.sample(wavs, min(args.vivos_per_speaker, len(wavs))))
        for p in sampled:
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
                'source_split': vivos_dir.name,
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

            with open(dest_wav.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

    all_manifest = out_dir / 'all_candidates.jsonl'
    with open(all_manifest, 'w', encoding='utf-8') as f:
        for a in all_anchors:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    eligible = sum(a['in_3_12s_window'] and bool(a['text']) for a in all_anchors)
    print(f'Indexed {len(all_anchors)} candidates; {eligible} in the 3-12 s window', file=sys.stderr)
    print(all_manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
