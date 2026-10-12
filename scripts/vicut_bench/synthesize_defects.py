"""Synthesize controlled ViCut-Bench curation defects on clean anchors.

Applies exact, parameterised defect operators relative to detected speech
onset and offset (never file edges):
  1. clipped_word_end   : Truncates audio at speech_offset - delta_t (10, 20, 40, 80 ms).
  2. clipped_word_start : Truncates audio at speech_onset + delta_t (10, 20, 40, 80 ms).
  3. secondary_speaker  : Overlays a 0.5 s fragment from a disjoint speaker at SIR (24, 18, 12, 6 dB).
  4. background_music   : Mixes additive music bed at SNR (30, 24, 18, 12 dB).

Writes sibling .json sidecars and a unified manifest linking each defect
to its exact clean anchor twin.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (
    LoggingArgumentParser,
    persist_path,
    progress,
    resolve_stored_path,
    write_json,
)

DEFECT_TYPES = ('clipped_word_end', 'clipped_word_start', 'secondary_speaker', 'background_music')
END_SEVERITIES_MS = (10, 20, 40, 80)
START_SEVERITIES_MS = (10, 20, 40, 80)
SPEAKER_SIRS_DB = (24.0, 18.0, 12.0, 6.0)
MUSIC_SNRS_DB = (30.0, 24.0, 18.0, 12.0)


def calculate_active_rms(audio: np.ndarray, threshold_db: float = -50.0) -> float:
    if len(audio) == 0:
        return 1e-9
    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))
    if peak <= 1e-7:
        return 1e-9
    threshold = peak * (10.0 ** (threshold_db / 20.0))
    active = audio[abs_audio >= threshold]
    if len(active) == 0:
        active = audio
    return float(np.sqrt(np.mean(active**2)))


def apply_clipped_word_end(
    audio: np.ndarray, sr: int, row: dict, delta_t_ms: int
) -> tuple[np.ndarray, dict]:
    delta_t_s = delta_t_ms / 1000.0
    speech_offset = float(row.get('speech_offset_s', len(audio) / sr))
    cut_time_s = max(0.1, speech_offset - delta_t_s)
    cut_sample = min(len(audio), max(1, int(round(cut_time_s * sr))))
    defective_audio = audio[:cut_sample].copy()

    meta = {
        'operator': 'clipped_word_end',
        'severity': f'{delta_t_ms}ms',
        'severity_ms': delta_t_ms,
        'delta_t_s': delta_t_s,
        'cut_time_s': round(cut_time_s, 4),
        'original_speech_offset_s': round(speech_offset, 4),
        'duration_s': round(len(defective_audio) / sr, 4),
        'is_stop_coda': row.get('is_stop_coda'),
        'final_word': row.get('final_word'),
    }
    return defective_audio, meta


def apply_clipped_word_start(
    audio: np.ndarray, sr: int, row: dict, delta_t_ms: int
) -> tuple[np.ndarray, dict]:
    delta_t_s = delta_t_ms / 1000.0
    speech_onset = float(row.get('speech_onset_s', 0.0))
    cut_time_s = min(float(row.get('duration_s', len(audio) / sr)) - 0.1, speech_onset + delta_t_s)
    cut_sample = min(len(audio) - 1, max(0, int(round(cut_time_s * sr))))
    defective_audio = audio[cut_sample:].copy()

    meta = {
        'operator': 'clipped_word_start',
        'severity': f'{delta_t_ms}ms',
        'severity_ms': delta_t_ms,
        'delta_t_s': delta_t_s,
        'cut_time_s': round(cut_time_s, 4),
        'original_speech_onset_s': round(speech_onset, 4),
        'duration_s': round(len(defective_audio) / sr, 4),
    }
    return defective_audio, meta


def apply_secondary_speaker(
    target_audio: np.ndarray,
    sr: int,
    target_row: dict,
    donor_audio: np.ndarray,
    donor_row: dict,
    sir_db: float,
    rng: random.Random,
) -> tuple[np.ndarray, dict]:
    donor_onset = float(donor_row.get('speech_onset_s', 0.0))
    donor_offset = float(donor_row.get('speech_offset_s', len(donor_audio) / sr))
    donor_speech_len = max(0.2, donor_offset - donor_onset)

    frag_dur = min(0.6, donor_speech_len * 0.8)
    frag_samples = int(round(frag_dur * sr))
    frag_start = int(round(donor_onset * sr))
    frag = donor_audio[frag_start : frag_start + frag_samples].copy()

    if len(frag) < int(0.2 * sr):
        frag = donor_audio[: min(len(donor_audio), int(0.5 * sr))].copy()

    t_onset = float(target_row.get('speech_onset_s', 0.0))
    t_offset = float(target_row.get('speech_offset_s', len(target_audio) / sr))
    active_span = max(0.1, t_offset - t_onset)
    rel_pos = rng.uniform(0.2, 0.7)
    overlay_start_s = t_onset + active_span * rel_pos
    pos = int(round(overlay_start_s * sr))
    pos = max(0, min(len(target_audio) - len(frag), pos))

    target_rms = calculate_active_rms(target_audio)
    donor_rms = calculate_active_rms(frag)
    gain = target_rms / (donor_rms * (10.0 ** (sir_db / 20.0)))
    scaled_frag = frag * gain

    out = target_audio.copy()
    end_pos = min(len(out), pos + len(scaled_frag))
    out[pos:end_pos] += scaled_frag[: end_pos - pos]

    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out = out * (0.98 / peak)

    meta = {
        'operator': 'secondary_speaker',
        'severity': f'SIR_{int(sir_db)}dB',
        'sir_db': sir_db,
        'donor_anchor_id': donor_row.get('id'),
        'donor_speaker': donor_row.get('speaker'),
        'fragment_duration_s': round(len(frag) / sr, 4),
        'overlay_start_s': round(pos / sr, 4),
        'duration_s': round(len(out) / sr, 4),
    }
    return out, meta


def apply_background_music(
    target_audio: np.ndarray,
    sr: int,
    music_audio: np.ndarray,
    music_sr: int,
    snr_db: float,
    rng: random.Random,
) -> tuple[np.ndarray, dict]:
    import scipy.signal

    if music_sr != sr:
        num_samples = int(round(len(music_audio) * sr / music_sr))
        music_audio = scipy.signal.resample(music_audio, num_samples)

    if music_audio.ndim > 1:
        music_audio = np.mean(music_audio, axis=1)

    needed = len(target_audio)
    if len(music_audio) < needed:
        repeats = int(np.ceil(needed / len(music_audio)))
        music_audio = np.tile(music_audio, repeats)

    max_start = max(0, len(music_audio) - needed)
    start_idx = rng.randint(0, max_start) if max_start > 0 else 0
    music_crop = music_audio[start_idx : start_idx + needed].copy()

    target_rms = calculate_active_rms(target_audio)
    music_rms = calculate_active_rms(music_crop)
    gain = target_rms / (music_rms * (10.0 ** (snr_db / 20.0)))
    scaled_music = music_crop * gain

    out = target_audio + scaled_music
    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out = out * (0.98 / peak)

    meta = {
        'operator': 'background_music',
        'severity': f'SNR_{int(snr_db)}dB',
        'snr_db': snr_db,
        'music_offset_s': round(start_idx / sr, 4),
        'duration_s': round(len(out) / sr, 4),
    }
    return out, meta


def main() -> int:
    default_manifest = Path('.data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl')
    default_out_dir = Path('.data/vicut_bench/defects')
    default_music = Path(
        '.data/s2-separate/htdemucs/instrumental/a9aVQCWi9MY_89-63s_44100Hz_1ch_htdemucs.wav'
    )

    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-im', '--input-manifest', type=Path, default=default_manifest,
                   help='Path to clean anchors manifest JSONL')
    p.add_argument('-od', '--output-dir', type=Path, default=default_out_dir,
                   help='Directory to write synthesized defect audio and sidecars')
    p.add_argument('-om', '--output-manifest', type=Path, default=None,
                   help='Path to output defects manifest JSONL')
    p.add_argument('--defect-type', choices=('all',) + DEFECT_TYPES, default='all',
                   help='Defect type to synthesize (default: all)')
    p.add_argument('--music-file', type=Path, default=default_music,
                   help='Background music source for background_music defect')
    p.add_argument('--max-anchors', type=int, default=None,
                   help='Limit number of anchors to process (useful for previewing)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for deterministic defect synthesis (default: 42)')
    p.add_argument('-w', '--overwrite', action='store_true',
                   help='Overwrite existing defect files')
    args = p.parse_args()

    input_manifest = resolve_stored_path(args.input_manifest)
    if not input_manifest.exists():
        p.error(f'Input manifest not found: {input_manifest}')

    out_dir = resolve_stored_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = resolve_stored_path(args.output_manifest) if args.output_manifest else out_dir / 'vicut_bench_defects.jsonl'
    if out_manifest.exists() and not args.overwrite:
        p.error(f'Output manifest exists: {out_manifest}; use --overwrite to replace')

    rows = [json.loads(line) for line in input_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.max_anchors is not None and args.max_anchors < len(rows):
        rows = rows[: args.max_anchors]

    progress('INIT', f'Loaded {len(rows)} clean anchors from {input_manifest}')

    anchor_base = input_manifest.parent
    anchors_by_id = {}
    speaker_anchors = {}

    for row in rows:
        audio_path = resolve_stored_path(anchor_base / row['rel_path'])
        if not audio_path.exists():
            continue
        anchors_by_id[row['id']] = (row, audio_path)
        speaker_anchors.setdefault(row['speaker'], []).append(row['id'])

    rng = random.Random(args.seed)
    selected_defects = DEFECT_TYPES if args.defect_type == 'all' else (args.defect_type,)

    music_audio, music_sr = None, None
    if 'background_music' in selected_defects:
        music_path = resolve_stored_path(args.music_file)
        if music_path.exists():
            music_audio, music_sr = sf.read(music_path, dtype='float32')
            progress('MUSIC_LOAD', f'Loaded music reference from {music_path} ({len(music_audio)/music_sr:.1f}s)')
        else:
            progress('WARN', f'Music file not found: {music_path}; skipping background_music operator')

    results = []
    total_generated = 0

    for idx, row in enumerate(rows, start=1):
        anchor_id = row['id']
        if anchor_id not in anchors_by_id:
            continue
        _, audio_path = anchors_by_id[anchor_id]
        clean_audio, sr = sf.read(audio_path, dtype='float32')
        if clean_audio.ndim > 1:
            clean_audio = np.mean(clean_audio, axis=1)

        for defect in selected_defects:
            defect_sub_dir = out_dir / defect
            defect_sub_dir.mkdir(parents=True, exist_ok=True)

            if defect == 'clipped_word_end':
                for sev in END_SEVERITIES_MS:
                    clip_id = f'{anchor_id}__end_{sev}ms'
                    out_wav = defect_sub_dir / f'{clip_id}.wav'
                    out_json = defect_sub_dir / f'{clip_id}.json'
                    if out_wav.exists() and not args.overwrite:
                        continue
                    mod_audio, meta = apply_clipped_word_end(clean_audio, sr, row, sev)
                    sf.write(out_wav, mod_audio, sr)
                    record = {
                        'id': clip_id,
                        'defect_type': defect,
                        'anchor_id': anchor_id,
                        'audio_path': persist_path(out_wav),
                        'twin_audio_path': row['rel_path'],
                        'split': row.get('split', 'train'),
                        'speaker': row['speaker'],
                        'sample_rate': sr,
                        **meta,
                        'transcript': row.get('text', ''),
                    }
                    write_json(out_json, record)
                    results.append(record)
                    total_generated += 1

            elif defect == 'clipped_word_start':
                for sev in START_SEVERITIES_MS:
                    clip_id = f'{anchor_id}__start_{sev}ms'
                    out_wav = defect_sub_dir / f'{clip_id}.wav'
                    out_json = defect_sub_dir / f'{clip_id}.json'
                    if out_wav.exists() and not args.overwrite:
                        continue
                    mod_audio, meta = apply_clipped_word_start(clean_audio, sr, row, sev)
                    sf.write(out_wav, mod_audio, sr)
                    record = {
                        'id': clip_id,
                        'defect_type': defect,
                        'anchor_id': anchor_id,
                        'audio_path': persist_path(out_wav),
                        'twin_audio_path': row['rel_path'],
                        'split': row.get('split', 'train'),
                        'speaker': row['speaker'],
                        'sample_rate': sr,
                        **meta,
                        'transcript': row.get('text', ''),
                    }
                    write_json(out_json, record)
                    results.append(record)
                    total_generated += 1

            elif defect == 'secondary_speaker':
                donor_candidates = [
                    spk for spk in speaker_anchors.keys() if spk != row['speaker']
                ]
                if not donor_candidates:
                    continue
                donor_spk = rng.choice(donor_candidates)
                donor_id = rng.choice(speaker_anchors[donor_spk])
                donor_row, donor_path = anchors_by_id[donor_id]
                donor_audio, donor_sr = sf.read(donor_path, dtype='float32')
                if donor_audio.ndim > 1:
                    donor_audio = np.mean(donor_audio, axis=1)

                for sir in SPEAKER_SIRS_DB:
                    clip_id = f'{anchor_id}__secspk_sir_{int(sir)}db'
                    out_wav = defect_sub_dir / f'{clip_id}.wav'
                    out_json = defect_sub_dir / f'{clip_id}.json'
                    if out_wav.exists() and not args.overwrite:
                        continue
                    mod_audio, meta = apply_secondary_speaker(
                        clean_audio, sr, row, donor_audio, donor_row, sir, rng
                    )
                    sf.write(out_wav, mod_audio, sr)
                    record = {
                        'id': clip_id,
                        'defect_type': defect,
                        'anchor_id': anchor_id,
                        'audio_path': persist_path(out_wav),
                        'twin_audio_path': row['rel_path'],
                        'split': row.get('split', 'train'),
                        'speaker': row['speaker'],
                        'sample_rate': sr,
                        **meta,
                        'transcript': row.get('text', ''),
                    }
                    write_json(out_json, record)
                    results.append(record)
                    total_generated += 1

            elif defect == 'background_music' and music_audio is not None:
                for snr in MUSIC_SNRS_DB:
                    clip_id = f'{anchor_id}__bgmusic_snr_{int(snr)}db'
                    out_wav = defect_sub_dir / f'{clip_id}.wav'
                    out_json = defect_sub_dir / f'{clip_id}.json'
                    if out_wav.exists() and not args.overwrite:
                        continue
                    mod_audio, meta = apply_background_music(
                        clean_audio, sr, music_audio, music_sr, snr, rng
                    )
                    sf.write(out_wav, mod_audio, sr)
                    record = {
                        'id': clip_id,
                        'defect_type': defect,
                        'anchor_id': anchor_id,
                        'audio_path': persist_path(out_wav),
                        'twin_audio_path': row['rel_path'],
                        'split': row.get('split', 'train'),
                        'speaker': row['speaker'],
                        'sample_rate': sr,
                        **meta,
                        'transcript': row.get('text', ''),
                    }
                    write_json(out_json, record)
                    results.append(record)
                    total_generated += 1

        if idx % 50 == 0 or idx == len(rows):
            progress('SYNTH_PROGRESS', f'Processed {idx}/{len(rows)} anchors ({total_generated} defect clips generated)')

    manifest_lines = [json.dumps(r, ensure_ascii=False) for r in results]
    out_manifest.write_text('\n'.join(manifest_lines) + '\n', encoding='utf-8')
    progress('COMPLETE', f'Synthesized {len(results)} defect clips across {len(rows)} anchors')
    print(persist_path(out_manifest))
    return 0


if __name__ == '__main__':
    sys.exit(main())
