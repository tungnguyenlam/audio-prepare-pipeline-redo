"""Evaluate existing separation outputs against references; never runs models."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, identity, positive_int, progress, read_json, write_json
import librosa
import numpy as np
import soundfile as sf

def si_sdr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    length = min(len(estimate), len(reference))
    if length == 0:
        return 0.0
    est = np.asarray(estimate[:length], dtype=np.float64)
    ref = np.asarray(reference[:length], dtype=np.float64)

    est = est - np.mean(est)
    ref = ref - np.mean(ref)

    ref_energy = np.dot(ref, ref) + 1e-12
    target = (np.dot(est, ref) / ref_energy) * ref
    noise = est - target

    target_energy = np.dot(target, target) + 1e-12
    noise_energy = np.dot(noise, noise) + 1e-12
    return float(10.0 * np.log10(target_energy / noise_energy))


def load_mono_waveform(path: str | Path, target_sr: int = 48000) -> np.ndarray:
    """Load audio file as mono float64 array."""
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    mono = np.mean(data, axis=1)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    return mono


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-file', type=Path, required=True)
    p.add_argument('--reference-file', type=Path, required=True)
    p.add_argument('--mixture-file', type=Path)
    p.add_argument('--sample-rate', type=positive_int, default=48000)
    p.add_argument('--output-file', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    paths = [args.input_file, args.reference_file] + ([args.mixture_file] if args.mixture_file else [])
    dest = args.output_file.resolve()
    if dest in {path.resolve() for path in paths}:
        p.error('Cannot overwrite an input')
    metadata = {'operation': 'separation_metrics', 'source': [identity(path) for path in paths],
                'parameters': {'sample_rate': args.sample_rate}}
    if dest.exists() and not args.overwrite:
        old = read_json(dest)
        if all(old.get(k) == v for k, v in metadata.items()) and 'metrics' in old:
            print(dest)
            return 0
        p.error('Conflicting output; use --overwrite')
    progress('EVAL_SEP', f'Evaluating SI-SDR: {args.input_file.name} vs {args.reference_file.name}')
    estimate, reference = [load_mono_waveform(path, args.sample_rate) for path in paths[:2]]
    if not len(estimate) or not len(reference):
        p.error('Inputs must contain audio')
    metrics = {'si_sdr_db': si_sdr_db(estimate, reference), 'scored_samples': min(len(estimate), len(reference))}
    if args.mixture_file:
        mixture = load_mono_waveform(args.mixture_file, args.sample_rate)
        metrics['mixture_si_sdr_db'] = si_sdr_db(mixture, reference)
        metrics['si_sdri_db'] = metrics['si_sdr_db'] - metrics['mixture_si_sdr_db']
    write_json(dest, {**metadata, 'metrics': metrics})
    progress('EVAL_SEP_DONE', f'SI-SDR: {metrics["si_sdr_db"]:.2f} dB -> {dest.name}')
    print(dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
