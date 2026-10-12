#!/usr/bin/env python3
"""Speech enhancement with ClearVoice.

Recommended models:
- MossFormer2_SE_48K: high-quality full-band enhancement.
- FRCRN_SE_16K: robust 16 kHz enhancement.
- MossFormerGAN_SE_16K: 16 kHz enhancement with GAN objective.

Useful for residual noise/SFX-like contamination and general speech enhancement.
It does not guarantee removal of another simultaneously speaking person.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from audio_utils import iter_audio, read_mono, resample_audio, resolve_output, safe_write

MODEL_SR = {
    'MossFormer2_SE_48K': 48000,
    'FRCRN_SE_16K': 16000,
    'MossFormerGAN_SE_16K': 16000,
}


def _pick_waveform(output) -> np.ndarray:
    if isinstance(output, dict):
        if len(output) != 1:
            raise RuntimeError(f'Expected one model output, got keys={list(output)}')
        output = next(iter(output.values()))
    x = np.asarray(output)
    # Single enhancement model normally returns [batch, time].
    while x.ndim > 1:
        x = x[0]
    return np.asarray(x, dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input', type=Path, required=True)
    p.add_argument('-o', '--output', type=Path, required=True)
    p.add_argument('-m', '--model', choices=tuple(MODEL_SR), default='MossFormer2_SE_48K')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    p.add_argument('--preserve-sr', action='store_true',
                   help='Resample enhanced waveform back to original sample rate')
    p.add_argument('--suffix', default='_cv')
    args = p.parse_args()

    dev = args.device.lower()
    if dev == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif dev.startswith('cuda:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = dev.split(':', 1)[1]

    from clearvoice import ClearVoice

    target_sr = MODEL_SR[args.model]
    processor = ClearVoice(task='speech_enhancement', model_names=[args.model])
    files = iter_audio(args.input)
    if not files:
        raise SystemExit('No audio files found')

    for idx, src in enumerate(files, 1):
        dest = resolve_output(src, args.input, args.output, args.suffix)
        x, sr = read_mono(src)
        model_x = resample_audio(x, sr, target_sr)
        model_x = model_x.reshape(1, -1).astype(np.float32)
        enhanced = _pick_waveform(processor(model_x, False))
        out_sr = target_sr
        if args.preserve_sr and sr != target_sr:
            enhanced = resample_audio(enhanced, target_sr, sr)
            out_sr = sr
        safe_write(dest, enhanced, out_sr)
        print(f'[{idx}/{len(files)}] {src.name} -> {dest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
