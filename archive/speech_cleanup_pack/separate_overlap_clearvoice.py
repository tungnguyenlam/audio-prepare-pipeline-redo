#!/usr/bin/env python3
"""Separate two simultaneously speaking people with ClearVoice MossFormer2_SS_16K.

Run this only on clips/intervals known to contain overlapping speakers. Output is
speaker stream 1 and speaker stream 2; speaker identity is permutation-ambiguous and
must be mapped back to diarization/enrollment embeddings downstream.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from audio_utils import iter_audio, read_mono, resample_audio, safe_write


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input', type=Path, required=True)
    p.add_argument('-o', '--output-dir', type=Path, required=True)
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    args = p.parse_args()

    dev = args.device.lower()
    if dev == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif dev.startswith('cuda:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = dev.split(':', 1)[1]

    from clearvoice import ClearVoice

    separator = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
    files = iter_audio(args.input)
    if not files:
        raise SystemExit('No audio files found')

    for idx, src in enumerate(files, 1):
        x, sr = read_mono(src)
        x16 = resample_audio(x, sr, 16000).reshape(1, -1).astype(np.float32)
        out = separator(x16, False)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        out = np.asarray(out)
        # Official API: [speaker, batch, time]
        if out.ndim != 3 or out.shape[0] < 2:
            raise RuntimeError(f'Unexpected separation shape for {src}: {out.shape}')
        rel_parent = src.parent.relative_to(args.input) if args.input.is_dir() else Path('.')
        base = args.output_dir / rel_parent
        safe_write(base / f'{src.stem}_spk1.wav', out[0, 0].astype(np.float32), 16000)
        safe_write(base / f'{src.stem}_spk2.wav', out[1, 0].astype(np.float32), 16000)
        print(f'[{idx}/{len(files)}] {src.name} -> 2 separated speaker streams')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
