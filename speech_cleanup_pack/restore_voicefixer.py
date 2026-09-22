#!/usr/bin/env python3
"""Aggressive speech restoration with VoiceFixer.

Use only for difficult clips (strong reverb/echo, clipping, severe degradation,
separator artifacts). Neural-vocoder restoration can alter speaker timbre, so this
should normally be a post-diarization optional stage, not a default whole-corpus step.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from audio_utils import iter_audio, read_mono, resample_audio, resolve_output, safe_write


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input', type=Path, required=True)
    p.add_argument('-o', '--output', type=Path, required=True)
    p.add_argument('--mode', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--preserve-sr', action='store_true')
    p.add_argument('--suffix', default='_vf')
    args = p.parse_args()

    from voicefixer import VoiceFixer

    restorer = VoiceFixer()
    files = iter_audio(args.input)
    if not files:
        raise SystemExit('No audio files found')

    for idx, src in enumerate(files, 1):
        dest = resolve_output(src, args.input, args.output, args.suffix)
        _, original_sr = read_mono(src)
        with tempfile.TemporaryDirectory(prefix='voicefixer-') as td:
            tmp = Path(td) / 'restored.wav'
            restorer.restore(input=str(src), output=str(tmp), cuda=args.cuda, mode=args.mode)
            y, restored_sr = read_mono(tmp)
            out_sr = restored_sr
            if args.preserve_sr and restored_sr != original_sr:
                y = resample_audio(y, restored_sr, original_sr)
                out_sr = original_sr
            safe_write(dest, y, out_sr)
        print(f'[{idx}/{len(files)}] {src.name} -> {dest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
