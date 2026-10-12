#!/usr/bin/env python3
"""Denoise speech with DeepFilterNet.

Best for hiss, white/stationary noise, fan/HVAC and residual broadband noise.
This is NOT a music separator and NOT a multi-speaker separator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF

from audio_utils import iter_audio, resolve_output


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input', type=Path, required=True, help='Input audio file or directory')
    p.add_argument('-o', '--output', type=Path, required=True, help='Output file or directory')
    p.add_argument('--model', default=None, help='DeepFilterNet model directory/tar; default uses packaged pretrained model')
    p.add_argument('--post-filter', action='store_true', help='Enable DeepFilterNet post-filter')
    p.add_argument('--atten-lim-db', type=float, default=None,
                   help='Limit maximum noise attenuation in dB. Omit for unrestricted model attenuation.')
    p.add_argument('--compensate-delay', action='store_true', help='Compensate STFT/model look-ahead delay')
    p.add_argument('--suffix', default='_df', help='Suffix when output is a directory')
    args = p.parse_args()

    from df.enhance import enhance, init_df, load_audio, save_audio

    model, df_state, _ = init_df(
        args.model,
        post_filter=args.post_filter,
        log_level='INFO',
        config_allow_defaults=True,
    )
    model_sr = int(df_state.sr())
    files = iter_audio(args.input)
    if not files:
        raise SystemExit('No audio files found')

    for idx, src in enumerate(files, 1):
        dest = resolve_output(src, args.input, args.output, args.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        original_sr = int(sf.info(str(src)).samplerate)

        audio, _ = load_audio(str(src), sr=model_sr)
        kwargs = {'pad': args.compensate_delay}
        if args.atten_lim_db is not None:
            kwargs['atten_lim_db'] = args.atten_lim_db
        with torch.inference_mode():
            enhanced = enhance(model, df_state, audio, **kwargs)

        if original_sr != model_sr:
            enhanced = AF.resample(enhanced.cpu(), model_sr, original_sr)
            out_sr = original_sr
        else:
            enhanced = enhanced.cpu()
            out_sr = model_sr
        save_audio(str(dest), enhanced, out_sr)
        print(f'[{idx}/{len(files)}] {src.name} -> {dest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
