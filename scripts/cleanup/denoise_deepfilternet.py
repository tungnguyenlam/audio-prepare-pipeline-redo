"""Reduce hiss, HVAC, and broadband noise with DeepFilterNet.

This is a conservative speech denoiser, not source separation. Mel-RoFormer
already removes some accompaniment; use a modest ``--atten-lim-db`` (default 12)
so residual noise is limited without over-suppressing the voice. Unlimited
attenuation (``--no-atten-lim``) and ``--post-filter`` are more aggressive and
can color the timbre.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.audio_utils import (  # noqa: E402
    as_sample_major, pin_device_visibility, publish_model_audio, save_waveform,
)
from _common.files import batch, destinations, parser, positive_int, probe  # noqa: E402


def main() -> int:
    p = parser(__doc__, 'cleanup', 'denoise_deepfilternet')
    p.add_argument('-m', '--model', default='DeepFilterNet3',
                   help='Pretrained DeepFilterNet name (DeepFilterNet, DeepFilterNet2, DeepFilterNet3)')
    p.add_argument('-d', '--device', default='auto',
                   help='Compute device (auto, cpu, cuda:0). Isolated venv is CPU on AMD ROCm hosts.')
    p.add_argument('--atten-lim-db', type=float, default=12.0,
                   help='Maximum noise attenuation in dB; leftover noise is mixed back (default: 12)')
    p.add_argument('--no-atten-lim', action='store_true',
                   help='Do not mix residual noise back (more aggressive, can distort the voice)')
    p.add_argument('--post-filter', action='store_true',
                   help='Enable DeepFilterNet post-filter (over-attenuates very noisy sections)')
    p.add_argument('-sr', '--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz (default: preserve source)')
    p.add_argument('-ch', '--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout (1=mono, 2=stereo)')
    args = p.parse_args()
    if args.no_atten_lim:
        atten_lim_db = None
    elif not math.isfinite(args.atten_lim_db) or args.atten_lim_db <= 0:
        p.error('--atten-lim-db must be a positive dB limit, or pass --no-atten-lim')
    else:
        atten_lim_db = float(args.atten_lim_db)

    pin_device_visibility(args.device)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.work_dir / 'deepfilternet.log'

    import torch
    from df.enhance import enhance as df_enhance, init_df

    init_kwargs = dict(
        model_base_dir=args.model,
        post_filter=args.post_filter,
        log_level='ERROR',
        log_file=str(log_path),
    )
    try:
        with torch.inference_mode():
            loaded = init_df(**init_kwargs)
    except TypeError:
        init_kwargs.pop('log_file', None)
        with torch.inference_mode():
            loaded = init_df(**init_kwargs)
    model, df_state = loaded[0], loaded[1]
    model_sr = df_state.sr() if callable(getattr(df_state, 'sr', None)) else int(df_state.sr)
    pairs = destinations(args, '_denoise')
    args.work_dir.mkdir(parents=True, exist_ok=True)

    def infer(prepared: Path, model_wav: Path) -> None:
        import soundfile as sf
        audio, _rate = sf.read(str(prepared), dtype='float32', always_2d=True)
        tensor = torch.from_numpy(np_as_channel_major(audio))
        with torch.inference_mode():
            enhanced = df_enhance(model, df_state, tensor, atten_lim_db=atten_lim_db)
        if hasattr(enhanced, 'detach'):
            enhanced = enhanced.detach().cpu().numpy()
        save_waveform(model_wav, as_sample_major(enhanced), model_sr)

    def process(src: Path, dest: Path) -> None:
        rate = args.sample_rate or probe(src)['sample_rate']
        publish_model_audio(
            src, dest,
            model_sample_rate=model_sr,
            output_sample_rate=rate,
            channels=args.channels,
            overwrite=args.overwrite,
            work_dir=args.work_dir,
            operation='denoise',
            model=args.model,
            parameters={
                'sample_rate': rate,
                'channels': args.channels,
                'device': args.device,
                'atten_lim_db': atten_lim_db,
                'post_filter': bool(args.post_filter),
                'model_sample_rate': model_sr,
            },
            infer=infer,
        )

    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


def np_as_channel_major(audio):
    """Return float32 ``[C, T]`` contiguous array for DeepFilterNet."""
    import numpy as np
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim != 2:
        raise ValueError(f'Expected 2-D audio, got shape {wave.shape}')
    if wave.shape[1] <= 8 and wave.shape[0] > wave.shape[1]:
        return np.ascontiguousarray(wave.T)
    return np.ascontiguousarray(wave)


if __name__ == '__main__':
    raise SystemExit(main())
