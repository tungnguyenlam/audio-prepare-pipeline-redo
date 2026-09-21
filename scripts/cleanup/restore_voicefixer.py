"""Aggressively restore reverb, echo, clipping, and heavy separator artifacts with VoiceFixer.

Mode 0 is the original restorer. Mode 1 adds a high-frequency preprocessor.
Mode 2 runs the network in train mode and can help on severely damaged speech
but is the most likely to color the voice. Prefer denoise/enhance first; use
this when residual degradation remains.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.audio_utils import pin_device_visibility, publish_model_audio  # noqa: E402
from _common.files import batch, destinations, parser, positive_int, probe  # noqa: E402

VOICEFIXER_SAMPLE_RATE = 44100


def main() -> int:
    p = parser(__doc__, 'cleanup', 'restore_voicefixer')
    p.add_argument('--mode', type=int, choices=(0, 1, 2), default=0,
                   help='VoiceFixer restore mode: 0 original, 1 high-frequency preprocess, 2 train-mode')
    p.add_argument('-d', '--device', default='auto',
                   help='Compute device (auto, cpu, cuda:0). Isolated venv is CPU on AMD ROCm hosts.')
    p.add_argument('-sr', '--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz (default: preserve source)')
    p.add_argument('-ch', '--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout (1=mono, 2=stereo)')
    args = p.parse_args()
    pin_device_visibility(args.device)

    import torch
    from voicefixer import VoiceFixer

    use_cuda = args.device != 'cpu' and torch.cuda.is_available()
    restorer = VoiceFixer()
    pairs = destinations(args, '_restore')
    args.work_dir.mkdir(parents=True, exist_ok=True)

    def infer(prepared: Path, model_wav: Path) -> None:
        restorer.restore(input=str(prepared), output=str(model_wav), cuda=use_cuda, mode=args.mode)
        if not model_wav.is_file():
            raise ValueError('VoiceFixer did not write an output WAV')

    def process(src: Path, dest: Path) -> None:
        rate = args.sample_rate or probe(src)['sample_rate']
        publish_model_audio(
            src, dest,
            model_sample_rate=VOICEFIXER_SAMPLE_RATE,
            output_sample_rate=rate,
            channels=args.channels,
            overwrite=args.overwrite,
            work_dir=args.work_dir,
            operation='restore',
            model='voicefixer',
            parameters={
                'sample_rate': rate,
                'channels': args.channels,
                'device': args.device,
                'mode': args.mode,
                'cuda': use_cuda,
                'model_sample_rate': VOICEFIXER_SAMPLE_RATE,
            },
            infer=infer,
        )

    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


if __name__ == '__main__':
    raise SystemExit(main())
