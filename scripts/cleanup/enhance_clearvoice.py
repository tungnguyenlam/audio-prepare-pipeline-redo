"""Enhance residual music bleed, light SFX, and separator artifacts with ClearVoice.

Default model is MossFormer2_SE_48K (full-band). FRCRN_SE_16K and
MossFormerGAN_SE_16K are the 16 kHz alternatives. This is speech enhancement,
not overlap separation; use separate_overlap_clearvoice for competing talkers.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.audio_utils import pin_device_visibility, publish_model_audio, working_directory  # noqa: E402
from _common.files import batch, destinations, parser, positive_int, probe  # noqa: E402

ENHANCE_MODELS = {
    'MossFormer2_SE_48K': 48000,
    'FRCRN_SE_16K': 16000,
    'MossFormerGAN_SE_16K': 16000,
}


def main() -> int:
    p = parser(__doc__, 'cleanup', 'enhance_clearvoice')
    p.add_argument('-m', '--model', choices=tuple(ENHANCE_MODELS), default='MossFormer2_SE_48K',
                   help='ClearVoice speech-enhancement checkpoint')
    p.add_argument('-d', '--device', default='auto',
                   help='Compute device (auto, cpu, cuda:0). Isolated venv is CPU on AMD ROCm hosts.')
    p.add_argument('-sr', '--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz (default: preserve source)')
    p.add_argument('-ch', '--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout (1=mono, 2=stereo)')
    args = p.parse_args()
    if args.concurrency > 1:
        p.error('ClearVoice enhancement keeps one session in-process; use --concurrency 1')
    pin_device_visibility(args.device)

    from clearvoice import ClearVoice

    model_sr = ENHANCE_MODELS[args.model]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with working_directory(args.work_dir / 'clearvoice_runtime'):
        session = ClearVoice(task='speech_enhancement', model_names=[args.model])
    pairs = destinations(args, '_enhance')

    def infer(prepared: Path, model_wav: Path) -> None:
        with working_directory(prepared.parent):
            output = session(input_path=str(prepared), online_write=False)
            session.write(output, output_path=str(model_wav))
        if not model_wav.is_file():
            raise ValueError('ClearVoice enhancement did not write an output WAV')

    def process(src: Path, dest: Path) -> None:
        rate = args.sample_rate or probe(src)['sample_rate']
        publish_model_audio(
            src, dest,
            model_sample_rate=model_sr,
            output_sample_rate=rate,
            channels=args.channels,
            overwrite=args.overwrite,
            work_dir=args.work_dir,
            operation='enhance',
            model=args.model,
            parameters={
                'sample_rate': rate,
                'channels': args.channels,
                'device': args.device,
                'model_sample_rate': model_sr,
                'task': 'speech_enhancement',
            },
            infer=infer,
        )

    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


if __name__ == '__main__':
    raise SystemExit(main())
