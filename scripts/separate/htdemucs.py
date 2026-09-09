"""Separate a selected Demucs stem, loading the checkpoint once per invocation."""
from __future__ import annotations

import contextlib
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import batch, completed, convert, destinations, identity, parser, positive_int, probe, publish, request


def main() -> int:
    p = parser(__doc__, 'separate', 'htdemucs')
    p.add_argument('--model', choices=('htdemucs', 'htdemucs_ft'), default='htdemucs')
    p.add_argument('--device', default='cpu')
    p.add_argument('--stem', default='vocals', choices=('vocals', 'instrumental', 'drums', 'bass', 'other'))
    p.add_argument('--sample-rate', type=positive_int)
    p.add_argument('--channels', type=int, choices=(1, 2), default=1)
    p.add_argument('--shifts', type=int, default=1)
    p.add_argument('--overlap', type=float, default=0.25)
    p.add_argument('--segment', type=float)
    args = p.parse_args()
    if args.shifts < 0 or not 0 <= args.overlap < 1 or (args.segment is not None and not 0 < args.segment <= 7.8):
        p.error('Require shifts >= 0, 0 <= overlap < 1, and 0 < segment <= 7.8')
    # Match the selected FT model for defaults without changing user paths.
    for attr in ('output_dir', 'work_dir'):
        if getattr(args, attr) == p.get_default(attr):
            setattr(args, attr, getattr(args, attr).parent.parent / args.model / getattr(args, attr).name)
    pairs = destinations(args, '_' + args.model)
    import torch
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    with contextlib.redirect_stdout(sys.stderr):
        model = get_model(args.model).to(args.device).eval()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    def process(src, dest):
        rate = args.sample_rate or probe(src)['sample_rate']
        parameters = {'sample_rate': rate, 'channels': args.channels, 'stem': args.stem, 'device': args.device,
                      'shifts': args.shifts, 'overlap': args.overlap, 'segment': args.segment}
        metadata = request(identity(src), 'separate', parameters, args.model)
        if completed(dest, metadata, args.overwrite):
            return
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            work = Path(work)
            prepared = work / 'input.wav'
            convert(src, prepared, model.samplerate, model.audio_channels, floating=True)
            data, _ = sf.read(prepared, dtype='float32', always_2d=True)
            audio = torch.from_numpy(data.T.copy())
            reference = audio.mean(0)
            mean, std = reference.mean(), reference.std()
            if std <= 1e-8:
                raise ValueError('Input has no usable audio variation')
            normalized = (audio - mean) / std
            with torch.inference_mode():
                separated = apply_model(model, normalized[None], device=args.device, shifts=args.shifts,
                                        overlap=args.overlap, split=True, progress=True, segment=args.segment)[0].cpu()
            separated = separated * std + mean
            if args.stem == 'instrumental':
                selected = separated[[i for i, name in enumerate(model.sources) if name != 'vocals']].sum(0)
            else:
                selected = separated[model.sources.index(args.stem)]
            stem = work / 'stem.wav'
            sf.write(stem, selected.T.numpy(), model.samplerate, subtype='FLOAT')
            output = work / 'output.wav'
            convert(stem, output, rate, args.channels)
            publish(output, dest, metadata)
    return batch(pairs, process)


if __name__ == '__main__':
    raise SystemExit(main())
