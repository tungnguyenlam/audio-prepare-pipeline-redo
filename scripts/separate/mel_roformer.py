"""Standalone mel_roformer inference with one session per invocation."""
from __future__ import annotations

import contextlib
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import batch, completed, convert, destinations, identity, parser, positive_int, probe, publish, request


def main() -> int:
    p = parser(__doc__, 'separate', 'mel_roformer')
    p.add_argument('--model', default='melband-roformer-kim-vocals',
                   help='Mel-Band RoFormer model checkpoint name (default: melband-roformer-kim-vocals)')
    p.add_argument('--device', default='auto',
                   help='Compute device for model inference (e.g. auto, cpu, cuda) (default: auto)')
    p.add_argument('--backend', default=None,
                   help='Inference backend engine (optional; default: None)')
    p.add_argument('--stem', default='vocals',
                   help='Target stem to separate and export (default: vocals)')
    p.add_argument('--model-sample-rate', type=positive_int, default=44100,
                   help='Internal sample rate expected by model checkpoint (default: 44100)')
    p.add_argument('--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz (default: preserve source)')
    p.add_argument('--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout (1=mono, 2=stereo) (default: 1)')
    args = p.parse_args()
    pairs = destinations(args, '_mel_roformer')
    from mel_band_roformer import MelBandRoformerSession
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        session = MelBandRoformerSession(model_name=args.model, device=None if args.device == 'auto' else args.device, backend=args.backend)
        session.load()
    def process(src, dest):
        rate = args.sample_rate or probe(src)['sample_rate']
        parameters = {'sample_rate': rate, 'channels': args.channels, 'stem': args.stem,
                      'device': args.device, 'backend': args.backend, 'model_sample_rate': args.model_sample_rate}
        metadata = request(identity(src), 'separate', parameters, args.model)
        if completed(dest, metadata, args.overwrite):
            return
        with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
            work = Path(directory)
            input_dir, output_dir = work / 'input', work / 'stems'
            output_dir.mkdir()
            convert(src, input_dir / 'source.wav', args.model_sample_rate, 2, floating=True)
            manifest = session.infer(str(input_dir), store_dir=str(output_dir))
            outputs = ([(item.output_id, Path(item.output_path)) for item in manifest.outputs]
                       if hasattr(manifest, 'outputs') else
                       [(item['output_id'], Path(item['output_path'])) for item in manifest])
            selected = next((path for stem, path in outputs if stem == args.stem), None)
            if selected is None:
                raise ValueError(f'Requested stem {args.stem!r} unavailable; found {[s for s, _ in outputs]}')
            staged = work / 'output.wav'
            convert(selected, staged, rate, args.channels)
            publish(staged, dest, metadata)
    try:
        return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            session.close()


if __name__ == '__main__':
    raise SystemExit(main())
