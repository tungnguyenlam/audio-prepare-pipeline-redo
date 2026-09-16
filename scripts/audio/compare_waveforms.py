"""Plot aligned input, reference, and their time-domain residual as waveforms."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import batch, completed, destinations, identity, parser, positive_int, publish, request


def main() -> int:
    p = parser(__doc__, 'compare_waveforms')
    p.add_argument('-rf', '--reference-file', type=Path, required=True,
                   help='Path to reference audio file to compare against')
    p.add_argument('-sr', '--sample-rate', type=positive_int, default=16000,
                   help='Sample rate in Hz for waveform analysis (default: 16000)')
    args = p.parse_args()
    pairs = destinations(args, '', '.png')
    reference = args.reference_file.resolve()
    if not reference.is_file():
        p.error('Reference file does not exist')
    if any(reference in {dest, dest.with_suffix('.json')} for _, dest in pairs):
        p.error('Output would overwrite reference')
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    args.work_dir.mkdir(parents=True, exist_ok=True)
    params = {'sample_rate': args.sample_rate}
    reference_identity = identity(reference)
    after, _ = librosa.load(reference, sr=args.sample_rate, mono=True)
    def process(src, dest):
        metadata = request({'input': identity(src), 'reference': reference_identity}, 'compare_waveforms', params)
        if completed(dest, metadata, args.overwrite):
            return
        before, _ = librosa.load(src, sr=args.sample_rate, mono=True)
        n = min(len(before), len(after))
        if not n:
            raise ValueError('Both files must contain audio')
        panels = (before[:n], after[:n], before[:n] - after[:n])
        figure, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True)
        try:
            peak = max(float(np.max(np.abs(y))) for y in panels) or 1.0
            for axis, data, title in zip(axes, panels, (src.name, reference.name, 'Residual (input - reference)')):
                librosa.display.waveshow(data, sr=args.sample_rate, ax=axis)
                axis.set_ylim(-peak, peak)
                axis.set_title(title)
                axis.set_ylabel('Amplitude')
            axes[-1].set_xlabel('Time (s)')
            figure.tight_layout()
            with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
                staged = Path(directory) / 'plot.png'
                figure.savefig(staged, format='png', dpi=150)
                publish(staged, dest, metadata, audio=False)
        finally:
            plt.close(figure)
    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


if __name__ == '__main__':
    raise SystemExit(main())
