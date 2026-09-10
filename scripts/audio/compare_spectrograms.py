"""Plot aligned input, reference, and their time-domain residual as spectrograms."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import batch, completed, destinations, identity, parser, positive_int, publish, request


def main() -> int:
    p = parser(__doc__, 'compare_spectrograms')
    p.add_argument('--reference-file', type=Path, required=True,
                   help='Path to reference audio file to compare against')
    p.add_argument('--sample-rate', type=positive_int, default=16000,
                   help='Sample rate in Hz for spectrogram analysis (default: 16000)')
    p.add_argument('--n-mels', type=positive_int, default=128,
                   help='Number of Mel frequency filter banks (default: 128)')
    p.add_argument('--hop-length', type=positive_int, default=512,
                   help='Hop length in audio samples between STFT frames (default: 512)')
    p.add_argument('--fmax', type=float, default=None,
                   help='Highest frequency in Hz displayed on mel scale (default: Nyquist)')
    p.add_argument('--top-db', type=float, default=80.0,
                   help='Threshold decibel range below peak to display (default: 80.0)')
    p.add_argument('--ref', type=float, default=1.0,
                   help='Reference power level for decibel conversion (default: 1.0)')
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
    if not math.isfinite(args.top_db) or args.top_db <= 0 or not math.isfinite(args.ref) or args.ref <= 0:
        p.error('top-db and ref must be finite and positive')
    if args.fmax is not None and (not math.isfinite(args.fmax) or not 0 < args.fmax <= args.sample_rate / 2):
        p.error('fmax must lie between zero and the Nyquist frequency')
    params.update(n_mels=args.n_mels, hop_length=args.hop_length, fmax=args.fmax, top_db=args.top_db, ref=args.ref)
    reference_identity = identity(reference)
    after, _ = librosa.load(reference, sr=args.sample_rate, mono=True)
    def process(src, dest):
        metadata = request({'input': identity(src), 'reference': reference_identity}, 'compare_spectrograms', params)
        if completed(dest, metadata, args.overwrite):
            return
        before, _ = librosa.load(src, sr=args.sample_rate, mono=True)
        n = min(len(before), len(after))
        if not n:
            raise ValueError('Both files must contain audio')
        panels = (before[:n], after[:n], before[:n] - after[:n])
        figure, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True)
        try:
            spectra = [librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=args.sample_rate,
                       n_mels=args.n_mels, hop_length=args.hop_length, fmax=args.fmax), ref=args.ref, top_db=args.top_db) for y in panels]
            maximum = max(float(spec.max()) for spec in spectra)
            for axis, spectrum, title in zip(axes, spectra, (src.name, reference.name, 'Residual (input - reference)')):
                displayed = librosa.display.specshow(spectrum, sr=args.sample_rate, hop_length=args.hop_length,
                            x_axis='time', y_axis='mel', fmax=args.fmax, ax=axis, vmin=maximum-args.top_db, vmax=maximum)
                axis.set_title(title)
                figure.colorbar(displayed, ax=axis, format='%+2.0f dB')
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
