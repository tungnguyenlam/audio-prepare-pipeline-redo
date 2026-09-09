"""Convert audio files to WAV with explicit metadata."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import batch, completed, convert, destinations, identity, parser, positive_int, publish, request


def main() -> int:
    p = parser(__doc__, 'convert')
    p.add_argument('--sample-rate', type=positive_int, default=44100)
    p.add_argument('--channels', type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    pairs = destinations(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    def process(src, dest):
        metadata = request(identity(src), 'convert', {'sample_rate': args.sample_rate, 'channels': args.channels})
        if completed(dest, metadata, args.overwrite):
            return
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            staged = Path(work) / 'output.wav'
            convert(src, staged, args.sample_rate, args.channels)
            publish(staged, dest, metadata)
    return batch(pairs, process)


if __name__ == '__main__':
    raise SystemExit(main())
