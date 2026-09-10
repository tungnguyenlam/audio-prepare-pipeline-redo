"""Cut a time interval from each input without overwriting the source."""
from __future__ import annotations

from pathlib import Path
import math
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import batch, completed, convert, destinations, identity, parser, positive_int, probe, publish, request


def main() -> int:
    p = parser(__doc__, 'cut')
    p.add_argument('--start', type=float, required=True, help='Start seconds relative to input')
    p.add_argument('--end', type=float, required=True, help='End seconds relative to input')
    p.add_argument('--sample-rate', type=positive_int, default=None,
                   help='Target audio sample rate in Hz (default: preserve source)')
    p.add_argument('--channels', type=int, choices=(1, 2), default=1,
                   help='Target audio channel layout (1=mono, 2=stereo) (default: 1)')
    args = p.parse_args()
    if not math.isfinite(args.start) or not math.isfinite(args.end) or not 0 <= args.start < args.end:
        p.error('Require finite 0 <= start < end')
    pairs = destinations(args, '_cut')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    def process(src, dest):
        info = probe(src)
        if args.end > info['duration_s']:
            raise ValueError('--end exceeds input duration')
        rate = args.sample_rate or info['sample_rate']
        metadata = request(identity(src), 'cut', {'sample_rate': rate, 'channels': args.channels, 'start_s': args.start, 'end_s': args.end})
        if completed(dest, metadata, args.overwrite):
            return
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            staged = Path(work) / 'output.wav'
            convert(src, staged, rate, args.channels, start=args.start, end=args.end)
            publish(staged, dest, metadata)
    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


if __name__ == '__main__':
    raise SystemExit(main())
