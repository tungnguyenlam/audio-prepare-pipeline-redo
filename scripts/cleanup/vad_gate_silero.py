"""Zero non-speech regions with Silero VAD so SFX, music, and hiss in gaps are silenced.

Speech frames at or above ``--threshold`` are kept, with ``--pad-ms`` hangover so
word edges are not clipped. This does not remove SFX that overlap active speech
and does not separate competing talkers.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.audio_utils import load_waveform, pin_device_visibility, save_waveform, speech_gate_mask  # noqa: E402
from _common.files import (  # noqa: E402
    batch, completed, convert, destinations, identity, parser, positive_int, probe,
    publish, request,
)
from _common.vad import (  # noqa: E402
    FRAME_SAMPLES, SAMPLE_RATE, ensure_vad_report, load_vad_report, select_vad_device,
)


def main() -> int:
    p = parser(__doc__, 'cleanup', 'vad_gate_silero')
    p.add_argument('-d', '--device', '--vad-device', dest='vad_device', default='auto',
                   help='Silero track: auto prefers cuda:0 then cpu (default: auto)')
    p.add_argument('--vad-report', type=Path, default=None,
                   help='Completed evaluate/silero_jit JSON for the same source bytes (default: auto cache)')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Keep frames with speech probability at or above this value (default: 0.5)')
    p.add_argument('--pad-ms', type=float, default=200.0,
                   help='Hangover padding on each kept speech region in milliseconds (default: 200)')
    p.add_argument('-sr', '--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz (default: preserve source)')
    p.add_argument('-ch', '--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout (1=mono, 2=stereo)')
    args = p.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        p.error('--threshold must be finite and between 0 and 1')
    if not math.isfinite(args.pad_ms) or args.pad_ms < 0:
        p.error('--pad-ms must be finite and >= 0')

    pin_device_visibility(args.vad_device)
    pairs = destinations(args, '_vad_gate')
    args.work_dir.mkdir(parents=True, exist_ok=True)

    def process(src: Path, dest: Path) -> None:
        info = probe(src)
        rate = args.sample_rate or info['sample_rate']
        pad_samples = int(round((args.pad_ms / 1000.0) * info['sample_rate']))
        report_path = (args.vad_report.resolve() if args.vad_report is not None
                       else ensure_vad_report(src, vad_device=args.vad_device))
        parameters = {
            'sample_rate': rate,
            'channels': args.channels,
            'vad_device': args.vad_device,
            'threshold': args.threshold,
            'pad_ms': args.pad_ms,
            'pad_samples': pad_samples,
            'vad_report': identity(report_path),
            'nonspeech_gain': 0.0,
        }
        metadata = request(identity(src), 'vad_gate', parameters, 'silero_vad_jit')
        if completed(dest, metadata, args.overwrite):
            return
        _candidates, report = load_vad_report(src, report_path, args.vad_device)
        analysis = report.get('parameters', {})
        analysis_rate = int(analysis.get('sample_rate') or SAMPLE_RATE)
        frame_samples = int(analysis.get('frame_samples') or FRAME_SAMPLES)
        selected = select_vad_device(report, args.vad_device)
        probabilities = report['devices'][selected]['probabilities']
        mask = speech_gate_mask(
            probabilities,
            analysis_sample_rate=analysis_rate,
            frame_samples=frame_samples,
            source_sample_rate=info['sample_rate'],
            source_frames=info['frames'],
            threshold=args.threshold,
            pad_samples=pad_samples,
        )
        with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
            work = Path(directory)
            wave, wave_rate = load_waveform(src)
            if wave_rate != info['sample_rate'] or wave.shape[0] != info['frames']:
                raise ValueError('Decoded geometry does not match the probed source')
            if mask.shape[0] != wave.shape[0]:
                raise ValueError('VAD mask length does not match the source')
            gated = wave * mask[:, None]
            gated_path = work / 'gated.wav'
            save_waveform(gated_path, gated, wave_rate)
            staged = work / 'output.wav'
            convert(gated_path, staged, rate, args.channels)
            publish(staged, dest, metadata)

    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


if __name__ == '__main__':
    raise SystemExit(main())
