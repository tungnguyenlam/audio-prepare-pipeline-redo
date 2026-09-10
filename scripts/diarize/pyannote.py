"""Pyannote diarization with independent, unfiltered turn and clip export."""
from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
import sys

# Prevent this script from shadowing the installed 'pyannote' package
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)
sys.modules.pop('pyannote', None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common.files import batch, identity, inputs, manifest_destinations, parser, positive_int, probe, request, safe_name
from _common.segments import export, manifest_complete


def main() -> int:
    p = parser(__doc__, 'diarize', 'pyannote_community1', segments=True)
    p.add_argument('--model', default='pyannote/speaker-diarization-community-1',
                   help='Pyannote pretrained model name or HF hub ID (default: pyannote/speaker-diarization-community-1)')
    p.add_argument('--device', default='auto',
                   help='Execution device (e.g. auto, cpu, cuda) (default: auto)')
    p.add_argument('--batch-size', type=positive_int, default=1,
                   help='Internal embedding and segmentation batch size (default: 1)')
    p.add_argument('--num-speakers', type=positive_int, default=None,
                   help='Exact known number of speakers if known in advance (default: None)')
    p.add_argument('--min-speakers', type=positive_int, default=None,
                   help='Minimum expected number of speakers (default: None)')
    p.add_argument('--max-speakers', type=positive_int, default=None,
                   help='Maximum expected number of speakers (default: None)')
    p.add_argument('--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz for exported turn clips (default: preserve source)')
    p.add_argument('--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout for clips (1=mono, 2=stereo) (default: 1)')
    p.add_argument('--min-duration-s', type=float, default=2.0, help='Minimum turn duration in seconds to keep and export (default: 2.0)')
    p.add_argument('--max-duration-s', type=float, default=15.0, help='Maximum turn duration in seconds to keep and export (default: 15.0)')
    args = p.parse_args()
    if args.min_duration_s is not None and (not math.isfinite(args.min_duration_s) or args.min_duration_s < 0):
        p.error('--min-duration-s must be finite and non-negative')
    if args.max_duration_s is not None and (not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0):
        p.error('--max-duration-s must be finite and positive')
    if args.min_duration_s is not None and args.max_duration_s is not None and args.min_duration_s > args.max_duration_s:
        p.error('--min-duration-s cannot exceed --max-duration-s')
    minimum, maximum, exact = args.min_speakers, args.max_speakers, args.num_speakers
    if ((minimum and maximum and minimum > maximum) or
        (exact and minimum and exact < minimum) or (exact and maximum and exact > maximum)):
        p.error('Inconsistent speaker count bounds')
    model_name = 'pyannote_31' if args.model == 'pyannote/speaker-diarization-3.1' else 'pyannote_community1'
    args._model = model_name
    if getattr(args, '_default_base', None) is not None:
        args._default_base = args._default_base.parent / model_name
    if args.work_dir == p.get_default('work_dir'):
        args.work_dir = args.work_dir.parent.parent / model_name / args.work_dir.name
    pairs = manifest_destinations(args)
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = Pipeline.from_pretrained(args.model, token=os.getenv('HF_TOKEN'))
        pipeline.to(torch.device(device))
        pipeline.segmentation_batch_size = args.batch_size
        pipeline.embedding_batch_size = args.batch_size
    kwargs = {key: getattr(args, key) for key in ('num_speakers', 'min_speakers', 'max_speakers') if getattr(args, key) is not None}
    def process(src, dest):
        rate = args.sample_rate or probe(src)['sample_rate']
        wanted = request(identity(src), 'diarize', {**kwargs, 'device': device, 'batch_size': args.batch_size,
                         'sample_rate': rate, 'channels': args.channels, 'checkpoint': args.model,
                         'min_duration_s': args.min_duration_s, 'max_duration_s': args.max_duration_s}, model_name)
        if manifest_complete(dest, wanted, args.overwrite):
            return
        data, source_rate = sf.read(src, dtype='float32', always_2d=True)
        if len(data) == 0:
            raise ValueError('Cannot diarize empty audio')
        output = pipeline({'waveform': torch.from_numpy(data.mean(axis=1, keepdims=True).T.copy()), 'sample_rate': source_rate}, **kwargs)
        annotation = (output.speaker_diarization if hasattr(output, 'speaker_diarization') else
                      output['speaker_diarization'] if isinstance(output, dict) else output)
        labels, turns = {}, []
        for segment, _, label in annotation.itertracks(yield_label=True):
            if label not in labels:
                labels[label] = f'spk{len(labels):02d}'
            if segment.end > segment.start:
                turns.append({'speaker_id': labels[label], 'start_s': float(segment.start), 'end_s': float(segment.end)})
        export({**wanted, 'speaker_ids': list(labels.values()), 'turns': turns}, src, dest, args.work_dir, rate, args.channels,
               args.min_duration_s, args.max_duration_s, concurrency=args.concurrency, batch_size=args.batch_size)
    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


if __name__ == '__main__':
    raise SystemExit(main())
