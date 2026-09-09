"""Pyannote diarization with independent, unfiltered turn and clip export."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import batch, identity, inputs, parser, positive_int, probe, request
from _common.segments import export, manifest_complete


def main() -> int:
    p = parser(__doc__, 'diarize', 'pyannote_community1', segments=True)
    p.add_argument('--model', default='pyannote/speaker-diarization-community-1')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=positive_int, default=1)
    p.add_argument('--num-speakers', type=positive_int)
    p.add_argument('--min-speakers', type=positive_int)
    p.add_argument('--max-speakers', type=positive_int)
    p.add_argument('--sample-rate', type=positive_int)
    p.add_argument('--channels', type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    minimum, maximum, exact = args.min_speakers, args.max_speakers, args.num_speakers
    if ((minimum and maximum and minimum > maximum) or
        (exact and minimum and exact < minimum) or (exact and maximum and exact > maximum)):
        p.error('Inconsistent speaker count bounds')
    model_name = 'pyannote_31' if args.model == 'pyannote/speaker-diarization-3.1' else 'pyannote_community1'
    for attr in ('output_dir', 'work_dir'):
        if getattr(args, attr) == p.get_default(attr):
            setattr(args, attr, getattr(args, attr).parent.parent / model_name / getattr(args, attr).name)
    pairs = [(src, args.output_dir.resolve() / rel.parent / rel.stem / 'segments.json') for src, rel in inputs(args)]
    if len({dest for _, dest in pairs}) != len(pairs):
        p.error('Multiple inputs map to the same output directory')
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
                         'sample_rate': rate, 'channels': args.channels, 'checkpoint': args.model}, model_name)
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
        export({**wanted, 'speaker_ids': list(labels.values()), 'turns': turns}, src, dest, args.work_dir, rate, args.channels)
    return batch(pairs, process)


if __name__ == '__main__':
    raise SystemExit(main())
