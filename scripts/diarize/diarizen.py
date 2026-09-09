"""DiariZen diarization with independent, unfiltered turn and clip export."""
from __future__ import annotations

import contextlib
from pathlib import Path
import sys

# Prevent this script from shadowing the installed 'diarizen' package
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)
sys.modules.pop('diarizen', None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common.files import batch, identity, inputs, parser, positive_int, probe, request, safe_name
from _common.segments import export, manifest_complete


def main() -> int:
    p = parser(__doc__, 'diarize', 'diarizen', segments=True)
    p.add_argument('--model', default='BUT-FIT/diarizen-wavlm-large-s80-md-v2')
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
    model_name = 'diarizen'
    for attr in ('output_dir', 'work_dir'):
        if getattr(args, attr) == p.get_default(attr):
            setattr(args, attr, getattr(args, attr).parent.parent / model_name / getattr(args, attr).name)
    safe_parent = lambda rel: Path(*[safe_name(p) for p in rel.parent.parts]) if rel.parent.parts else Path('.')
    pairs = [(src, args.output_dir.resolve() / safe_parent(rel) / safe_name(rel.stem) / 'segments.json') for src, rel in inputs(args)]
    if len({dest for _, dest in pairs}) != len(pairs):
        p.error('Multiple inputs map to the same output directory')
    import torch
    from diarizen.pipelines.inference import DiariZenPipeline
    import tempfile
    from _common.files import convert
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = DiariZenPipeline.from_pretrained(args.model)
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
        pipeline.min_speakers = exact or minimum or pipeline.min_speakers
        pipeline.max_speakers = exact or maximum or pipeline.max_speakers
        args.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            normalized = Path(work) / 'input.wav'
            convert(src, normalized, 16000, 1)
            if probe(normalized)['frames'] == 0:
                raise ValueError('Cannot diarize empty audio')
            annotation = pipeline(str(normalized))
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
