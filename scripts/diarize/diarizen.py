"""DiariZen diarization with independent, unfiltered turn and clip export."""
from __future__ import annotations

import contextlib
import math
from pathlib import Path
import sys

# Prevent this script from shadowing the installed 'diarizen' package
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)
sys.modules.pop('diarizen', None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common.files import batch, identity, inputs, manifest_destinations, parser, positive_int, probe, request, safe_name
from _common.segments import export, manifest_complete


def main() -> int:
    p = parser(__doc__, 'diarize', 'diarizen', segments=True)
    p.add_argument('--model', default='BUT-FIT/diarizen-wavlm-large-s80-md-v2', help='HuggingFace model repository or local checkpoint directory')
    p.add_argument('--device', default='auto', help='Inference device ("auto", "cpu", "cuda", or "hip")')
    p.add_argument('--batch-size', type=positive_int, default=1, help='Inference batch size for segmentation and embedding models')
    p.add_argument('--num-speakers', type=positive_int, help='Exact number of speakers if known in advance')
    p.add_argument('--min-speakers', type=positive_int, help='Minimum speaker cluster count')
    p.add_argument('--max-speakers', type=positive_int, help='Maximum speaker cluster count')
    p.add_argument('--sample-rate', type=positive_int, help='Output target sample rate in Hz (defaults to source sample rate)')
    p.add_argument('--channels', type=int, choices=(1, 2), default=1, help='Output audio channel count (1=mono, 2=stereo)')
    p.add_argument('--min-duration-s', type=float, default=2.0, help='Minimum turn duration in seconds to keep and export')
    p.add_argument('--max-duration-s', type=float, default=15.0, help='Maximum turn duration in seconds to keep and export')
    p.add_argument('--segmentation-step', type=float, default=0.1, help='Segmentation shifting ratio step')
    p.add_argument('--binarize-onset', type=float, default=0.5, help='Binarize onset threshold')
    p.add_argument('--binarize-offset', type=float, default=0.5, help='Binarize offset threshold')
    args = p.parse_args()
    if args.min_duration_s is not None and (not math.isfinite(args.min_duration_s) or args.min_duration_s < 0):
        p.error('--min-duration-s must be finite and non-negative')
    if args.max_duration_s is not None and (not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0):
        p.error('--max-duration-s must be finite and positive')
    if args.min_duration_s is not None and args.max_duration_s is not None and args.min_duration_s > args.max_duration_s:
        p.error('--min-duration-s cannot exceed --max-duration-s')
    if not math.isfinite(args.segmentation_step) or not (0 < args.segmentation_step <= 1):
        p.error('--segmentation-step must satisfy 0 < segmentation_step <= 1')
    if not math.isfinite(args.binarize_onset) or not (0 <= args.binarize_onset <= 1):
        p.error('--binarize-onset must satisfy 0 <= binarize_onset <= 1')
    if not math.isfinite(args.binarize_offset) or not (0 <= args.binarize_offset <= 1):
        p.error('--binarize-offset must satisfy 0 <= binarize_offset <= 1')
    minimum, maximum, exact = args.min_speakers, args.max_speakers, args.num_speakers
    if ((minimum and maximum and minimum > maximum) or
        (exact and minimum and exact < minimum) or (exact and maximum and exact > maximum)):
        p.error('Inconsistent speaker count bounds')
    if args.work_dir == p.get_default('work_dir'):
        args.work_dir = args.work_dir.parent.parent / 'diarizen' / args.work_dir.name
    pairs = manifest_destinations(args)
    import os
    import tempfile
    from io import BytesIO
    import numpy as np
    from scipy.ndimage import median_filter
    import toml
    import torch
    import torchaudio
    from pyannote.audio.utils.signal import Binarize
    from pyannote.database.protocol.protocol import ProtocolFile
    from diarizen.pipelines.inference import DiariZenPipeline as _UpstreamDiariZenPipeline
    from _common.files import convert

    class DiariZenPipeline(_UpstreamDiariZenPipeline):
        def __init__(
            self,
            diarizen_hub: Path,
            embedding_model: str,
            config_parse: dict | None = None,
            rttm_out_dir: str | None = None,
            binarize_onset: float = 0.5,
            binarize_offset: float = 0.5,
        ):
            config_path = Path(diarizen_hub / 'config.toml')
            base_config = toml.load(config_path.as_posix())
            merged_config = {
                'inference': {'args': dict(base_config.get('inference', {}).get('args', {}))},
                'clustering': {'args': dict(base_config.get('clustering', {}).get('args', {}))},
            }
            if config_parse is not None:
                if 'inference' in config_parse and 'args' in config_parse['inference']:
                    merged_config['inference']['args'].update(config_parse['inference']['args'])
                if 'clustering' in config_parse and 'args' in config_parse['clustering']:
                    merged_config['clustering']['args'].update(config_parse['clustering']['args'])

            super().__init__(
                diarizen_hub=diarizen_hub,
                embedding_model=embedding_model,
                config_parse=merged_config,
                rttm_out_dir=rttm_out_dir,
            )
            self.to_annotation = Binarize(
                onset=binarize_onset,
                offset=binarize_offset,
                min_duration_on=0.0,
                min_duration_off=0.0,
            )

        @classmethod
        def from_pretrained(
            cls,
            repo_id: str,
            cache_dir: str | None = None,
            rttm_out_dir: str | None = None,
            config_parse: dict | None = None,
            binarize_onset: float = 0.5,
            binarize_offset: float = 0.5,
        ) -> DiariZenPipeline:
            from huggingface_hub import snapshot_download, hf_hub_download

            diarizen_hub = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=cache_dir is not None,
            )
            embedding_model = hf_hub_download(
                repo_id='pyannote/wespeaker-voxceleb-resnet34-LM',
                filename='pytorch_model.bin',
                cache_dir=cache_dir,
                local_files_only=cache_dir is not None,
            )
            return cls(
                diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
                embedding_model=embedding_model,
                config_parse=config_parse,
                rttm_out_dir=rttm_out_dir,
                binarize_onset=binarize_onset,
                binarize_offset=binarize_offset,
            )

        def __call__(self, in_wav, sess_name=None):
            assert isinstance(in_wav, (str, BytesIO, ProtocolFile)), (
                f'input must be either a str, BytesIO or a ProtocolFile; there was {type(in_wav)}'
            )
            in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']

            waveform, sample_rate = torchaudio.load(in_wav)
            waveform = torch.unsqueeze(waveform[0], 0)  # force to use the SDM data
            segmentations = self.get_segmentations({'waveform': waveform, 'sample_rate': sample_rate}, soft=False)

            if self.apply_median_filtering:
                segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

            binarized_segmentations = segmentations

            count = self.speaker_count(
                binarized_segmentations,
                self._segmentation.model._receptive_field,
                warm_up=(0.0, 0.0),
            )

            embeddings = self.get_embeddings(
                {'waveform': waveform, 'sample_rate': sample_rate},
                binarized_segmentations,
                exclude_overlap=self.embedding_exclude_overlap,
            )

            hard_clusters, _, _ = self.clustering(
                embeddings=embeddings,
                segmentations=binarized_segmentations,
                min_clusters=self.min_speakers,
                max_clusters=self.max_speakers,
            )

            count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)

            inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
            hard_clusters[inactive_speakers] = -2
            discrete_diarization, _ = self.reconstruct(
                segmentations,
                hard_clusters,
                count,
            )

            result = self.to_annotation(discrete_diarization)
            result.uri = sess_name

            if self.rttm_out_dir is not None:
                assert sess_name is not None
                rttm_out = os.path.join(self.rttm_out_dir, sess_name + '.rttm')
                with open(rttm_out, 'w') as f:
                    f.write(result.to_rttm())
            return result

    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    config_parse = {
        'inference': {
            'args': {
                'segmentation_step': args.segmentation_step,
                'batch_size': args.batch_size,
            }
        }
    }
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = DiariZenPipeline.from_pretrained(
            args.model,
            config_parse=config_parse,
            binarize_onset=args.binarize_onset,
            binarize_offset=args.binarize_offset,
        )
        pipeline.to(torch.device(device))
        pipeline.segmentation_batch_size = args.batch_size
        pipeline.embedding_batch_size = args.batch_size

    print(f"DiariZen segmentation_step: {pipeline.segmentation_step:g}", file=sys.stderr)
    print(f"DiariZen binarize onset: {pipeline.to_annotation.onset:.2f}", file=sys.stderr)
    print(f"DiariZen binarize offset: {pipeline.to_annotation.offset:.2f}", file=sys.stderr)

    kwargs = {key: getattr(args, key) for key in ('num_speakers', 'min_speakers', 'max_speakers') if getattr(args, key) is not None}
    def process(src, dest):
        rate = args.sample_rate or probe(src)['sample_rate']
        wanted = request(identity(src), 'diarize', {**kwargs, 'device': device, 'batch_size': args.batch_size,
                         'sample_rate': rate, 'channels': args.channels, 'checkpoint': args.model,
                         'min_duration_s': args.min_duration_s, 'max_duration_s': args.max_duration_s,
                         'segmentation_step': args.segmentation_step, 'binarize_onset': args.binarize_onset,
                         'binarize_offset': args.binarize_offset}, 'diarizen')
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
            with contextlib.redirect_stdout(sys.stderr):
                annotation = pipeline(str(normalized))
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
