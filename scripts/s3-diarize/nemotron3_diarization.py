"""Nemotron 3 eight-speaker diarization with native cached chunk inference."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import sys
import tempfile

# Adjacent pyannote.py must not shadow third-party packages imported by NeMo.
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common.files import (ROOT, batch, convert, digest, identity,
                           manifest_destinations, parser, positive_int, probe,
                           read_json, request, write_json)
from _common.merge import add_diarization_merge_arguments, merge_parameters
from _common.segments import (add_long_segment_arguments, ensure_family_plots,
                              ensure_plots, export, long_segment_parameters,
                              manifest_complete, resolve_vad_report,
                              validate_long_segment_arguments)

DEFAULT_MODEL_ID = 'nvidia/Nemotron-3-Diarization'
DEFAULT_REVISION = 'a435e9867d79e789e90053f9b6d6834053af564a'
MODEL_FILENAME = 'Nemotron-3-Diarization.nemo'
# NeMo streaming parameters are measured in 80 ms encoder frames.
OFFLINE_CONFIG = dict(spkcache_len=264, fifo_len=40, chunk_len=340,
                      chunk_right_context=40, spkcache_update_period=300)


def load_model(args):
    """Restore the pinned checkpoint and configure native long-file inference."""
    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel

    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif device == 'hip' or device.startswith('hip:'):
        device = device.replace('hip', 'cuda', 1)
        if not torch.version.hip:
            raise ValueError('--device hip requires a ROCm PyTorch build')
    target = torch.device(device)
    if target.type not in {'cpu', 'cuda'}:
        raise ValueError('Supported devices: auto, cpu, cuda[:index], hip[:index]')
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU is unavailable; check the environment or use --device cpu')
    checkpoint = args.checkpoint_path
    if checkpoint is None:
        checkpoint = hf_hub_download(
            repo_id=args.model_id, filename=MODEL_FILENAME, revision=args.revision,
            cache_dir=ROOT / '.data/models/nemotron3', token=os.environ.get('HF_TOKEN'))
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(checkpoint), map_location=torch.device('cpu'), strict=True)
    if not model.streaming_mode or model._cfg.max_num_of_spks != 8 or not model.high_resolution:
        raise ValueError('Expected a streaming, high-resolution eight-speaker Nemotron 3 checkpoint')
    for key, value in OFFLINE_CONFIG.items():
        setattr(model.sortformer_modules, key, value)
    model._check_streaming_parameters()
    return model.to(target).eval()


def generate(model, source: Path, work_dir: Path):
    """Return native segment strings and raw probabilities without task parsing."""
    import numpy as np
    import soundfile as sf
    import torch

    with tempfile.TemporaryDirectory(prefix='nemotron3-', dir=work_dir) as temp:
        normalized = Path(temp) / 'audio_16khz_mono.wav'
        convert(source, normalized, 16000, 1, floating=True)
        waveform, rate = sf.read(normalized, dtype='float32')
        if not waveform.size or not np.isfinite(waveform).all():
            raise ValueError('Input must contain finite, nonempty audio')
        # Array input avoids torchaudio/torchcodec file loaders on ROCm.
        with torch.inference_mode():
            segments, probabilities = model.diarize(
                audio=[waveform], sample_rate=rate, batch_size=1,
                include_tensor_outputs=True, num_workers=0)
    return segments[0], probabilities[0].squeeze(0).detach().float().cpu().numpy()


def parse_segments(segments: list[str]) -> list[dict]:
    """Validate NeMo's start/end/speaker lines separately from generation."""
    turns = []
    for segment in segments:
        start_text, end_text, speaker = segment.split()
        start, end = float(start_text), float(end_text)
        match = re.fullmatch(r'speaker_([0-7])', speaker)
        if not match or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError(f'Invalid native diarization segment: {segment!r}')
        turns.append(dict(speaker_id=f'spk_{int(match[1]):02d}', start_s=start, end_s=end))
    return turns


def main() -> int:
    p = parser(__doc__, 's3-diarize', 'nemotron3', segments=True)
    p.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    p.add_argument('--revision', default=DEFAULT_REVISION, help='Pinned Hugging Face model revision')
    p.add_argument('--checkpoint-path', type=Path, help='Local Nemotron 3 .nemo checkpoint')
    p.add_argument('-d', '--device', default='auto', help='auto, cpu, cuda[:index], or hip[:index]')
    p.add_argument('-min', '--min-duration-s', type=float, default=1.5)
    p.add_argument('-max', '--max-duration-s', type=float, default=15.0)
    p.add_argument('-sr', '--sample-rate', type=positive_int, help='Clip sample rate; defaults to source rate')
    p.add_argument('-ch', '--channels', type=int, choices=(1, 2), default=1)
    add_long_segment_arguments(p)
    add_diarization_merge_arguments(p)
    args = p.parse_args()
    validate_long_segment_arguments(args, p)
    if args.concurrency != 1:
        p.error('--concurrency must be 1: one cached NeMo model processes files sequentially')
    if not math.isfinite(args.min_duration_s) or args.min_duration_s < 0:
        p.error('--min-duration-s must be finite and non-negative')
    if not math.isfinite(args.max_duration_s) or args.max_duration_s <= 0:
        p.error('--max-duration-s must be finite and positive')
    if args.min_duration_s > args.max_duration_s:
        p.error('--min-duration-s cannot exceed --max-duration-s')
    if args.checkpoint_path is not None and not args.checkpoint_path.is_file():
        p.error('--checkpoint-path must name an existing .nemo checkpoint')
    merge_options = {**merge_parameters(args, p), 'adjust_mean': args.adjust_mean,
                     'dynamic_merge': args.adjust_mean}
    pairs = manifest_destinations(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    parameters = dict(model_id=args.model_id, revision=args.revision, device=args.device,
                      streaming_config=OFFLINE_CONFIG, inference_sample_rate=16000)
    if args.checkpoint_path:
        parameters['checkpoint_path'] = identity(args.checkpoint_path)
    model = None

    def process(src, dest):
        nonlocal model
        import numpy as np

        rate = args.sample_rate or probe(src)['sample_rate']
        vad_report = resolve_vad_report(args, src, p)
        wanted = request(identity(src), 'diarize', {
            **parameters, **({'merge': merge_options} if args.merge else {}),
            'sample_rate': rate, 'channels': args.channels,
            'min_duration_s': args.min_duration_s, 'max_duration_s': args.max_duration_s,
            **long_segment_parameters(args, vad_report)}, 'nemotron3')
        raw_path = dest.with_name('nemotron3.native.json')
        probabilities_path = dest.with_name('nemotron3.probabilities.npy')
        if manifest_complete(dest, wanted, args.overwrite):
            old = read_json(dest)
            if (raw_path.is_file() and probabilities_path.is_file()
                    and digest(raw_path) == old.get('native_output_sha256')
                    and digest(probabilities_path) == old.get('probabilities_sha256')):
                ensure_plots(dest, overwrite=False)
                return
        if model is None:
            model = load_model(args)
        native, probabilities = generate(model, src, args.work_dir)
        # Persist untouched model responses before parsing or filtering turns.
        write_json(raw_path, {**wanted, 'segments': native, 'device': str(model.device),
                             'frame_duration_s': model.output_subsampling_factor * 0.01})
        np.save(probabilities_path, probabilities, allow_pickle=False)
        turns = parse_segments(native)
        export({**wanted, 'speaker_ids': sorted({t['speaker_id'] for t in turns}),
                'turns': turns, 'native_output_sha256': digest(raw_path),
                'probabilities_sha256': digest(probabilities_path)},
               src, dest, args.work_dir, rate, args.channels,
               args.min_duration_s, args.max_duration_s,
               concurrency=1, batch_size=args.batch_size,
               long_segment_strategy=args.long_segment_strategy, vad_report=vad_report,
               vad_device=args.vad_device, vad_cut_threshold=args.vad_cut_threshold)
        ensure_plots(dest, overwrite=True)

    result = batch(pairs, process, concurrency=1, batch_size=args.batch_size)
    ensure_family_plots(pairs, args, concurrency=1, batch_size=args.batch_size)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
