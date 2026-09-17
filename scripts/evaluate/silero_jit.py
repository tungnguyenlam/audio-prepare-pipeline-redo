"""Check native Silero JIT inference and compare CPU/GPU probabilities and timing.

This measures VAD execution, not sentence segmentation or phoneme completeness.
Raw frame probabilities are retained in the JSON report for independent review.
"""
from __future__ import annotations

import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, identity, positive_int, progress, write_json

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512


def infer_probabilities(model, waveform, *, transfer_block_frames: int = 256):
    """Keep recurrent state across frames; transfer results in bounded blocks."""
    import torch

    model.reset_states()
    blocks, pending = [], []
    with torch.inference_mode():
        for start in range(0, waveform.numel(), FRAME_SAMPLES):
            frame = waveform[start:start + FRAME_SAMPLES]
            if frame.numel() < FRAME_SAMPLES:
                frame = torch.nn.functional.pad(frame, (0, FRAME_SAMPLES - frame.numel()))
            pending.append(model(frame, SAMPLE_RATE).reshape(()))
            if len(pending) == transfer_block_frames:
                blocks.append(torch.stack(pending).cpu())
                pending.clear()
        if pending:
            blocks.append(torch.stack(pending).cpu())
    return torch.cat(blocks).numpy()


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-file', type=Path, required=True)
    p.add_argument('--model-file', type=Path, required=True, help='Existing silero_vad.jit; no downloads or ONNX')
    p.add_argument('--output-file', type=Path, default=ROOT / '.data/evaluate/silero_jit/report.json')
    p.add_argument('--devices', nargs='+', default=['cpu', 'cuda:0'], help='ROCm also uses cuda:0')
    p.add_argument('--repeats', type=positive_int, default=3)
    p.add_argument('--torch-threads', type=positive_int, default=1)
    p.add_argument('--threshold', type=float, default=0.35)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        p.error('--threshold must be finite and between 0 and 1')
    if args.model_file.suffix != '.jit':
        p.error('--model-file must be a native .jit model')
    destination = args.output_file.resolve()
    if destination.suffix != '.json':
        p.error('--output-file must end in .json')
    if destination in {args.input_file.resolve(), args.model_file.resolve()}:
        p.error('Output cannot replace an input')
    if destination.exists() and not args.overwrite:
        p.error('Output exists; choose another --output-file or use --overwrite')

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    torch.set_num_threads(args.torch_threads)
    t0 = time.perf_counter()
    audio, rate = sf.read(args.input_file, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        p.error('Input must contain finite nonempty audio')
    source_frames, channels = audio.shape
    waveform = torch.from_numpy(audio.mean(axis=1))
    if rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)
    preprocessing_s = time.perf_counter() - t0
    duration = source_frames / rate
    report = {
        'schema_version': 1, 'operation': 'evaluate_silero_jit',
        'source': identity(args.input_file), 'model': identity(args.model_file),
        'parameters': {'devices': args.devices, 'repeats': args.repeats,
                       'torch_threads': args.torch_threads, 'threshold': args.threshold,
                       'sample_rate': SAMPLE_RATE, 'frame_samples': FRAME_SAMPLES,
                       'transfer_block_frames': 256, 'downmix': 'channel_mean',
                       'dtype': 'float32', 'resampler': 'torchaudio.functional.resample defaults'},
        'environment': {'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
                        'hip': torch.version.hip, 'cuda': torch.version.cuda},
        'audio': {'source_sample_rate': rate, 'source_frames': source_frames,
                  'channels': channels, 'duration_s': duration,
                  'analysis_samples': waveform.numel(),
                  'frame_count': math.ceil(waveform.numel() / FRAME_SAMPLES),
                  'timestamp_origin_s': 0, 'preprocessing_s': preprocessing_s},
        'devices': {}, 'complete': False,
    }
    write_json(destination, report)
    raw = {}
    for name in dict.fromkeys(args.devices):
        progress('SILERO_CHECK', f'Checking {name} on {duration:.3f}s of audio')
        try:
            device = torch.device(name)
            if device.type not in {'cpu', 'cuda'}:
                raise ValueError('Use cpu or cuda:N (also for ROCm)')
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA/HIP device unavailable in this PyTorch environment')

            def synchronize():
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            model = torch.jit.load(str(args.model_file), map_location=device).eval()
            synchronize()
            load_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            resident = waveform.to(device)
            synchronize()
            upload_s = time.perf_counter() - t0
            # Warm TorchScript and device kernels, without contaminating run state.
            infer_probabilities(model, resident[:min(resident.numel(), 10 * FRAME_SAMPLES)])
            timings, arrays = [], []
            for _ in range(args.repeats):
                synchronize()
                t0 = time.perf_counter()
                probabilities = infer_probabilities(model, resident)
                synchronize()
                timings.append(time.perf_counter() - t0)
                if len(probabilities) != report['audio']['frame_count']:
                    raise ValueError('Incorrect probability frame count')
                if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
                    raise ValueError('Invalid probability values')
                arrays.append(probabilities)
            repeat_delta = max(float(np.max(np.abs(a - arrays[0]))) for a in arrays)
            raw[name] = arrays[0]
            median_s = statistics.median(timings)
            report['devices'][name] = {
                'status': 'ok', 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                'load_s': load_s, 'upload_s': upload_s, 'inference_s': timings,
                'median_inference_s': median_s, 'real_time_factor': median_s / duration,
                'max_repeat_probability_difference': repeat_delta,
                'frames_above_threshold': int(np.sum(arrays[0] >= args.threshold)),
                'probabilities': arrays[0].tolist(),
            }
            del resident, model
        except Exception as exc:
            report['devices'][name] = {'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc)}
        write_json(destination, report)
    if 'cpu' in raw:
        for name, values in raw.items():
            if name == 'cpu':
                continue
            delta = np.abs(values - raw['cpu'])
            changed = np.flatnonzero((values >= args.threshold) != (raw['cpu'] >= args.threshold))
            report['devices'][name]['comparison_to_cpu'] = {
                'max_absolute_probability_difference': float(delta.max()),
                'mean_absolute_probability_difference': float(delta.mean()),
                'threshold_disagreement_frames': changed.tolist(),
                'threshold_disagreement_count': len(changed),
            }
    report['complete'] = True
    report['all_devices_ok'] = all(r['status'] == 'ok' for r in report['devices'].values())
    report['limitations'] = [
        'Execution and numerical comparison only; no labelled VAD accuracy or phoneme completeness claim.',
        'Sequential single-recording inference, not a multi-recording GPU batch benchmark.',
        'Warm inference timings exclude model loading, preprocessing, and initial upload.',
        'Threshold frame disagreements do not measure final hysteresis/cut decisions.',
    ]
    write_json(destination, report)
    print(destination)
    return 0 if report['all_devices_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
