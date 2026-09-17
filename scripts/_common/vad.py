"""Shared generation, validation, and planning helpers for Silero VAD cuts."""
from __future__ import annotations

import bisect
import math
from pathlib import Path
import time

from _common.files import (FileContractError, ROOT, identity, probe, progress,
                           read_json, write_json)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
DEFAULT_MODEL_FILE = Path.home() / '.cache/silero-vad/silero_vad.jit'


def infer_probabilities(model, waveform, *, transfer_block_frames: int = 256):
    """Keep Silero's recurrent state across frames and transfer bounded blocks."""
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


def vad_device_candidates(vad_device: str) -> tuple[str, ...]:
    """Return the ordered devices attempted for an explicit device or ``auto``."""
    return ('cuda:0', 'cpu') if vad_device == 'auto' else (vad_device,)


def select_vad_device(report: dict, vad_device: str) -> str:
    """Select a successful probability track, with ``auto`` preferring GPU."""
    devices = report.get('devices', {})
    for candidate in vad_device_candidates(vad_device):
        if devices.get(candidate, {}).get('status') == 'ok':
            if vad_device == 'auto' and candidate != 'cuda:0':
                progress('VAD_FALLBACK', 'Silero auto device using cpu')
            return candidate
    if vad_device == 'auto':
        raise FileContractError('No successful Silero VAD track is available for auto (tried cuda:0, cpu)')
    raise FileContractError(f'Requested VAD track did not complete successfully: {vad_device}')


def auto_vad_report_path(source: Path) -> Path:
    """Return the content-addressed runtime report path for automatic VAD."""
    return ROOT / '.data' / 'vad' / 'auto' / f'{identity(source)["sha256"]}.json'


def _read_vad_audio(source: Path):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    audio, rate = sf.read(source, dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise FileContractError('Input must contain finite nonempty audio')
    source_frames, channels = audio.shape
    waveform = torch.from_numpy(audio.mean(axis=1))
    if rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, rate, SAMPLE_RATE)
    return waveform, rate, source_frames, channels


def ensure_vad_report(source: Path, destination: Path | None = None,
                      vad_device: str = 'auto') -> Path:
    """Create or reuse an automatic Silero report and return its path.

    ``auto`` attempts ``cuda:0`` first and retries the same source on CPU after
    any GPU loading or inference error. Explicit report paths are still loaded
    by ``load_vad_report`` and never regenerated here.
    """
    source = source.resolve()
    destination = (destination or auto_vad_report_path(source)).resolve()
    try:
        cached = read_json(destination)
        if (cached.get('source', {}).get('sha256') == identity(source)['sha256'] and
                cached.get('complete')):
            load_vad_report(source, destination, vad_device)
            return destination
    except (FileContractError, OSError, ValueError, KeyError):
        pass

    model_file = DEFAULT_MODEL_FILE
    if not model_file.is_file():
        raise FileContractError(
            f'Missing Silero JIT model: {model_file}; run ./envs/setup_worker_envs.sh diarizen '
            '(or audio)'
        )

    import numpy as np
    import torch
    import torchaudio

    waveform, rate, source_frames, channels = _read_vad_audio(source)
    duration = source_frames / rate
    report = {
        'schema_version': 1,
        'operation': 'evaluate_silero_jit',
        'source': identity(source),
        'model': identity(model_file),
        'parameters': {
            'devices': [vad_device],
            'repeats': 1,
            'torch_threads': torch.get_num_threads(),
            'threshold': 0.1,
            'sample_rate': SAMPLE_RATE,
            'frame_samples': FRAME_SAMPLES,
            'transfer_block_frames': 256,
            'downmix': 'channel_mean',
            'dtype': 'float32',
            'resampler': 'torchaudio.functional.resample defaults',
            'automatic': True,
        },
        'environment': {
            'torch': torch.__version__,
            'torchaudio': torchaudio.__version__,
            'hip': torch.version.hip,
            'cuda': torch.version.cuda,
        },
        'audio': {
            'source_sample_rate': rate,
            'source_frames': source_frames,
            'channels': channels,
            'duration_s': duration,
            'analysis_samples': waveform.numel(),
            'frame_count': math.ceil(waveform.numel() / FRAME_SAMPLES),
            'timestamp_origin_s': 0,
        },
        'devices': {},
        'complete': False,
    }
    write_json(destination, report)
    raw = {}
    attempted = vad_device_candidates(vad_device)
    for name in attempted:
        progress('SILERO_AUTO', f'Checking {name} on {duration:.3f}s of audio')
        device = None
        model = None
        resident = None
        try:
            device = torch.device(name)
            if device.type not in {'cpu', 'cuda'}:
                raise ValueError('Use auto, cpu, or cuda:N (also for ROCm)')
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA/HIP device unavailable in this PyTorch environment')

            def synchronize():
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            model = torch.jit.load(str(model_file), map_location=device).eval()
            synchronize()
            load_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            resident = waveform.to(device)
            synchronize()
            upload_s = time.perf_counter() - t0
            infer_probabilities(model, resident[:min(resident.numel(), 10 * FRAME_SAMPLES)])
            synchronize()
            t0 = time.perf_counter()
            probabilities = infer_probabilities(model, resident)
            synchronize()
            inference_s = time.perf_counter() - t0
            if len(probabilities) != report['audio']['frame_count']:
                raise ValueError('Incorrect probability frame count')
            if (not np.isfinite(probabilities).all() or
                    np.any((probabilities < 0) | (probabilities > 1))):
                raise ValueError('Invalid probability values')
            raw[name] = probabilities
            report['devices'][name] = {
                'status': 'ok',
                'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                'load_s': load_s,
                'upload_s': upload_s,
                'inference_s': [inference_s],
                'median_inference_s': inference_s,
                'real_time_factor': inference_s / duration,
                'max_repeat_probability_difference': 0.0,
                'frames_above_threshold': int(np.sum(probabilities >= 0.1)),
                'probabilities': probabilities.tolist(),
            }
            report['active_device'] = name
            del resident, model
            resident = model = None
            break
        except Exception as exc:
            if resident is not None:
                del resident
            if model is not None:
                del model
            if device is not None and device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            report['devices'][name] = {
                'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc)
            }
            write_json(destination, report)
            if vad_device != 'auto':
                raise
            progress('SILERO_AUTO', f'{name} failed; trying cpu')

        write_json(destination, report)

    if not raw:
        errors = '; '.join(
            f'{name}: {track.get("error", "unknown error")}'
            for name, track in report['devices'].items()
        )
        raise RuntimeError(f'Silero VAD failed on {vad_device}: {errors}')

    if 'cpu' in raw:
        for name, values in raw.items():
            if name == 'cpu':
                continue
            delta = np.abs(values - raw['cpu'])
            changed = np.flatnonzero((values >= 0.1) != (raw['cpu'] >= 0.1))
            report['devices'][name]['comparison_to_cpu'] = {
                'max_absolute_probability_difference': float(delta.max()),
                'mean_absolute_probability_difference': float(delta.mean()),
                'threshold_disagreement_frames': changed.tolist(),
                'threshold_disagreement_count': len(changed),
            }
    report['complete'] = True
    report['all_devices_ok'] = all(track['status'] == 'ok'
                                   for track in report['devices'].values())
    report['active_device_ok'] = True
    write_json(destination, report)
    return destination


def load_vad_report(source: Path, report_path: Path, vad_device: str) -> tuple[list[tuple[int, float, int, float]], dict]:
    """Validate a completed Silero report and return source-sample candidates."""
    source = source.resolve()
    report_path = report_path.resolve()
    source_info = probe(source)
    report = read_json(report_path)
    if report.get('operation') != 'evaluate_silero_jit' or not report.get('complete'):
        raise FileContractError('VAD report must be a completed evaluate/silero_jit report')
    if report.get('source', {}).get('sha256') != identity(source)['sha256']:
        raise FileContractError('VAD report and audio must identify the same source bytes')
    audio = report.get('audio', {})
    if (audio.get('source_sample_rate') != source_info['sample_rate'] or
            audio.get('source_frames') != source_info['frames']):
        raise FileContractError('VAD report source geometry does not match the input audio')
    parameters = report.get('parameters', {})
    analysis_rate = parameters.get('sample_rate')
    frame_samples = parameters.get('frame_samples')
    analysis_samples = audio.get('analysis_samples')
    if not all(isinstance(value, int) and value > 0
               for value in (analysis_rate, frame_samples, analysis_samples)):
        raise FileContractError('VAD report has invalid analysis geometry')
    selected_device = select_vad_device(report, vad_device)
    track = report.get('devices', {}).get(selected_device, {})
    if track.get('status') != 'ok':
        raise FileContractError(f'Requested VAD track did not complete successfully: {vad_device}')
    probabilities = track.get('probabilities')
    if (not isinstance(probabilities, list) or len(probabilities) != audio.get('frame_count') or
            not probabilities or
            not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                    for value in probabilities)):
        raise FileContractError('Requested VAD track has invalid probabilities')

    candidates = []
    for index, probability in enumerate(probabilities):
        frame_start = index * frame_samples
        frame_end = frame_start + frame_samples
        # The evaluator zero-pads its final partial frame. It cannot create a
        # valid boundary beyond the real source duration.
        if frame_end > analysis_samples:
            break
        center_s = ((frame_start + frame_end) / 2) / analysis_rate
        candidates.append((round(center_s * source_info['sample_rate']),
                           float(probability), index, center_s))
    return candidates, report


def plan_segments(source_frames: int, max_samples: int,
                  candidates: list[tuple[int, float, int, float]], *,
                  start_sample: int = 0, end_sample: int | None = None,
                  vad_cut_threshold: float | None = None) -> tuple[list[tuple[int, int, int, int | None]], list[dict]]:
    """Recursively split one interval at eligible low VAD probabilities.

    When the lowest available probability is not below ``vad_cut_threshold``,
    the interval is retained as an overlong leaf and an explicit rejection
    event is returned. Exporters can then apply their normal duration filter.
    """
    if max_samples <= 0:
        raise ValueError('max_samples must be positive')
    if (vad_cut_threshold is not None and
            (not math.isfinite(vad_cut_threshold) or not 0 <= vad_cut_threshold <= 1)):
        raise ValueError('vad_cut_threshold must be finite and between 0 and 1')
    end_sample = source_frames if end_sample is None else end_sample
    if not 0 <= start_sample < end_sample <= source_frames:
        raise ValueError('Expected an interval within the source timeline')

    candidate_samples = [candidate[0] for candidate in candidates]
    pending = [(start_sample, end_sample, 0, None)]
    leaves, cuts = [], []
    next_cut_id = 0
    while pending:
        start, end, depth, parent_cut_id = pending.pop()
        if end - start <= max_samples:
            leaves.append((start, end, depth, parent_cut_id))
            continue
        midpoint = (start + end) / 2
        first = bisect.bisect_right(candidate_samples, start)
        stop = bisect.bisect_left(candidate_samples, end)
        if first == stop:
            raise FileContractError(
                f'No interior VAD frame can split oversized interval {start}:{end}'
            )
        minimum_probability = min(candidates[index][1] for index in range(first, stop))
        if (vad_cut_threshold is not None and
                minimum_probability >= vad_cut_threshold):
            leaves.append((start, end, depth, parent_cut_id))
            cuts.append({
                'action': 'reject',
                'reason': 'no_vad_cut_below_threshold',
                'parent_cut_id': parent_cut_id,
                'depth': depth,
                'interval_start_sample': start,
                'interval_end_sample': end,
                'interval_duration_samples': end - start,
                'minimum_speech_probability': minimum_probability,
                'vad_cut_threshold': vad_cut_threshold,
            })
            continue
        selected_index = min(
            range(first, stop),
            key=lambda index: (
                candidates[index][1], abs(candidates[index][0] - midpoint),
                candidates[index][0], candidates[index][2]
            ),
        )
        selected = candidates[selected_index]
        cut_sample, probability, frame_index, frame_center_s = selected
        cut_id = next_cut_id
        next_cut_id += 1
        cuts.append({
            'cut_id': cut_id,
            'parent_cut_id': parent_cut_id,
            'depth': depth,
            'interval_start_sample': start,
            'interval_end_sample': end,
            'interval_duration_samples': end - start,
            'cut_sample': cut_sample,
            'vad_frame_index': frame_index,
            'candidate_index': selected_index,
            'vad_frame_center_s': frame_center_s,
            'speech_probability': probability,
            'tie_break': 'lowest_probability_then_nearest_midpoint_then_earliest',
        })
        pending.append((cut_sample, end, depth + 1, cut_id))
        pending.append((start, cut_sample, depth + 1, cut_id))
    return sorted(leaves), cuts
