"""WAV I/O, ffmpeg resampling, and device pinning shared by speech-cleanup commands."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import tempfile

import numpy as np

from _common.files import completed, convert, identity, probe, publish, request


def pin_device_visibility(device: str) -> None:
    """Force CPU execution by clearing CUDA/HIP/ROCm visibility.

    Call this before importing torch. ``auto`` and ``cuda*`` leave visibility
    unchanged so the process can use the default GPU when the venv has one.
    """
    lowered = (device or 'auto').strip().lower()
    if lowered == 'cpu' or lowered.startswith('cpu:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HIP_VISIBLE_DEVICES'] = ''
        os.environ['ROCR_VISIBLE_DEVICES'] = ''


@contextlib.contextmanager
def working_directory(path: Path):
    """Temporarily chdir so model checkpoints are not written into the repo root."""
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


def as_sample_major(array: np.ndarray) -> np.ndarray:
    """Coerce model layouts to sample-major float32 ``(frames, channels)``."""
    wave = np.asarray(array, dtype=np.float32)
    if wave.ndim == 0:
        raise ValueError('Audio array is empty')
    if wave.ndim == 1:
        return wave.reshape(-1, 1)
    if wave.ndim == 3:
        wave = np.squeeze(wave)
        if wave.ndim == 3:
            raise ValueError(f'Unsupported audio array shape: {array.shape}')
        return as_sample_major(wave)
    if wave.ndim != 2:
        raise ValueError(f'Unsupported audio array shape: {array.shape}')
    if wave.shape[0] <= 8 and wave.shape[1] > wave.shape[0]:
        return np.ascontiguousarray(wave.T)
    return np.ascontiguousarray(wave)


def load_waveform(path: Path) -> tuple[np.ndarray, int]:
    """Return sample-major float32 audio and its sample rate."""
    import soundfile as sf

    wave, rate = sf.read(str(path), dtype='float32', always_2d=True)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError(f'Input must contain finite nonempty audio: {path}')
    return np.ascontiguousarray(wave), int(rate)


def save_waveform(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Write a floating-point WAV. Peak is scaled only when it would clip."""
    import soundfile as sf

    wave = as_sample_major(waveform)
    if wave.size == 0:
        raise ValueError(f'Refusing to write empty audio: {path}')
    peak = float(np.max(np.abs(wave)))
    if peak > 1.0:
        wave = wave / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, int(sample_rate), subtype='FLOAT')


def speech_gate_mask(
    probabilities: list[float],
    *,
    analysis_sample_rate: int,
    frame_samples: int,
    source_sample_rate: int,
    source_frames: int,
    threshold: float,
    pad_samples: int,
) -> np.ndarray:
    """Boolean speech mask at the source sample rate, with hangover padding."""
    if source_frames <= 0:
        raise ValueError('source_frames must be positive')
    mask = np.zeros(source_frames, dtype=np.float32)
    pad = max(0, int(pad_samples))
    for index, probability in enumerate(probabilities):
        if float(probability) < threshold:
            continue
        start_s = (index * frame_samples) / analysis_sample_rate
        end_s = ((index + 1) * frame_samples) / analysis_sample_rate
        start = max(0, int(round(start_s * source_sample_rate)) - pad)
        end = min(source_frames, int(round(end_s * source_sample_rate)) + pad)
        if end > start:
            mask[start:end] = 1.0
    return mask


def publish_model_audio(
    src: Path,
    dest: Path,
    *,
    model_sample_rate: int,
    output_sample_rate: int,
    channels: int,
    overwrite: bool,
    work_dir: Path,
    operation: str,
    model: str | None,
    parameters: dict,
    infer,
) -> None:
    """Resample to the model rate, run ``infer``, resample to the output rate, publish.

    ``infer(prepared_wav, model_wav)`` must write ``model_wav`` at ``model_sample_rate``.
    """
    metadata = request(identity(src), operation, parameters, model)
    if completed(dest, metadata, overwrite):
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as directory:
        work = Path(directory)
        prepared = work / 'input.wav'
        model_wav = work / 'model.wav'
        staged = work / 'output.wav'
        convert(src, prepared, model_sample_rate, channels, floating=True)
        infer(prepared, model_wav)
        if not model_wav.is_file():
            raise ValueError('Model inference did not write an output WAV')
        probe(model_wav)
        convert(model_wav, staged, output_sample_rate, channels)
        publish(staged, dest, metadata)
