"""Create deterministic RMS-controlled mixtures and exact reference stems."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import ROOT, completed, identity, positive_int, publish, request
import librosa
import numpy as np
import soundfile as sf

_SILENCE_RMS = 1e-12


def _load_audio(audio: Path, sample_rate: int, channels: int) -> np.ndarray:
    """Load an audio file as a floating-point, sample-major waveform."""
    waveform, source_rate = sf.read(
        audio,
        dtype="float64",
        always_2d=True,
    )
    if waveform.shape[0] == 0:
        raise ValueError(f"Audio source is empty: {audio}")

    waveform = _convert_channels(waveform, channels)
    if source_rate != sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=source_rate,
            target_sr=sample_rate,
            axis=0,
        )
    return np.asarray(waveform, dtype=np.float64)


def _convert_channels(waveform: np.ndarray, channels: int) -> np.ndarray:
    """Convert arbitrary input channel layouts to mono or sensible stereo."""
    source_channels = waveform.shape[1]
    if channels == 1:
        return np.mean(waveform, axis=1, keepdims=True)
    if source_channels == 1:
        return np.repeat(waveform, 2, axis=1)
    if source_channels == 2:
        return waveform

    # Keep left/right channel groups distinct while folding surround channels.
    left = np.mean(waveform[:, 0::2], axis=1)
    right = np.mean(waveform[:, 1::2], axis=1)
    return np.column_stack((left, right))


def _rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def mix(speech: Path, music: Path, target_smr_db: float, seed: int, output_dir: Path,
        sample_rate: int, channels: int, peak_ceiling_dbfs: float) -> dict:
    """Write mixture and references using the existing version-2 mixer arithmetic."""
    speech_waveform = _load_audio(speech, sample_rate, channels)
    music_waveform = _load_audio(music, sample_rate, channels)

    speech_samples = speech_waveform.shape[0]
    music_samples = music_waveform.shape[0]
    rng = np.random.default_rng(seed)
    if music_samples >= speech_samples:
        maximum_start = music_samples - speech_samples
        music_start_sample = int(rng.integers(0, maximum_start + 1))
        music_crop = music_waveform[
            music_start_sample : music_start_sample + speech_samples
        ]
    else:
        # Start at a deterministic point, wrap at the end of the track,
        # and repeat until the music bed exactly matches the speech.
        music_start_sample = int(rng.integers(0, music_samples))
        music_crop = np.empty(
            (speech_samples, music_waveform.shape[1]),
            dtype=np.float64,
        )
        write_position = 0
        source_position = music_start_sample
        while write_position < speech_samples:
            copy_samples = min(
                music_samples - source_position,
                speech_samples - write_position,
            )
            music_crop[write_position : write_position + copy_samples] = (
                music_waveform[source_position : source_position + copy_samples]
            )
            write_position += copy_samples
            source_position = 0

    speech_rms = _rms(speech_waveform)
    music_rms = _rms(music_crop)
    if speech_rms <= _SILENCE_RMS:
        raise ValueError("Speech is effectively silent; cannot establish an SMR")
    if music_rms <= _SILENCE_RMS:
        raise ValueError("Selected music crop is effectively silent; cannot establish an SMR")

    speech_gain_db = 0.0
    music_gain = speech_rms / (music_rms * _db_to_gain(target_smr_db))
    music_gain_db = float(20.0 * np.log10(music_gain))
    speech_reference = speech_waveform
    music_reference = music_crop * music_gain
    mixture = speech_reference + music_reference

    peak = float(np.max(np.abs(mixture)))
    ceiling = _db_to_gain(peak_ceiling_dbfs)
    common_output_gain_db = 0.0
    if peak > ceiling:
        common_gain = ceiling / peak
        common_output_gain_db = float(20.0 * np.log10(common_gain))
        speech_reference = speech_reference * common_gain
        music_reference = music_reference * common_gain
        mixture = speech_reference + music_reference

    realized_rms_smr_db = float(
        20.0 * np.log10(_rms(speech_reference) / _rms(music_reference))
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    speech_path = output_path / "speech_reference.wav"
    music_path = output_path / "music_reference.wav"
    mixture_path = output_path / "mixture.wav"
    input_paths = {Path(speech).resolve(), Path(music).resolve()}
    if any(path.resolve() in input_paths for path in (speech_path, music_path, mixture_path)):
        raise ValueError("output_dir would overwrite an input audio file")

    sf.write(speech_path, speech_reference, sample_rate, subtype="FLOAT")
    sf.write(music_path, music_reference, sample_rate, subtype="FLOAT")
    sf.write(mixture_path, mixture, sample_rate, subtype="FLOAT")

    return {'target_smr_db': target_smr_db, 'seed': seed, 'sample_rate': sample_rate,
            'channels': channels, 'music_start_sample': music_start_sample,
            'speech_gain_db': speech_gain_db, 'music_gain_db': music_gain_db,
            'common_output_gain_db': common_output_gain_db, 'peak_ceiling_dbfs': peak_ceiling_dbfs,
            'realized_rms_smr_db': realized_rms_smr_db, 'mixer_version': '2'}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--speech', type=Path, required=True)
    p.add_argument('--music', type=Path, required=True)
    p.add_argument('--smr-db', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--sample-rate', type=positive_int, default=44100)
    p.add_argument('--channels', type=int, choices=(1, 2), default=2)
    p.add_argument('--peak-ceiling-dbfs', type=float, default=-1.0)
    p.add_argument('--output-dir', type=Path, default=ROOT / '.data/mix/out')
    p.add_argument('--work-dir', type=Path, default=ROOT / '.data/mix/work')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not math.isfinite(args.smr_db) or not math.isfinite(args.peak_ceiling_dbfs) or args.peak_ceiling_dbfs > 0 or args.seed < 0:
        p.error('Require finite SMR, finite peak ceiling <= 0, and nonnegative seed')
    source = {'speech': identity(args.speech), 'music': identity(args.music)}
    params = {'smr_db': args.smr_db, 'seed': args.seed, 'sample_rate': args.sample_rate,
              'channels': args.channels, 'peak_ceiling_dbfs': args.peak_ceiling_dbfs, 'mixer_version': '2'}
    names = ('speech_reference.wav', 'music_reference.wav', 'mixture.wav')
    destinations = [args.output_dir.resolve() / name for name in names]
    if any(path in {args.speech.resolve(), args.music.resolve()} for path in destinations):
        p.error('Output would overwrite an input')
    wanted = [request(source, 'mix', {**params, 'stem': name}) for name in names]
    skips = [completed(path, metadata, args.overwrite) for path, metadata in zip(destinations, wanted)]
    if not all(skips):
        args.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            effective = mix(args.speech, args.music, args.smr_db, args.seed, Path(work),
                            args.sample_rate, args.channels, args.peak_ceiling_dbfs)
            for dest, metadata in zip(destinations, wanted):
                publish(Path(work) / dest.name, dest, {**metadata, 'mixing': effective})
    for dest in destinations:
        print(dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
