"""Reusable audio mixer for the separation benchmark."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.benchmark.separation.schemas import AudioMixResult, MixingParameters
from src.utils.AudioClass import Audio


_SILENCE_RMS = 1e-12


def _load_audio(audio: Audio, sample_rate: int, channels: int) -> np.ndarray:
    """Load an Audio as a floating-point, sample-major waveform."""
    waveform, source_rate = sf.read(
        audio.path,
        dtype="float64",
        always_2d=True,
    )
    if waveform.shape[0] == 0:
        raise ValueError(f"Audio source is empty: {audio.path}")

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


class AudioMixer:
    """Create deterministic, RMS-controlled speech/music benchmark mixtures."""

    def __init__(
        self,
        sample_rate: int = 44_100,
        channels: int = 2,
        peak_ceiling_dbfs: float = -1.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels not in (1, 2):
            raise ValueError("channels must be 1 (mono) or 2 (stereo)")
        if peak_ceiling_dbfs > 0.0:
            raise ValueError("peak_ceiling_dbfs must not exceed 0 dBFS")

        self.sample_rate = sample_rate
        self.channels = channels
        self.peak_ceiling_dbfs = float(peak_ceiling_dbfs)

    def mix(
        self,
        speech: Audio,
        music: Audio,
        *,
        target_smr_db: float,
        seed: int,
        output_dir: str | Path,
    ) -> AudioMixResult:
        """Mix speech with deterministic cropped/looped music and write references."""
        speech_waveform = _load_audio(speech, self.sample_rate, self.channels)
        music_waveform = _load_audio(music, self.sample_rate, self.channels)

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
        ceiling = _db_to_gain(self.peak_ceiling_dbfs)
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
        input_paths = {Path(speech.path).resolve(), Path(music.path).resolve()}
        if any(path.resolve() in input_paths for path in (speech_path, music_path, mixture_path)):
            raise ValueError("output_dir would overwrite an input audio file")

        sf.write(speech_path, speech_reference, self.sample_rate, subtype="FLOAT")
        sf.write(music_path, music_reference, self.sample_rate, subtype="FLOAT")
        sf.write(mixture_path, mixture, self.sample_rate, subtype="FLOAT")

        parameters = MixingParameters(
            target_smr_db=float(target_smr_db),
            sample_rate=self.sample_rate,
            channels=self.channels,
            seed=int(seed),
            music_start_sample=music_start_sample,
            speech_gain_db=speech_gain_db,
            music_gain_db=music_gain_db,
            common_output_gain_db=common_output_gain_db,
            peak_ceiling_dbfs=self.peak_ceiling_dbfs,
            mixer_version="2",
            speech_lufs_before=None,
            music_lufs_before=None,
            realized_rms_smr_db=realized_rms_smr_db,
        )
        return AudioMixResult(
            speech_reference=Audio.from_file(
                speech_path,
                source_id=f"{speech.source_id}__speech_reference",
            ),
            music_reference=Audio.from_file(
                music_path,
                source_id=f"{music.source_id}__music_reference",
            ),
            mixture=Audio.from_file(
                mixture_path,
                source_id=f"{speech.source_id}__{music.source_id}__mixture",
            ),
            parameters=parameters,
        )
