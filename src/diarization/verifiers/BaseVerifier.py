from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("verifier")

DEFAULT_ACOUSTIC_PROMPT = """Listen to the supplied audio directly. Do not transcribe it.
Evaluate three strict acoustic dimensions required for clean Text-to-Speech (TTS) training:

1. SPEAKER PURITY:
   - "pure": Exactly one primary speaker throughout. No background chatter, secondary voices, or intruder laughter.
   - "secondary_speaker": Another speaker's voice is audible (even a short word, whisper, or breath).
   - "overlapping_speech": Multiple speakers speaking or laughing simultaneously.

2. WORD COMPLETENESS (Không lẹm chữ, đủ âm tiết):
   - "complete": All words start and finish on clean acoustic word boundaries with their full vowel decay and coda consonant closure.
   - "clipped_word_start": The initial word has its onset consonant abruptly cut off.
   - "clipped_word_end": The final word is cut off abruptly while vocal fold vibration or tone is still in flight.

3. AUDIO QUALITY:
   - "studio_clean": Clear vocal signal, minimal background artifacts, and no residual music bleed.
   - "music_bleed": Audible residual background music, beats, or synthetic melodies.
   - "noisy_reverberant": Severe room echo, reverb, or excessive environmental noise.
   - "distorted": Clipping distortion, phase artifacts, or muffled frequency response.

DECISION RULE:
- "pass" ONLY if speaker_purity == "pure" AND word_completeness == "complete" AND audio_quality == "studio_clean".
- Otherwise "reject".

Return strict JSON only (no markdown, no other text):
{
  "speaker_purity": "pure" | "secondary_speaker" | "overlapping_speech",
  "word_completeness": "complete" | "clipped_word_start" | "clipped_word_end",
  "audio_quality": "studio_clean" | "music_bleed" | "noisy_reverberant" | "distorted",
  "decision": "pass" | "reject",
  "failure_codes": ["clipped_word_start", "clipped_word_end", "secondary_speaker", "overlapping_speech", "music_bleed", "noisy_reverberant", "distorted"],
  "reason": "Concise English explanation of the acoustic decision."
}"""


def extract_json_payload(text: str) -> dict[str, Any]:
    """Robustly extract and parse JSON object from model output text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def load_audio_waveform(audio_path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file as mono float32 numpy array resampled to target_sr."""
    path_str = str(audio_path)
    audio_data = None
    try:
        import soundfile as sf
        raw_audio, sr = sf.read(path_str, dtype="float32")
        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)
        if sr == target_sr:
            audio_data = raw_audio
        else:
            try:
                import torch
                import torchaudio
                t_audio = torch.from_numpy(raw_audio).unsqueeze(0)
                audio_data = torchaudio.functional.resample(t_audio, orig_freq=sr, new_freq=target_sr).squeeze(0).numpy()
            except Exception:
                pass
    except Exception:
        pass

    if audio_data is None:
        import librosa
        raw_audio, _ = librosa.load(path_str, sr=target_sr, mono=True)
        audio_data = np.asarray(raw_audio, dtype=np.float32)
    else:
        audio_data = np.asarray(audio_data, dtype=np.float32)

    return audio_data


class BaseVerifier(ABC):
    """Abstract base class for all direct-audio speech acoustic verifiers."""

    supports_concurrency: bool = False

    @abstractmethod
    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        """Verify an audio turn and return structured verdict dictionary.

        Args:
            audio_path: Path to the target audio file.
            prompt: Text prompt detailing evaluation dimensions and JSON schema.

        Returns:
            dict containing decision ('pass' | 'reject'), acoustic dimensions, failure_codes,
            reason, and '_latency_s'.
        """
        raise NotImplementedError
