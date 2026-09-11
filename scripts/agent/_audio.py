from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("verifier")


class VerifierResponseError(ValueError):
    """A model returned text that cannot become a verifier verdict."""

    def __init__(self, code: str, raw_response: str) -> None:
        super().__init__(code)
        self.code = code
        self.raw_response = raw_response


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
    """Robustly extract and parse JSON object from model output text.

    Handles thinking/reasoning tags (<think>...</think>), markdown code fences
    (```json ... ```), nested braces, and surrounding commentary.
    """
    cleaned = text.strip()

    # 1. Strip thinking / reasoning tags if present
    cleaned_no_think = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    target_text = cleaned_no_think if cleaned_no_think else cleaned

    # 2. Check for markdown code blocks (e.g. ```json ... ``` or ``` ... ```)
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", target_text)
    for block in reversed(code_blocks):
        try:
            return json.loads(block.strip())
        except Exception:
            pass

    # 3. Direct JSON parse
    try:
        return json.loads(target_text)
    except Exception:
        pass

    # 4. Search for balanced { ... } starting from the end (final model answer)
    end_idx = target_text.rfind("}")
    if end_idx != -1:
        depth = 0
        for i in range(end_idx, -1, -1):
            if target_text[i] == "}":
                depth += 1
            elif target_text[i] == "{":
                depth -= 1
                if depth == 0:
                    candidate = target_text[i : end_idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
                    break

    # 5. Fallback regex search for { ... }
    match = re.search(r"\{.*\}", target_text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"No valid JSON object could be extracted from: {text[:200]}")


def parse_verifier_response(text: str) -> dict[str, Any]:
    """Parse model text while retaining the exact response for artifact writing."""
    if not isinstance(text, str):
        text = str(text)
    try:
        parsed = extract_json_payload(text)
    except Exception as exc:
        raise VerifierResponseError("invalid_json", text) from exc
    if not isinstance(parsed, dict):
        raise VerifierResponseError("json_not_object", text)
    parsed["_raw_response"] = text
    parsed["_response_kind"] = "text"
    return parsed


def load_audio_waveform(audio_path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file as mono float32 numpy array resampled to target_sr."""
    import numpy as np
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
