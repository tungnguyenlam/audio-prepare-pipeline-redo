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
