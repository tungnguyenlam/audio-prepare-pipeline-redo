"""WhisperX Word-Level Forced Alignment Implementation.

Stage 3 in SOTA Audio Processing Pipeline:
Extracts exact millisecond timestamps (start, end) for every spoken word using Wav2Vec2 CTC alignment,
preventing word clipping at segment boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AlignedWord:
    word: str
    start_s: float
    end_s: float
    score: Optional[float] = None


class WhisperXAligner:
    """WhisperX transcription & word alignment wrapper."""

    def __init__(
        self,
        model_size: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._asr_model = None

    def _lazy_init(self):
        if self._asr_model is None:
            import whisperx
            import torch
            dev = self.device if torch.cuda.is_available() else "cpu"
            c_type = self.compute_type if torch.cuda.is_available() else "float32"
            logger.info(f"Loading WhisperX model '{self.model_size}' on {dev} ({c_type})...")
            self._asr_model = whisperx.load_model(self.model_size, dev, compute_type=c_type)

    def align_audio(
        self,
        audio_path: Union[str, Path],
        language_code: Optional[str] = None
    ) -> List[AlignedWord]:
        """Transcribe and align words with exact millisecond timestamps."""
        self._lazy_init()
        import whisperx
        import torch
        dev = self.device if torch.cuda.is_available() else "cpu"

        audio = whisperx.load_audio(str(audio_path))
        result = self._asr_model.transcribe(audio, batch_size=16)

        lang = language_code or result.get("language", "en")
        model_a, metadata = whisperx.load_align_model(language_code=lang, device=dev)
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            dev,
            return_char_alignments=False
        )

        words: List[AlignedWord] = []
        for word_dict in result_aligned.get("word_segments", []):
            if "start" in word_dict and "end" in word_dict:
                words.append(AlignedWord(
                    word=word_dict.get("word", ""),
                    start_s=float(word_dict["start"]),
                    end_s=float(word_dict["end"]),
                    score=float(word_dict.get("score", 1.0))
                ))

        return words
