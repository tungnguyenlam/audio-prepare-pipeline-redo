"""Smart Chunking Engine Implementation.

Stage 4 in SOTA Audio Processing Pipeline:
Groups words into high-quality single-speaker speech segments satisfying:
- Constraint 1: 3.0s <= Duration <= 30.0s
- Constraint 2: Strictly cut at word boundary (current_word.end), never cutting inside words
- Constraint 3: Splits at natural pause points (silence >= 0.3s) whenever minimum duration is met
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from src.alignment.whisperx_aligner import AlignedWord
from src.utils.AudioClass import Audio, _probe_wav, _write_sidecar

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    start_s: float
    end_s: float
    duration_s: float
    text: str
    words: List[AlignedWord] = field(default_factory=list)
    speaker_id: Optional[str] = None
    audio_path: Optional[Path] = None


class SmartChunker:
    """Intelligent audio segmenter with word boundary preservation."""

    def __init__(
        self,
        min_duration_s: float = 3.0,
        max_duration_s: float = 28.0,
        pause_threshold_s: float = 0.3,
    ):
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.pause_threshold_s = pause_threshold_s

    def chunk_words(
        self, 
        words: List[AlignedWord], 
        speaker_id: Optional[str] = None
    ) -> List[AudioSegment]:
        """Compute optimal segment slices from aligned words."""
        if not words:
            return []

        segments: List[AudioSegment] = []
        current_chunk_words: List[AlignedWord] = []
        chunk_start = words[0].start_s

        for i, w in enumerate(words):
            current_chunk_words.append(w)
            current_duration = w.end_s - chunk_start

            is_last_word = (i == len(words) - 1)
            next_word = words[i + 1] if not is_last_word else None

            # Calculate gap to next word
            gap = (next_word.start_s - w.end_s) if next_word else 0.0

            should_split = False

            if current_duration >= self.min_duration_s:
                # Split condition 1: Met min duration + encountered a natural pause
                if gap >= self.pause_threshold_s:
                    should_split = True
                # Split condition 2: Approaching max duration threshold
                elif current_duration >= self.max_duration_s:
                    should_split = True

            if is_last_word or should_split:
                chunk_end = w.end_s
                text = " ".join(cw.word for cw in current_chunk_words)
                segments.append(AudioSegment(
                    start_s=chunk_start,
                    end_s=chunk_end,
                    duration_s=round(chunk_end - chunk_start, 4),
                    text=text,
                    words=list(current_chunk_words),
                    speaker_id=speaker_id
                ))
                
                # Reset for next chunk
                current_chunk_words = []
                if next_word:
                    chunk_start = next_word.start_s

        return segments
