"""Base interfaces and data structures for Word Alignment & VAD (Stage 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AlignedWord:
    """Represents a single transcribed word with exact millisecond timestamps."""
    word: str
    start_s: float
    end_s: float
    probability: float = 1.0
    duration_s: float = 0.0

    def __post_init__(self):
        self.start_s = round(float(self.start_s), 3)
        self.end_s = round(float(self.end_s), 3)
        if self.duration_s == 0.0:
            self.duration_s = round(max(0.0, self.end_s - self.start_s), 3)
        self.probability = round(float(self.probability), 3)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlignedSegment:
    """Represents a transcribed sentence or clause segment containing word-level details."""
    id: int
    text: str
    start_s: float
    end_s: float
    words: List[AlignedWord] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    duration_s: float = 0.0

    def __post_init__(self):
        self.start_s = round(float(self.start_s), 3)
        self.end_s = round(float(self.end_s), 3)
        if self.duration_s == 0.0:
            self.duration_s = round(max(0.0, self.end_s - self.start_s), 3)
        self.text = self.text.strip()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "avg_logprob": round(float(self.avg_logprob), 3),
            "no_speech_prob": round(float(self.no_speech_prob), 3),
            "words": [w.to_dict() for w in self.words],
        }


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    seconds = max(0.0, float(seconds))
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        seconds += 1
        millis = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds into WebVTT timestamp HH:MM:SS.mmm."""
    seconds = max(0.0, float(seconds))
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        seconds += 1
        millis = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


@dataclass
class AlignmentResult:
    """Overall result of Word Alignment & Transcription execution."""
    run_id: str
    input_file: Path
    source_type: str
    speaker_id: Optional[str]
    language: str
    language_probability: float
    model_size: str
    total_duration_s: float
    total_words: int
    total_segments: int
    words_per_minute: float
    segments: List[AlignedSegment]
    words: List[AlignedWord]
    full_transcript: str
    srt_content: str
    vtt_content: str
    output_dir: Path
    created_at: str
    audio_url: Optional[str] = None
    metadata_file: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "input_file": str(self.input_file),
            "input_filename": self.input_file.name,
            "source_type": self.source_type,
            "speaker_id": self.speaker_id,
            "language": self.language,
            "language_probability": round(float(self.language_probability), 3),
            "model_size": self.model_size,
            "total_duration_s": round(float(self.total_duration_s), 2),
            "total_words": self.total_words,
            "total_segments": self.total_segments,
            "words_per_minute": round(float(self.words_per_minute), 1),
            "full_transcript": self.full_transcript,
            "audio_url": self.audio_url,
            "created_at": self.created_at,
            "segments": [s.to_dict() for s in self.segments],
            "words": [w.to_dict() for w in self.words],
        }
