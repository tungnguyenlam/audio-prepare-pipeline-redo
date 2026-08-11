"""Data structures for the separation golden benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.utils.AudioClass import Audio


class Difficulty(str, Enum):
    """Difficulty level assigned to a planned separation sample."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MusicCategory(str, Enum):
    """Music category used to organize and stratify benchmark samples."""

    ACOUSTIC = "acoustic"
    ROCK = "rock"
    ELECTRONIC = "electronic"
    ORCHESTRAL = "orchestral"
    TRADITIONAL = "traditional"
    BEAT_DRIVEN = "beat_driven"


@dataclass
class BenchmarkDefinition:
    """One planned mixture before it is rendered."""

    sample_id: str
    speech_path: str
    music_path: str
    music_category: MusicCategory
    difficulty: Difficulty
    target_smr_db: float
    seed: int


@dataclass
class MixingParameters:
    """Parameters and realized values needed to reproduce a rendered mix."""

    target_smr_db: float
    sample_rate: int
    channels: int
    seed: int
    music_start_sample: int
    speech_gain_db: float
    music_gain_db: float
    common_output_gain_db: float
    peak_ceiling_dbfs: float
    mixer_version: str
    speech_lufs_before: float | None = None
    music_lufs_before: float | None = None
    realized_rms_smr_db: float | None = None


@dataclass
class AudioMixResult:
    """Exact source components and mixture produced by a future mixer."""

    speech_reference: Audio
    music_reference: Audio
    mixture: Audio
    parameters: MixingParameters


@dataclass
class SeparationBenchmarkSample:
    """One fully rendered separation benchmark sample."""

    sample_id: str
    speech_source: Audio
    music_source: Audio
    speech_reference: Audio
    music_reference: Audio
    mixture: Audio
    music_category: MusicCategory
    difficulty: Difficulty
    mixing: MixingParameters
