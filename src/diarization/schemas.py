"""Backend-independent schemas for speaker diarization results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _validate_timestamp(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass
class SpeakerTurn:
    """One speaker being active during a time interval in an audio item."""

    speaker_id: str
    start_s: float
    end_s: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.speaker_id, "speaker_id")
        _validate_timestamp(self.start_s, "start_s")
        _validate_timestamp(self.end_s, "end_s")
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")

        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, (int, float)
            ):
                raise TypeError("confidence must be a number or None")
            if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")


@dataclass
class Speaker:
    """A speaker identity local to one diarization result."""

    speaker_id: str
    global_speaker_id: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.speaker_id, "speaker_id")
        if self.global_speaker_id is not None:
            _validate_non_empty_string(self.global_speaker_id, "global_speaker_id")


@dataclass
class DiarizationModelInfo:
    """Metadata identifying the backend and model used for diarization."""

    backend: str
    model_id: str
    revision: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.backend, "backend")
        _validate_non_empty_string(self.model_id, "model_id")
        if self.revision is not None:
            _validate_non_empty_string(self.revision, "revision")


@dataclass
class ScoredSegment:
    """One diarization turn scored against a target speaker profile."""

    speaker_id: str
    start_s: float
    end_s: float
    similarity: float
    overlaps_other_speaker: bool = False

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.speaker_id, "speaker_id")
        _validate_timestamp(self.start_s, "start_s")
        _validate_timestamp(self.end_s, "end_s")
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")

        if isinstance(self.similarity, bool) or not isinstance(
            self.similarity, (int, float)
        ):
            raise TypeError("similarity must be a number")
        if not isfinite(self.similarity) or not -1 <= self.similarity <= 1:
            raise ValueError("similarity must be between -1 and 1")

        if not isinstance(self.overlaps_other_speaker, bool):
            raise TypeError("overlaps_other_speaker must be a bool")

    @property
    def duration_s(self) -> float:
        """Length of the segment in seconds."""
        return self.end_s - self.start_s


@dataclass
class TargetSpeakerResult:
    """Diarization turns of one audio item scored against a speaker profile."""

    schema_version: str
    audio_id: str
    profile_name: str
    segments: list[ScoredSegment]
    model: DiarizationModelInfo | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    channel_url: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.schema_version, "schema_version")
        _validate_non_empty_string(self.audio_id, "audio_id")
        _validate_non_empty_string(self.profile_name, "profile_name")
        if not isinstance(self.segments, list):
            raise TypeError("segments must be a list")


@dataclass
class DiarizationResult:
    """Complete speaker and activity information for one audio item."""

    schema_version: str
    audio_id: str
    speakers: list[Speaker]
    turns: list[SpeakerTurn]
    model: DiarizationModelInfo | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    channel_url: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.schema_version, "schema_version")
        _validate_non_empty_string(self.audio_id, "audio_id")

        if not isinstance(self.speakers, list):
            raise TypeError("speakers must be a list")
        if not isinstance(self.turns, list):
            raise TypeError("turns must be a list")

        speaker_ids = [speaker.speaker_id for speaker in self.speakers]
        if len(speaker_ids) != len(set(speaker_ids)):
            raise ValueError("speakers must not contain duplicate speaker_id values")

        declared_speaker_ids = set(speaker_ids)
        unknown_speaker_ids = {
            turn.speaker_id
            for turn in self.turns
            if turn.speaker_id not in declared_speaker_ids
        }
        if unknown_speaker_ids:
            unknown = ", ".join(sorted(unknown_speaker_ids))
            raise ValueError(f"turns reference unknown speaker_id values: {unknown}")
