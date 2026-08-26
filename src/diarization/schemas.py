"""Backend-independent schemas for speaker diarization results."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from src.utils.AudioClass import Audio


DIARIZATION_RESULT_KIND = "diarization.result"
DIARIZATION_SCHEMA_VERSION = "2.0"

_SPEAKER_FIELDS = {"speaker_id", "global_speaker_id"}
_MODEL_FIELDS = {"backend", "model_id", "revision"}
_TURN_FIELDS = {
    "speaker_id",
    "start_s",
    "end_s",
    "confidence",
    "overlaps_other_speaker",
}


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


def _speaker_from_payload(raw: Any) -> Speaker:
    """Build a speaker from JSON, ignoring viewer-only extra keys."""
    if isinstance(raw, str):
        return Speaker(speaker_id=raw)
    if not isinstance(raw, dict):
        raise TypeError("each speaker must be an object")
    payload = {key: raw[key] for key in _SPEAKER_FIELDS if key in raw}
    return Speaker(**payload)


def _model_from_payload(raw: Any) -> DiarizationModelInfo | None:
    if not isinstance(raw, dict):
        return None
    payload = {key: raw[key] for key in _MODEL_FIELDS if key in raw}
    if "backend" not in payload or "model_id" not in payload:
        return None
    return DiarizationModelInfo(**payload)


def _turn_from_payload(
    raw: Any,
    *,
    duration_s: float | None,
) -> SpeakerTurn | None:
    """Build one turn from JSON, clamping last-frame overshoot on load.

    Unknown viewer keys such as ``duration_s`` and ``has_overlap`` are ignored
    so previously persisted results can still be reopened in history.
    """
    if not isinstance(raw, dict):
        return None
    payload = dict(raw)
    payload.pop("duration_s", None)
    if "overlaps_other_speaker" not in payload and "has_overlap" in payload:
        payload["overlaps_other_speaker"] = bool(payload.get("has_overlap"))
    payload.pop("has_overlap", None)
    payload = {key: payload[key] for key in _TURN_FIELDS if key in payload}
    try:
        start_s = float(payload.get("start_s", 0))
        end_s = float(payload.get("end_s", 0))
    except (TypeError, ValueError):
        return None
    if duration_s is not None and isfinite(duration_s) and duration_s >= 0:
        start_s = min(max(0.0, start_s), duration_s)
        end_s = min(max(0.0, end_s), duration_s)
    payload["start_s"] = start_s
    payload["end_s"] = end_s
    try:
        return SpeakerTurn(**payload)
    except (TypeError, ValueError):
        return None


@dataclass
class SpeakerTurn:
    """One speaker being active during a time interval in an audio item."""

    speaker_id: str
    start_s: float
    end_s: float
    confidence: float | None = None
    overlaps_other_speaker: bool = False

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
        if not isinstance(self.overlaps_other_speaker, bool):
            raise TypeError("overlaps_other_speaker must be a bool")

    @property
    def duration_s(self) -> float:
        """Length of this turn in seconds."""
        return self.end_s - self.start_s


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


@dataclass(frozen=True)
class SpeakerSimilarityWindow:
    """One candidate sub-window scored against an enrolled speaker."""

    start_s: float
    end_s: float
    similarity: float

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class SpeakerPurityResult:
    """Speaker-purity decision and evidence for one candidate segment."""

    schema_version: str
    audio_id: str
    profile_name: str
    speaker_id: str
    start_s: float
    end_s: float
    decision: Literal["pass", "reject", "error"]
    overlap_duration_s: float
    overlap_ratio: float
    windows: tuple[SpeakerSimilarityWindow, ...]
    reason: str | None = None
    model: DiarizationModelInfo | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.schema_version, "schema_version")
        _validate_non_empty_string(self.audio_id, "audio_id")
        _validate_non_empty_string(self.profile_name, "profile_name")
        _validate_non_empty_string(self.speaker_id, "speaker_id")
        _validate_timestamp(self.start_s, "start_s")
        _validate_timestamp(self.end_s, "end_s")
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        if self.decision not in {"pass", "reject", "error"}:
            raise ValueError("decision must be 'pass', 'reject', or 'error'")
        if self.decision == "pass" and self.reason is not None:
            raise ValueError("passing results cannot contain a reason")
        if self.decision != "pass":
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("rejected and error results require a reason")
        if self.decision == "error":
            if not isinstance(self.error, str) or not self.error:
                raise ValueError("error results require a non-empty error message")
        elif self.error is not None:
            raise ValueError("only error results may contain an error message")

        _validate_timestamp(self.overlap_duration_s, "overlap_duration_s")
        if (
            isinstance(self.overlap_ratio, bool)
            or not isinstance(self.overlap_ratio, (int, float))
            or not isfinite(self.overlap_ratio)
            or not 0 <= self.overlap_ratio <= 1
        ):
            raise ValueError("overlap_ratio must be between 0 and 1")
        if not isinstance(self.windows, tuple) or not all(
            isinstance(window, SpeakerSimilarityWindow) for window in self.windows
        ):
            raise TypeError("windows must be a tuple of SpeakerSimilarityWindow")

    @property
    def passed(self) -> bool:
        """Whether this candidate is safe to admit to the dataset."""
        return self.decision == "pass"

    @property
    def duration_s(self) -> float:
        """Candidate duration in seconds."""
        return self.end_s - self.start_s

    @property
    def min_target_similarity(self) -> float | None:
        """Lowest target similarity across successfully embedded windows."""
        if not self.windows:
            return None
        return min(window.similarity for window in self.windows)


@dataclass
class DiarizationResult:
    """Canonical, file-backed diarization handoff for one audio item.

    Newly produced results always include ``source_audio``. ``None`` remains
    accepted only so schema-1.0 payloads can be read and migrated.
    """

    schema_version: str
    audio_id: str
    speakers: list[Speaker]
    turns: list[SpeakerTurn]
    source_audio: Audio | None = None
    model: DiarizationModelInfo | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    channel_url: str | None = None
    result_id: str = field(default_factory=lambda: f"diar_{uuid.uuid4().hex}")
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.schema_version, "schema_version")
        _validate_non_empty_string(self.audio_id, "audio_id")
        _validate_non_empty_string(self.result_id, "result_id")
        if (
            isinstance(self.created_at, bool)
            or not isinstance(self.created_at, (int, float))
            or not isfinite(self.created_at)
            or self.created_at < 0
        ):
            raise ValueError("created_at must be a finite non-negative timestamp")

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

        if self.schema_version.startswith("2") and self.source_audio is None:
            raise ValueError("schema 2.0 DiarizationResult requires source_audio")
        if self.source_audio is not None:
            if not isinstance(self.source_audio, Audio):
                raise TypeError("source_audio must be an Audio or None")
            if self.source_audio.source_id != self.audio_id:
                raise ValueError(
                    "source_audio.source_id must match audio_id: "
                    f"{self.source_audio.source_id!r} != {self.audio_id!r}"
                )
            for field_name in ("channel_id", "channel_name", "channel_url"):
                result_value = getattr(self, field_name)
                source_value = getattr(self.source_audio, field_name)
                if result_value is None:
                    setattr(self, field_name, source_value)
                elif source_value is not None and result_value != source_value:
                    raise ValueError(
                        f"{field_name} must match source_audio.{field_name}"
                    )
            if self.source_audio.duration_s is not None:
                out_of_bounds = [
                    turn for turn in self.turns
                    if turn.end_s > self.source_audio.duration_s + 0.05
                ]
                if out_of_bounds:
                    turn = out_of_bounds[0]
                    raise ValueError(
                        "turn exceeds source audio duration: "
                        f"{turn.speaker_id} ends at {turn.end_s:.3f}s, source "
                        f"duration is {self.source_audio.duration_s:.3f}s"
                    )

        # Normalize overlap evidence once so every serializer and consumer sees
        # the same value, regardless of whether the backend emitted it.
        for index, turn in enumerate(self.turns):
            if turn.overlaps_other_speaker:
                continue
            turn.overlaps_other_speaker = any(
                other.speaker_id != turn.speaker_id
                and turn.start_s < other.end_s
                and other.start_s < turn.end_s
                for other in self.turns[index + 1 :]
            ) or any(
                other.speaker_id != turn.speaker_id
                and turn.start_s < other.end_s
                and other.start_s < turn.end_s
                for other in self.turns[:index]
            )

    @property
    def speaker_count(self) -> int:
        """Number of declared speakers."""
        return len(self.speakers)

    @property
    def turn_count(self) -> int:
        """Number of speaker turns."""
        return len(self.turns)

    @property
    def total_speech_duration_s(self) -> float:
        """Sum of all turn durations, including simultaneous speech."""
        return sum(turn.duration_s for turn in self.turns)

    @property
    def duration_per_speaker_s(self) -> dict[str, float]:
        """Summed speech duration keyed by speaker ID."""
        totals = {speaker.speaker_id: 0.0 for speaker in self.speakers}
        for turn in self.turns:
            totals[turn.speaker_id] += turn.duration_s
        return totals

    @property
    def turns_by_speaker(self) -> dict[str, list[SpeakerTurn]]:
        """Turns grouped by speaker ID in their existing order."""
        grouped = {speaker.speaker_id: [] for speaker in self.speakers}
        for turn in self.turns:
            grouped[turn.speaker_id].append(turn)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        """Return the single canonical JSON-compatible representation."""
        source_audio = self.source_audio.metadata() if self.source_audio else None
        return {
            "kind": DIARIZATION_RESULT_KIND,
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "created_at": self.created_at,
            "audio_id": self.audio_id,
            "source_audio": source_audio,
            "speakers": [asdict(speaker) for speaker in self.speakers],
            "turns": [
                {
                    **asdict(turn),
                    "duration_s": turn.duration_s,
                }
                for turn in self.turns
            ],
            "model": asdict(self.model) if self.model else None,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "channel_url": self.channel_url,
            "summary": {
                "speaker_count": self.speaker_count,
                "turn_count": self.turn_count,
                "total_speech_duration_s": self.total_speech_duration_s,
                "duration_per_speaker_s": self.duration_per_speaker_s,
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_audio: Audio | None = None,
    ) -> DiarizationResult:
        """Restore a result from its canonical JSON representation.

        ``source_audio`` may be supplied only to migrate an older schema-1.0
        payload that did not contain a file-backed source snapshot.

        Load is more tolerant than direct construction: unknown viewer keys are
        ignored, last-frame timestamps that overshoot ``duration_s`` are
        clamped, and speakers referenced only by turns are added.
        """
        if not isinstance(data, dict):
            raise TypeError("DiarizationResult payload must be an object")
        kind = data.get("kind")
        if kind not in (None, DIARIZATION_RESULT_KIND):
            raise ValueError(f"Unsupported diarization payload kind: {kind!r}")

        source_payload = data.get("source_audio")
        if source_audio is None and isinstance(source_payload, dict):
            raw_path = source_payload.get("path")
            raw_source_id = source_payload.get("source_id") or data.get("audio_id")
            if raw_path and raw_source_id:
                source_audio = Audio(
                    path=Path(raw_path).expanduser().resolve(),
                    source_id=str(raw_source_id),
                    title=source_payload.get("title"),
                    sample_rate=source_payload.get("sample_rate"),
                    duration_s=source_payload.get("duration_s"),
                    channels=source_payload.get("channels"),
                    format=str(source_payload.get("format", "wav")),
                    native_sample_rate=source_payload.get("native_sample_rate"),
                    history=tuple(source_payload.get("history") or ()),
                    source_url=source_payload.get("source_url"),
                    channel_id=source_payload.get("channel_id"),
                    channel_name=source_payload.get("channel_name"),
                    channel_url=source_payload.get("channel_url"),
                )

        duration_s = getattr(source_audio, "duration_s", None)
        turns = [
            turn
            for turn in (
                _turn_from_payload(raw_turn, duration_s=duration_s)
                for raw_turn in data.get("turns", [])
            )
            if turn is not None
        ]
        speaker_payloads = data.get("speakers") or [
            {"speaker_id": speaker_id}
            for speaker_id in sorted({turn.speaker_id for turn in turns})
        ]
        speakers = [_speaker_from_payload(speaker) for speaker in speaker_payloads]
        declared_ids = {speaker.speaker_id for speaker in speakers}
        for turn in turns:
            if turn.speaker_id not in declared_ids:
                speakers.append(Speaker(speaker_id=turn.speaker_id))
                declared_ids.add(turn.speaker_id)

        model_payload = data.get("model")
        if not model_payload and data.get("backend"):
            model_payload = {
                "backend": str(data["backend"]),
                "model_id": str(data.get("model_id") or data["backend"]),
            }
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            audio_id=str(data["audio_id"]),
            speakers=speakers,
            turns=turns,
            source_audio=source_audio,
            model=_model_from_payload(model_payload),
            channel_id=data.get("channel_id") or getattr(source_audio, "channel_id", None),
            channel_name=data.get("channel_name") or getattr(source_audio, "channel_name", None),
            channel_url=data.get("channel_url") or getattr(source_audio, "channel_url", None),
            result_id=str(data.get("result_id") or f"diar_{uuid.uuid4().hex}"),
            created_at=float(data.get("created_at", time.time())),
        )

    def save(self, destination: str | Path) -> Path:
        """Persist this canonical result as JSON and return its path."""
        destination_path = Path(destination)
        path = (
            destination_path
            if destination_path.suffix.lower() == ".json"
            else destination_path / f"{self.result_id}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> DiarizationResult:
        """Load a canonical result JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
