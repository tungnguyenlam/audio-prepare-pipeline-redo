"""Backend-independent schemas for audiovisual speaker identity verification."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from src.data_paths import portable_data_path, resolve_data_path
from src.diarization.schemas import DiarizationModelInfo, SpeakerSimilarityWindow
from src.utils.AudioClass import Audio
from src.visual.Video import Video


AV_RESULT_KIND = "av.verification"
AV_SCHEMA_VERSION = "1.0"
FACE_TRACK_SET_KIND = "av.face_tracks"
ASD_RESULT_KIND = "av.asd"
SPEAKER_ENTITY_KIND = "av.speaker_entity"
ENTITY_SCHEMA_VERSION = "1.0"

VisualStatus = Literal["visual_verified", "audio_only", "visual_conflict"]
AVDecision = Literal["accept", "audio_only", "reject", "error"]


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


def _validate_unit_interval(value: float, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field_name} must be between 0 and 1")


def _embedding_to_list(embedding: tuple[float, ...] | None) -> list[float] | None:
    if embedding is None:
        return None
    return [float(value) for value in embedding]


def _embedding_from_payload(raw: Any) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise TypeError("embedding must be a list of numbers or null")
    return tuple(float(value) for value in raw)


@dataclass(frozen=True)
class FaceObservation:
    """One detected face in a single video frame."""

    frame_index: int
    time_s: float
    bbox_xyxy: tuple[float, float, float, float]
    det_score: float
    landmarks: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an int")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        _validate_timestamp(self.time_s, "time_s")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must be (x1, y1, x2, y2)")
        x1, y1, x2, y2 = (float(value) for value in self.bbox_xyxy)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive width and height")
        _validate_unit_interval(self.det_score, "det_score")

    @property
    def width(self) -> float:
        """Bounding-box width in pixels."""
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        """Bounding-box height in pixels."""
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]


@dataclass(frozen=True)
class FaceTrack:
    """A temporally linked face identity inside one video."""

    track_id: str
    observations: tuple[FaceObservation, ...]
    embedding: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.track_id, "track_id")
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("observations must be a non-empty tuple")
        if not all(isinstance(item, FaceObservation) for item in self.observations):
            raise TypeError("observations must contain FaceObservation values")
        if self.embedding is not None and len(self.embedding) == 0:
            raise ValueError("embedding cannot be empty when provided")

    @property
    def start_s(self) -> float:
        """First observation timestamp."""
        return self.observations[0].time_s

    @property
    def end_s(self) -> float:
        """Last observation timestamp."""
        return self.observations[-1].time_s

    @property
    def duration_s(self) -> float:
        """Inclusive observation span in seconds."""
        return self.end_s - self.start_s

    def overlaps(self, start_s: float, end_s: float) -> bool:
        """Return whether this track intersects ``[start_s, end_s]``."""
        return self.start_s < end_s and self.end_s > start_s


@dataclass
class FaceTrackSet:
    """All face tracks extracted from one video."""

    video_id: str
    tracks: tuple[FaceTrack, ...]
    fps: float
    duration_s: float
    model: DiarizationModelInfo | None = None
    video_path: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.video_id, "video_id")
        if not isinstance(self.tracks, tuple):
            raise TypeError("tracks must be a tuple")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("fps must be a positive number")
        _validate_timestamp(self.duration_s, "duration_s")

    def tracks_during(self, start_s: float, end_s: float) -> tuple[FaceTrack, ...]:
        """Return tracks that intersect an interval."""
        return tuple(track for track in self.tracks if track.overlaps(start_s, end_s))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot."""
        return {
            "kind": FACE_TRACK_SET_KIND,
            "schema_version": AV_SCHEMA_VERSION,
            "video_id": self.video_id,
            "video_path": (
                portable_data_path(self.video_path) if self.video_path else None
            ),
            "fps": self.fps,
            "duration_s": self.duration_s,
            "model": asdict(self.model) if self.model else None,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "embedding": _embedding_to_list(track.embedding),
                    "start_s": track.start_s,
                    "end_s": track.end_s,
                    "observations": [
                        {
                            "frame_index": obs.frame_index,
                            "time_s": obs.time_s,
                            "bbox_xyxy": list(obs.bbox_xyxy),
                            "det_score": obs.det_score,
                            "landmarks": (
                                [list(point) for point in obs.landmarks]
                                if obs.landmarks is not None
                                else None
                            ),
                        }
                        for obs in track.observations
                    ],
                }
                for track in self.tracks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceTrackSet:
        """Restore tracks from JSON."""
        if not isinstance(data, dict):
            raise TypeError("FaceTrackSet payload must be an object")
        kind = data.get("kind")
        if kind not in (None, FACE_TRACK_SET_KIND):
            raise ValueError(f"Unsupported face-track payload kind: {kind!r}")
        tracks = []
        for raw_track in data.get("tracks") or []:
            observations = tuple(
                FaceObservation(
                    frame_index=int(raw["frame_index"]),
                    time_s=float(raw["time_s"]),
                    bbox_xyxy=tuple(float(value) for value in raw["bbox_xyxy"]),
                    det_score=float(raw["det_score"]),
                    landmarks=(
                        tuple(tuple(float(v) for v in point) for point in raw["landmarks"])
                        if raw.get("landmarks")
                        else None
                    ),
                )
                for raw in raw_track.get("observations") or []
            )
            if not observations:
                continue
            tracks.append(
                FaceTrack(
                    track_id=str(raw_track["track_id"]),
                    observations=observations,
                    embedding=_embedding_from_payload(raw_track.get("embedding")),
                )
            )
        model_payload = data.get("model")
        model = None
        if isinstance(model_payload, dict) and "backend" in model_payload:
            model = DiarizationModelInfo(
                backend=str(model_payload["backend"]),
                model_id=str(model_payload.get("model_id") or model_payload["backend"]),
                revision=model_payload.get("revision"),
            )
        video_path = data.get("video_path")
        return cls(
            video_id=str(data["video_id"]),
            tracks=tuple(tracks),
            fps=float(data["fps"]),
            duration_s=float(data["duration_s"]),
            model=model,
            video_path=str(resolve_data_path(video_path)) if video_path else None,
        )

    def save(self, destination: str | Path) -> Path:
        """Persist tracks as JSON and return the path."""
        return _write_json(destination, self.to_dict(), default_stem=self.video_id)

    @classmethod
    def load(cls, path: str | Path) -> FaceTrackSet:
        """Load tracks from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ASDFrameScore:
    """Active-speaker score for one face track at one timestamp."""

    time_s: float
    track_id: str
    score: float
    speaking: bool

    def __post_init__(self) -> None:
        _validate_timestamp(self.time_s, "time_s")
        _validate_non_empty_string(self.track_id, "track_id")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a number")
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if not isinstance(self.speaking, bool):
            raise TypeError("speaking must be a bool")


@dataclass
class ASDResult:
    """Per-track active-speaker scores for one video."""

    video_id: str
    scores: tuple[ASDFrameScore, ...]
    active_threshold: float
    model: DiarizationModelInfo | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.video_id, "video_id")
        if not isinstance(self.scores, tuple):
            raise TypeError("scores must be a tuple")
        if isinstance(self.active_threshold, bool) or not isinstance(
            self.active_threshold, (int, float)
        ):
            raise TypeError("active_threshold must be a number")
        if not isfinite(self.active_threshold):
            raise ValueError("active_threshold must be finite")

    def scores_during(self, start_s: float, end_s: float) -> tuple[ASDFrameScore, ...]:
        """Return scores whose timestamps fall inside ``[start_s, end_s]``."""
        return tuple(
            item for item in self.scores if start_s <= item.time_s <= end_s
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot."""
        return {
            "kind": ASD_RESULT_KIND,
            "schema_version": AV_SCHEMA_VERSION,
            "video_id": self.video_id,
            "active_threshold": self.active_threshold,
            "model": asdict(self.model) if self.model else None,
            "scores": [asdict(item) for item in self.scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ASDResult:
        """Restore ASD scores from JSON."""
        if not isinstance(data, dict):
            raise TypeError("ASDResult payload must be an object")
        kind = data.get("kind")
        if kind not in (None, ASD_RESULT_KIND):
            raise ValueError(f"Unsupported ASD payload kind: {kind!r}")
        model_payload = data.get("model")
        model = None
        if isinstance(model_payload, dict) and "backend" in model_payload:
            model = DiarizationModelInfo(
                backend=str(model_payload["backend"]),
                model_id=str(model_payload.get("model_id") or model_payload["backend"]),
                revision=model_payload.get("revision"),
            )
        scores = tuple(
            ASDFrameScore(
                time_s=float(raw["time_s"]),
                track_id=str(raw["track_id"]),
                score=float(raw["score"]),
                speaking=bool(raw["speaking"]),
            )
            for raw in data.get("scores") or []
        )
        return cls(
            video_id=str(data["video_id"]),
            scores=scores,
            active_threshold=float(data["active_threshold"]),
            model=model,
        )

    def save(self, destination: str | Path) -> Path:
        """Persist ASD scores as JSON and return the path."""
        return _write_json(destination, self.to_dict(), default_stem=self.video_id)

    @classmethod
    def load(cls, path: str | Path) -> ASDResult:
        """Load ASD scores from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class SpeakerEntity:
    """A multimodal person identity: voice clips plus face images.

    Clips and face images are the source of truth. Embeddings are computed at
    verification time so an entity remains model-independent.
    """

    name: str
    clip_paths: list[Path]
    face_paths: list[Path]
    created_at: str
    entity_dir: Path
    updated_at: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    channel_url: str | None = None


@dataclass(frozen=True)
class AVSegmentDecision:
    """Audiovisual purity decision for one candidate speech interval."""

    schema_version: str
    audio_id: str
    video_id: str
    entity_name: str
    speaker_id: str
    start_s: float
    end_s: float
    refined_start_s: float
    refined_end_s: float
    decision: AVDecision
    visual_status: VisualStatus
    overlap_duration_s: float
    overlap_ratio: float
    windows: tuple[SpeakerSimilarityWindow, ...]
    face_similarity: float | None = None
    voice_similarity: float | None = None
    asd_purity: float | None = None
    associated_track_id: str | None = None
    reason: str | None = None
    model: DiarizationModelInfo | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.schema_version, "schema_version")
        _validate_non_empty_string(self.audio_id, "audio_id")
        _validate_non_empty_string(self.video_id, "video_id")
        _validate_non_empty_string(self.entity_name, "entity_name")
        _validate_non_empty_string(self.speaker_id, "speaker_id")
        _validate_timestamp(self.start_s, "start_s")
        _validate_timestamp(self.end_s, "end_s")
        _validate_timestamp(self.refined_start_s, "refined_start_s")
        _validate_timestamp(self.refined_end_s, "refined_end_s")
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        if self.refined_end_s <= self.refined_start_s:
            raise ValueError("refined_end_s must be greater than refined_start_s")
        if self.decision not in {"accept", "audio_only", "reject", "error"}:
            raise ValueError(
                "decision must be 'accept', 'audio_only', 'reject', or 'error'"
            )
        if self.visual_status not in {
            "visual_verified",
            "audio_only",
            "visual_conflict",
        }:
            raise ValueError("visual_status is invalid")
        if self.decision == "accept" and self.reason is not None:
            raise ValueError("accepted results cannot contain a reason")
        if self.decision == "accept" and self.visual_status != "visual_verified":
            raise ValueError("accept requires visual_status='visual_verified'")
        if self.decision == "audio_only" and self.visual_status != "audio_only":
            raise ValueError("audio_only decisions require visual_status='audio_only'")
        if self.decision != "accept" and self.decision != "audio_only":
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("rejected and error results require a reason")
        if self.decision == "error":
            if not isinstance(self.error, str) or not self.error:
                raise ValueError("error results require a non-empty error message")
        elif self.error is not None:
            raise ValueError("only error results may contain an error message")
        _validate_timestamp(self.overlap_duration_s, "overlap_duration_s")
        _validate_unit_interval(self.overlap_ratio, "overlap_ratio")
        if not isinstance(self.windows, tuple) or not all(
            isinstance(window, SpeakerSimilarityWindow) for window in self.windows
        ):
            raise TypeError("windows must be a tuple of SpeakerSimilarityWindow")
        for field_name in ("face_similarity", "voice_similarity"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number or None")
            if not isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"{field_name} must be between -1 and 1")
        if self.asd_purity is not None:
            _validate_unit_interval(self.asd_purity, "asd_purity")

    @property
    def passed(self) -> bool:
        """Whether this candidate is high-trust audiovisual TTS material."""
        return self.decision == "accept"

    @property
    def admitted(self) -> bool:
        """Whether the candidate may enter a corpus, including audio-only."""
        return self.decision in {"accept", "audio_only"}

    @property
    def duration_s(self) -> float:
        """Original candidate duration in seconds."""
        return self.end_s - self.start_s


@dataclass
class AVVerificationResult:
    """Audiovisual decisions for every candidate turn of one source."""

    schema_version: str
    audio_id: str
    video_id: str
    entity_name: str
    decisions: list[AVSegmentDecision]
    source_audio: Audio | None = None
    source_video: Video | None = None
    model: DiarizationModelInfo | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    channel_url: str | None = None
    result_id: str = field(default_factory=lambda: f"av_{uuid.uuid4().hex}")
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.schema_version, "schema_version")
        _validate_non_empty_string(self.audio_id, "audio_id")
        _validate_non_empty_string(self.video_id, "video_id")
        _validate_non_empty_string(self.entity_name, "entity_name")
        _validate_non_empty_string(self.result_id, "result_id")
        if not isinstance(self.decisions, list):
            raise TypeError("decisions must be a list")
        if (
            isinstance(self.created_at, bool)
            or not isinstance(self.created_at, (int, float))
            or not isfinite(self.created_at)
            or self.created_at < 0
        ):
            raise ValueError("created_at must be a finite non-negative timestamp")

    @property
    def accepted(self) -> list[AVSegmentDecision]:
        """High-trust audiovisual accepts."""
        return [item for item in self.decisions if item.decision == "accept"]

    @property
    def audio_only(self) -> list[AVSegmentDecision]:
        """Audio-verified segments without usable visual evidence."""
        return [item for item in self.decisions if item.decision == "audio_only"]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""
        source_audio = self.source_audio.metadata() if self.source_audio else None
        source_video = self.source_video.metadata() if self.source_video else None
        return {
            "kind": AV_RESULT_KIND,
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "created_at": self.created_at,
            "audio_id": self.audio_id,
            "video_id": self.video_id,
            "entity_name": self.entity_name,
            "source_audio": source_audio,
            "source_video": source_video,
            "decisions": [
                {
                    **asdict(decision),
                    "duration_s": decision.duration_s,
                    "passed": decision.passed,
                    "admitted": decision.admitted,
                }
                for decision in self.decisions
            ],
            "model": asdict(self.model) if self.model else None,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "channel_url": self.channel_url,
            "summary": {
                "candidate_count": len(self.decisions),
                "accept_count": len(self.accepted),
                "audio_only_count": len(self.audio_only),
                "reject_count": sum(
                    1 for item in self.decisions if item.decision == "reject"
                ),
                "error_count": sum(
                    1 for item in self.decisions if item.decision == "error"
                ),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AVVerificationResult:
        """Restore a verification result from JSON."""
        if not isinstance(data, dict):
            raise TypeError("AVVerificationResult payload must be an object")
        kind = data.get("kind")
        if kind not in (None, AV_RESULT_KIND):
            raise ValueError(f"Unsupported AV payload kind: {kind!r}")

        source_audio = _audio_from_payload(data.get("source_audio"))
        source_video = _video_from_payload(data.get("source_video"))
        model_payload = data.get("model")
        model = None
        if isinstance(model_payload, dict) and "backend" in model_payload:
            model = DiarizationModelInfo(
                backend=str(model_payload["backend"]),
                model_id=str(model_payload.get("model_id") or model_payload["backend"]),
                revision=model_payload.get("revision"),
            )
        decisions = [
            _decision_from_payload(raw)
            for raw in data.get("decisions") or []
        ]
        return cls(
            schema_version=str(data.get("schema_version", AV_SCHEMA_VERSION)),
            audio_id=str(data["audio_id"]),
            video_id=str(data["video_id"]),
            entity_name=str(data["entity_name"]),
            decisions=decisions,
            source_audio=source_audio,
            source_video=source_video,
            model=model,
            channel_id=data.get("channel_id"),
            channel_name=data.get("channel_name"),
            channel_url=data.get("channel_url"),
            result_id=str(data.get("result_id") or f"av_{uuid.uuid4().hex}"),
            created_at=float(data.get("created_at", time.time())),
        )

    def save(self, destination: str | Path) -> Path:
        """Persist this result as JSON and return its path."""
        payload = self.to_dict()
        source_audio = payload.get("source_audio")
        if isinstance(source_audio, dict) and source_audio.get("path"):
            source_audio["path"] = portable_data_path(source_audio["path"])
        source_video = payload.get("source_video")
        if isinstance(source_video, dict) and source_video.get("path"):
            source_video["path"] = portable_data_path(source_video["path"])
        return _write_json(destination, payload, default_stem=self.result_id)

    @classmethod
    def load(cls, path: str | Path) -> AVVerificationResult:
        """Load a verification result JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _audio_from_payload(raw: Any) -> Audio | None:
    if not isinstance(raw, dict) or not raw.get("path") or not raw.get("source_id"):
        return None
    return Audio(
        path=resolve_data_path(raw["path"]),
        source_id=str(raw["source_id"]),
        title=raw.get("title"),
        sample_rate=raw.get("sample_rate"),
        duration_s=raw.get("duration_s"),
        channels=raw.get("channels"),
        format=str(raw.get("format", "wav")),
        native_sample_rate=raw.get("native_sample_rate"),
        history=tuple(raw.get("history") or ()),
        source_url=raw.get("source_url"),
        channel_id=raw.get("channel_id"),
        channel_name=raw.get("channel_name"),
        channel_url=raw.get("channel_url"),
    )


def _video_from_payload(raw: Any) -> Video | None:
    if not isinstance(raw, dict) or not raw.get("path") or not raw.get("source_id"):
        return None
    return Video(
        path=resolve_data_path(raw["path"]),
        source_id=str(raw["source_id"]),
        title=raw.get("title"),
        duration_s=raw.get("duration_s"),
        fps=raw.get("fps"),
        width=raw.get("width"),
        height=raw.get("height"),
        format=str(raw.get("format", "mp4")),
        source_url=raw.get("source_url"),
        channel_id=raw.get("channel_id"),
        channel_name=raw.get("channel_name"),
        channel_url=raw.get("channel_url"),
    )


def _decision_from_payload(raw: Any) -> AVSegmentDecision:
    if not isinstance(raw, dict):
        raise TypeError("each decision must be an object")
    windows = tuple(
        SpeakerSimilarityWindow(
            start_s=float(window["start_s"]),
            end_s=float(window["end_s"]),
            similarity=float(window["similarity"]),
        )
        for window in raw.get("windows") or []
    )
    model = None
    model_payload = raw.get("model")
    if isinstance(model_payload, dict) and "backend" in model_payload:
        model = DiarizationModelInfo(
            backend=str(model_payload["backend"]),
            model_id=str(model_payload.get("model_id") or model_payload["backend"]),
            revision=model_payload.get("revision"),
        )
    payload = {
        key: raw[key]
        for key in (
            "schema_version",
            "audio_id",
            "video_id",
            "entity_name",
            "speaker_id",
            "start_s",
            "end_s",
            "refined_start_s",
            "refined_end_s",
            "decision",
            "visual_status",
            "overlap_duration_s",
            "overlap_ratio",
            "face_similarity",
            "voice_similarity",
            "asd_purity",
            "associated_track_id",
            "reason",
            "error",
        )
        if key in raw
    }
    return AVSegmentDecision(
        **payload,
        windows=windows,
        model=model,
    )


def _write_json(destination: str | Path, payload: dict[str, Any], *, default_stem: str) -> Path:
    destination_path = Path(destination)
    path = (
        destination_path
        if destination_path.suffix.lower() == ".json"
        else destination_path / f"{default_stem}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    return path
