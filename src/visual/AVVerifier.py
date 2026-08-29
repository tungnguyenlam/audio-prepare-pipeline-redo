"""Multimodal speaker-entity enrollment and audiovisual purity gates.

Callers compose this with ``FaceAnalyzer``, ``LightASD``, an existing
``SpeakerVerifier``, and a ``DiarizationResult``. This module does not crawl,
separate, or diarize.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_paths import DATA_DIR
from src.diarization.SpeakerVerifier import (
    DEFAULT_MAX_OVERLAP_DURATION_S,
    DEFAULT_PURITY_WINDOW_DURATION_S,
    DEFAULT_PURITY_WINDOW_HOP_S,
    SpeakerProfile,
    SpeakerVerifier,
)
from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    SpeakerPurityResult,
    SpeakerTurn,
)
from src.utils.AudioClass import Audio, _sanitize_filename_component
from src.visual.FaceAnalyzer import FaceAnalyzer
from src.visual.Video import Video
from src.visual.schemas import (
    ASDResult,
    AV_SCHEMA_VERSION,
    AVSegmentDecision,
    AVVerificationResult,
    ENTITY_SCHEMA_VERSION,
    FaceTrack,
    FaceTrackSet,
    SPEAKER_ENTITY_KIND,
    SpeakerEntity,
)

DEFAULT_FACE_SIMILARITY_THRESHOLD = 0.50
DEFAULT_ASD_PURITY_MIN = 0.95
DEFAULT_TRANSITION_MARGIN_S = 0.08
DEFAULT_EXPAND_FACE_THRESHOLD = 0.70
DEFAULT_EXPAND_VOICE_THRESHOLD = 0.75


class AVVerifierError(RuntimeError):
    """Raised when entity enrollment or audiovisual verification fails."""


class AVVerifier:
    """Enroll face+voice speaker entities and gate diarization turns.

    Entity management copies reference files and does not need models.
    ``verify()`` requires a loaded ``SpeakerVerifier`` and ``FaceAnalyzer``.
    Active-speaker scores come from a prior ``LightASD.score()`` call.

    Example::

        verifier = AVVerifier()
        entity = verifier.enroll("host", voice_clips=[clip], face_images=[face])
        with face_analyzer, asd, speaker_verifier:
            tracks = face_analyzer.analyze(video)
            asd_scores = asd.score(video, audio, tracks)
            result = verifier.verify(
                audio, diarization, entity, tracks, asd_scores,
                speaker_verifier, face_analyzer,
                similarity_threshold=0.6,
            )
    """

    def __init__(self, entities_dir: str | Path | None = None) -> None:
        """Initialize the verifier.

        Args:
            entities_dir: Directory holding speaker entities. Defaults to
                ``.data/speaker_entities``.
        """
        self.entities_dir = (
            Path(entities_dir).expanduser()
            if entities_dir is not None
            else DATA_DIR / "speaker_entities"
        )

    def enroll(
        self,
        name: str,
        *,
        voice_clips: list[Audio],
        face_images: list[str | Path],
        overwrite: bool = False,
        channel_id: str | None = None,
        channel_name: str | None = None,
        channel_url: str | None = None,
    ) -> SpeakerEntity:
        """Create a multimodal entity from clean voice clips and face images.

        Does not require models to be loaded. Each clip should contain only the
        target speaker; each image should show a clearly visible target face.

        Args:
            name: Entity name (sanitized to a filesystem-safe identifier).
            voice_clips: Single-speaker reference clips.
            face_images: High-quality face stills (JPEG/PNG).
            overwrite: Replace an existing entity with the same name.
            channel_id: Optional source-channel provenance.
            channel_name: Optional human-readable source-channel provenance.
            channel_url: Optional canonical source-channel provenance.

        Returns:
            The stored entity.

        Raises:
            AVVerifierError: If inputs are empty, the name is invalid, or the
                entity exists and ``overwrite`` is False.
            FileNotFoundError: If a clip or face file is missing.
        """
        safe_name = _sanitize_filename_component(name)
        if not safe_name:
            raise AVVerifierError(f"Invalid entity name: {name!r}")
        if not voice_clips:
            raise AVVerifierError("enroll requires at least one voice clip")
        if not face_images:
            raise AVVerifierError("enroll requires at least one face image")

        clip_paths = [Path(clip.path) for clip in voice_clips]
        face_paths = [Path(path) for path in face_images]
        for path in clip_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Voice clip does not exist: {path}")
        for path in face_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Face image does not exist: {path}")

        entity_dir = self.entities_dir / safe_name
        if entity_dir.exists():
            if not overwrite:
                raise AVVerifierError(
                    f"Entity {safe_name!r} already exists. "
                    "Pass overwrite=True to replace it."
                )
            shutil.rmtree(entity_dir)

        clips_dir = entity_dir / "clips"
        faces_dir = entity_dir / "faces"
        clips_dir.mkdir(parents=True)
        faces_dir.mkdir(parents=True)
        stored_clips = _copy_numbered(clip_paths, clips_dir, prefix="clip")
        stored_faces = _copy_numbered(face_paths, faces_dir, prefix="face")
        created_at = datetime.now(timezone.utc).isoformat()
        _write_entity_manifest(
            entity_dir,
            name=str(name).strip(),
            created_at=created_at,
            updated_at=created_at,
            clips=stored_clips,
            faces=stored_faces,
            channel_id=channel_id,
            channel_name=channel_name,
            channel_url=channel_url,
        )
        return self.load_entity(name)

    def load_entity(self, name: str) -> SpeakerEntity:
        """Load a stored entity by name.

        Raises:
            AVVerifierError: If the entity or any of its files is missing.
        """
        safe_name = _sanitize_filename_component(name)
        entity_dir = self.entities_dir / safe_name
        manifest_path = entity_dir / "profile.json"
        if not manifest_path.is_file():
            raise AVVerifierError(
                f"Entity {safe_name!r} not found under {self.entities_dir}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            display_name = str(manifest.get("name") or safe_name).strip()
            clip_names = list(manifest["clips"])
            face_names = list(manifest["faces"])
            created_at = str(manifest.get("created_at", ""))
            updated_at = str(manifest.get("updated_at") or created_at)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AVVerifierError(
                f"Malformed entity manifest: {manifest_path}"
            ) from exc

        clip_paths = [entity_dir / "clips" / item for item in clip_names]
        face_paths = [entity_dir / "faces" / item for item in face_names]
        missing = [
            str(path) for path in clip_paths + face_paths if not path.is_file()
        ]
        if not clip_paths or not face_paths or missing:
            raise AVVerifierError(
                f"Entity {safe_name!r} has missing files: {missing or 'none listed'}"
            )
        return SpeakerEntity(
            name=display_name,
            clip_paths=clip_paths,
            face_paths=face_paths,
            created_at=created_at,
            entity_dir=entity_dir,
            updated_at=updated_at,
            channel_id=_optional_str(manifest.get("channel_id")),
            channel_name=_optional_str(manifest.get("channel_name")),
            channel_url=_optional_str(manifest.get("channel_url")),
        )

    def list_entities(self) -> list[str]:
        """Return the names of all stored entities, sorted."""
        if not self.entities_dir.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.entities_dir.iterdir()
            if (entry / "profile.json").is_file()
        )

    def delete_entity(self, name: str) -> None:
        """Delete a stored entity and its reference files.

        Raises:
            AVVerifierError: If the entity does not exist.
        """
        safe_name = _sanitize_filename_component(name)
        entity_dir = self.entities_dir / safe_name
        if not (entity_dir / "profile.json").is_file():
            raise AVVerifierError(
                f"Entity {safe_name!r} not found under {self.entities_dir}"
            )
        shutil.rmtree(entity_dir)

    def add_evidence(
        self,
        name: str,
        *,
        voice_clips: list[Audio] | None = None,
        face_images: list[str | Path] | None = None,
    ) -> SpeakerEntity:
        """Append high-confidence voice clips or face images to an entity.

        Only add evidence that already passed strict audiovisual gates.
        This is the Phase-3 enrollment expansion hook.

        Raises:
            AVVerifierError: If the entity is missing or no files are given.
            FileNotFoundError: If an input file does not exist.
        """
        clips = list(voice_clips or [])
        faces = [Path(path) for path in (face_images or [])]
        if not clips and not faces:
            raise AVVerifierError("add_evidence requires voice clips or face images")
        entity = self.load_entity(name)
        for clip in clips:
            if not Path(clip.path).is_file():
                raise FileNotFoundError(f"Voice clip does not exist: {clip.path}")
        for path in faces:
            if not path.is_file():
                raise FileNotFoundError(f"Face image does not exist: {path}")

        clips_dir = entity.entity_dir / "clips"
        faces_dir = entity.entity_dir / "faces"
        stored_clips = [path.name for path in entity.clip_paths]
        stored_faces = [path.name for path in entity.face_paths]
        if clips:
            stored_clips.extend(
                _copy_numbered([Path(clip.path) for clip in clips], clips_dir, prefix="clip")
            )
        if faces:
            stored_faces.extend(_copy_numbered(faces, faces_dir, prefix="face"))
        _write_entity_manifest(
            entity.entity_dir,
            name=entity.name,
            created_at=entity.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            clips=stored_clips,
            faces=stored_faces,
            channel_id=entity.channel_id,
            channel_name=entity.channel_name,
            channel_url=entity.channel_url,
        )
        return self.load_entity(entity.name)

    def verify(
        self,
        audio: Audio,
        result: DiarizationResult,
        entity: SpeakerEntity,
        tracks: FaceTrackSet,
        asd: ASDResult,
        speaker_verifier: SpeakerVerifier,
        face_analyzer: FaceAnalyzer,
        *,
        candidates: list[SpeakerTurn] | None = None,
        similarity_threshold: float,
        face_similarity_threshold: float = DEFAULT_FACE_SIMILARITY_THRESHOLD,
        min_candidate_duration_s: float = 1.5,
        max_overlap_duration_s: float = DEFAULT_MAX_OVERLAP_DURATION_S,
        window_duration_s: float = DEFAULT_PURITY_WINDOW_DURATION_S,
        window_hop_s: float = DEFAULT_PURITY_WINDOW_HOP_S,
        asd_purity_min: float = DEFAULT_ASD_PURITY_MIN,
        transition_margin_s: float = DEFAULT_TRANSITION_MARGIN_S,
        video: Video | None = None,
    ) -> AVVerificationResult:
        """Gate diarization turns with independent face, voice, and ASD evidence.

        Audio purity and sliding voice-identity windows are delegated to
        ``SpeakerVerifier.verify_purity``. Visual gates then classify each
        remaining candidate as ``accept``, ``audio_only``, or ``reject``.
        Vision never overrides an audio rejection.

        Args:
            audio: Source audio the diarization result belongs to.
            result: Diarization turns (overlap authority).
            entity: Enrolled multimodal target person.
            tracks: Face tracks from ``FaceAnalyzer.analyze``.
            asd: Active-speaker scores from ``LightASD.score``.
            speaker_verifier: Loaded voice-embedding verifier.
            face_analyzer: Loaded face embedder, used for the entity centroid.
            candidates: Optional subset of ``result.turns``.
            similarity_threshold: Minimum cosine for every voice window.
            face_similarity_threshold: Minimum cosine between the speaking
                face track and the entity face centroid.
            min_candidate_duration_s: Shorter turns are rejected.
            max_overlap_duration_s: Maximum tolerated other-speaker overlap.
            window_duration_s: Sliding voice-identity window length.
            window_hop_s: Hop between voice windows.
            asd_purity_min: Minimum fraction of speech frames with exactly one
                active visible speaker.
            transition_margin_s: Trim applied around uncertain ASD edges.
            video: Optional source video copied into the result snapshot.

        Returns:
            One decision per candidate, plus corpus-level summary counts.

        Raises:
            RuntimeError: If a required model is not loaded.
            FileNotFoundError: If the audio file is missing.
            AVVerifierError: If the entity face images cannot be embedded.
        """
        if not speaker_verifier.is_loaded:
            raise RuntimeError(
                "SpeakerVerifier is not loaded. Call load() before verify()."
            )
        if not face_analyzer.is_loaded:
            raise RuntimeError(
                "FaceAnalyzer is not loaded. Call load() before verify()."
            )
        if not Path(audio.path).is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio.path}")
        if asd_purity_min < 0 or asd_purity_min > 1:
            raise ValueError("asd_purity_min must be between 0 and 1")
        if not -1 <= face_similarity_threshold <= 1:
            raise ValueError("face_similarity_threshold must be between -1 and 1")
        if transition_margin_s < 0:
            raise ValueError("transition_margin_s must be non-negative")

        import numpy as np

        try:
            face_centroid = face_analyzer.embed_images(entity.face_paths)
        except Exception as exc:
            raise AVVerifierError(
                f"Could not embed face enrollment for {entity.name!r}: {exc}"
            ) from exc

        profile = SpeakerProfile(
            name=entity.name,
            clip_paths=entity.clip_paths,
            created_at=entity.created_at,
            profile_dir=entity.entity_dir,
            updated_at=entity.updated_at,
            channel_id=entity.channel_id,
            channel_name=entity.channel_name,
            channel_url=entity.channel_url,
        )
        purity_results = speaker_verifier.verify_purity(
            result,
            profile,
            candidates=candidates,
            similarity_threshold=similarity_threshold,
            min_candidate_duration_s=min_candidate_duration_s,
            max_overlap_duration_s=max_overlap_duration_s,
            window_duration_s=window_duration_s,
            window_hop_s=window_hop_s,
        )
        model_info = DiarizationModelInfo(
            backend="av-identity",
            model_id="late-fusion",
        )
        video_id = tracks.video_id
        decisions: list[AVSegmentDecision] = []
        for purity in purity_results:
            visual = _visual_evidence(
                tracks,
                asd,
                face_centroid=face_centroid,
                start_s=purity.start_s,
                end_s=purity.end_s,
                face_similarity_threshold=face_similarity_threshold,
                asd_purity_min=asd_purity_min,
            )
            refined_start_s, refined_end_s = _refine_bounds(
                purity.start_s,
                purity.end_s,
                asd=asd,
                associated_track_id=visual["associated_track_id"],
                margin_s=transition_margin_s,
            )
            decision, visual_status, reason = _fuse_decision(purity, visual)
            voice_similarity = (
                min(window.similarity for window in purity.windows)
                if purity.windows
                else None
            )
            decisions.append(
                AVSegmentDecision(
                    schema_version=AV_SCHEMA_VERSION,
                    audio_id=audio.source_id,
                    video_id=video_id,
                    entity_name=entity.name,
                    speaker_id=purity.speaker_id,
                    start_s=purity.start_s,
                    end_s=purity.end_s,
                    refined_start_s=refined_start_s,
                    refined_end_s=refined_end_s,
                    decision=decision,
                    visual_status=visual_status,
                    overlap_duration_s=purity.overlap_duration_s,
                    overlap_ratio=purity.overlap_ratio,
                    windows=purity.windows,
                    face_similarity=visual["face_similarity"],
                    voice_similarity=voice_similarity,
                    asd_purity=visual["asd_purity"],
                    associated_track_id=visual["associated_track_id"],
                    reason=reason,
                    model=model_info,
                    error=purity.error if decision == "error" else None,
                )
            )
        return AVVerificationResult(
            schema_version=AV_SCHEMA_VERSION,
            audio_id=audio.source_id,
            video_id=video_id,
            entity_name=entity.name,
            decisions=decisions,
            source_audio=audio,
            source_video=video,
            model=model_info,
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
        )


def _visual_evidence(
    tracks: FaceTrackSet,
    asd: ASDResult,
    *,
    face_centroid: Any,
    start_s: float,
    end_s: float,
    face_similarity_threshold: float,
    asd_purity_min: float,
) -> dict[str, Any]:
    import numpy as np

    interval_scores = asd.scores_during(start_s, end_s)
    times = sorted({item.time_s for item in interval_scores})
    asd_purity: float | None = None
    if times:
        exact_one = 0
        for time_s in times:
            active = sum(
                1
                for item in interval_scores
                if item.time_s == time_s and item.speaking
            )
            if active == 1:
                exact_one += 1
        asd_purity = exact_one / len(times)

    mean_by_track: dict[str, float] = {}
    for item in interval_scores:
        mean_by_track.setdefault(item.track_id, [])
    grouped: dict[str, list[float]] = {}
    for item in interval_scores:
        grouped.setdefault(item.track_id, []).append(item.score)
    mean_by_track = {
        track_id: float(sum(values) / len(values))
        for track_id, values in grouped.items()
        if values
    }
    associated_track_id = None
    associated_mean = None
    if mean_by_track:
        associated_track_id = max(mean_by_track, key=mean_by_track.get)
        associated_mean = mean_by_track[associated_track_id]
        if associated_mean < asd.active_threshold:
            associated_track_id = None

    face_similarity = None
    face_match = False
    associated_track: FaceTrack | None = None
    if associated_track_id is not None:
        associated_track = next(
            (track for track in tracks.tracks if track.track_id == associated_track_id),
            None,
        )
        if associated_track is not None and associated_track.embedding is not None:
            face_similarity = float(
                np.clip(
                    np.dot(np.asarray(associated_track.embedding), face_centroid),
                    -1.0,
                    1.0,
                )
            )
            face_match = face_similarity >= face_similarity_threshold

    speaking_identities_match = True
    for track in tracks.tracks_during(start_s, end_s):
        track_mean = mean_by_track.get(track.track_id)
        if track_mean is None or track_mean < asd.active_threshold:
            continue
        if track.embedding is None:
            continue
        similarity = float(
            np.clip(np.dot(np.asarray(track.embedding), face_centroid), -1.0, 1.0)
        )
        if associated_track is not None and track.track_id != associated_track.track_id:
            if similarity >= face_similarity_threshold and not face_match:
                speaking_identities_match = False
            if similarity < face_similarity_threshold and face_match:
                speaking_identities_match = False

    no_visible_speaker = associated_track_id is None
    return {
        "asd_purity": asd_purity,
        "associated_track_id": associated_track_id,
        "face_similarity": face_similarity,
        "face_match": face_match,
        "speaking_identities_match": speaking_identities_match,
        "no_visible_speaker": no_visible_speaker,
        "asd_purity_ok": asd_purity is None or asd_purity >= asd_purity_min,
        "asd_purity_min": asd_purity_min,
    }


def _fuse_decision(
    purity: SpeakerPurityResult,
    visual: dict[str, Any],
) -> tuple[str, str, str | None]:
    """Combine audio purity with visual evidence. Audio rejection wins."""
    if purity.decision == "error":
        return "error", "audio_only", purity.reason
    if purity.decision == "reject":
        return "reject", "audio_only", purity.reason

    voice_match = True
    if visual["no_visible_speaker"]:
        return "audio_only", "audio_only", None

    if not visual["asd_purity_ok"]:
        return "reject", "visual_conflict", "asd_purity_below_threshold"
    if not visual["speaking_identities_match"]:
        return "reject", "visual_conflict", "face_identity_mismatch"

    face_match = bool(visual["face_match"])
    if face_match and voice_match:
        return "accept", "visual_verified", None
    if (not face_match) and voice_match:
        return "reject", "visual_conflict", "visual_conflict"
    return "reject", "visual_conflict", "visual_conflict"


def _refine_bounds(
    start_s: float,
    end_s: float,
    *,
    asd: ASDResult,
    associated_track_id: str | None,
    margin_s: float,
) -> tuple[float, float]:
    if associated_track_id is None or margin_s <= 0:
        return start_s, end_s
    track_scores = [
        item
        for item in asd.scores_during(start_s, end_s)
        if item.track_id == associated_track_id
    ]
    if len(track_scores) < 2:
        return start_s, end_s
    refined_start = start_s
    refined_end = end_s
    first = track_scores[0]
    last = track_scores[-1]
    if not first.speaking:
        refined_start = min(end_s - 1e-3, start_s + margin_s)
    if not last.speaking:
        refined_end = max(refined_start + 1e-3, end_s - margin_s)
    return refined_start, refined_end


def _copy_numbered(
    sources: list[Path],
    dest_dir: Path,
    *,
    prefix: str,
) -> list[str]:
    stored: list[str] = []
    next_index = 0
    for source in sources:
        suffix = source.suffix or ".bin"
        while (dest_dir / f"{prefix}_{next_index:02d}{suffix}").exists():
            next_index += 1
        name = f"{prefix}_{next_index:02d}{suffix}"
        shutil.copy2(source, dest_dir / name)
        stored.append(name)
        next_index += 1
    return stored


def _write_entity_manifest(
    entity_dir: Path,
    *,
    name: str,
    created_at: str,
    updated_at: str,
    clips: list[str],
    faces: list[str],
    channel_id: str | None,
    channel_name: str | None,
    channel_url: str | None,
) -> None:
    manifest = {
        "schema_version": ENTITY_SCHEMA_VERSION,
        "kind": SPEAKER_ENTITY_KIND,
        "name": name,
        "created_at": created_at,
        "updated_at": updated_at,
        "clips": clips,
        "faces": faces,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "channel_url": channel_url,
    }
    (entity_dir / "profile.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
