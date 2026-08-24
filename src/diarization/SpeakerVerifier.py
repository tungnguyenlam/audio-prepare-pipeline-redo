"""Target-speaker enrollment and verification-based segment filtering.

Enroll a target speaker from a few manually cut single-speaker reference
clips, then score diarization turns against the enrollment centroid with a
speaker-verification embedding model. Filtering is precision-first: keep only
segments whose embedding is close enough to the target, optionally excluding
anything that overlaps another speaker's turns.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.base.model import ManagedModel
from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    ScoredSegment,
    TargetSpeakerResult,
)
from src.utils.AudioClass import Audio, _sanitize_filename_component

DEFAULT_EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
PROFILE_SCHEMA_VERSION = "2.0"
MIN_EMBEDDING_DURATION_S = 0.15

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class SpeakerVerifierError(RuntimeError):
    """Raised when enrollment, profile access, or scoring fails."""


@dataclass
class SpeakerProfile:
    """A named target speaker backed by reference clips on disk.

    Clips are the source of truth; each consuming pipeline computes its own
    enrollment representation so profiles remain model-independent.
    """

    name: str
    clip_paths: list[Path]
    created_at: str
    profile_dir: Path
    updated_at: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    channel_url: str | None = None


class SpeakerVerifier(ManagedModel):
    """Enroll target speakers and score diarization turns against them.

    Profiles live under ``profiles_dir`` (default
    ``.data/speaker_profiles/<name>/``) as copied reference clips plus a
    ``profile.json`` manifest. ``enroll``, ``load_profile``, ``list_profiles``
    and ``delete_profile`` work without loading the model; ``score`` requires
    ``load()`` or a ``with`` block. ``filter`` is a pure post-processing step.

    Example::

        verifier = SpeakerVerifier()
        profile = verifier.enroll("khanh_vy", [clip1, clip2, clip3])
        with verifier:
            scored = verifier.score(audio, diarization_result, profile)
        kept = SpeakerVerifier.filter(scored, threshold=0.6)
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
        device: str = "auto",
        token: str | None = None,
        profiles_dir: str | Path | None = None,
    ) -> None:
        """Initialize the verifier.

        Args:
            model_id: Hugging Face repository ID of the pyannote embedding
                model used for verification.
            device: Compute device (``"auto"``, ``"cuda"``, ``"cpu"``, etc.).
            token: Optional Hugging Face access token (or ``HF_TOKEN`` env).
            profiles_dir: Directory holding speaker profiles. Defaults to
                ``.data/speaker_profiles`` under the repository root.
        """
        ManagedModel.__init__(self)
        self.model_id = model_id
        self.device = str(device)
        self.token = token
        self.profiles_dir = (
            Path(profiles_dir).expanduser()
            if profiles_dir is not None
            else _REPO_ROOT / ".data" / "speaker_profiles"
        )
        self._inference: Any | None = None

    # ------------------------------------------------------------------
    # Profile management (no model required)
    # ------------------------------------------------------------------

    def enroll(
        self,
        name: str,
        clips: list[Audio],
        *,
        overwrite: bool = False,
        channel_id: str | None = None,
        channel_name: str | None = None,
        channel_url: str | None = None,
    ) -> SpeakerProfile:
        """Create a speaker profile from manually cut reference clips.

        Each clip must contain only the target speaker. Profiles are global;
        channel fields are retained only as optional provenance metadata.

        Args:
            name: Profile name (sanitized to a filesystem-safe identifier).
            clips: Single-speaker reference clips (file-backed).
            overwrite: Replace an existing profile with the same name.
            channel_id: Optional source-channel provenance.
            channel_name: Optional human-readable source-channel provenance.
            channel_url: Optional canonical source-channel provenance.

        Returns:
            The stored profile.

        Raises:
            SpeakerVerifierError: If ``clips`` is empty, the name is invalid,
                or the profile exists and ``overwrite`` is False.
            FileNotFoundError: If a clip file does not exist.
        """
        safe_name = _sanitize_filename_component(name)
        if not safe_name:
            raise SpeakerVerifierError(f"Invalid profile name: {name!r}")
        display_name = str(name).strip()
        if not clips:
            raise SpeakerVerifierError("enroll requires at least one clip")
        for clip in clips:
            if not Path(clip.path).is_file():
                raise FileNotFoundError(f"Clip file does not exist: {clip.path}")

        profile_dir = self.profiles_dir / safe_name
        if profile_dir.exists():
            if not overwrite:
                raise SpeakerVerifierError(
                    f"Profile {safe_name!r} already exists. "
                    "Pass overwrite=True to replace it."
                )
            shutil.rmtree(profile_dir)

        clips_dir = profile_dir / "clips"
        clips_dir.mkdir(parents=True)
        clip_names: list[str] = []
        for index, clip in enumerate(clips):
            suffix = Path(clip.path).suffix or ".wav"
            clip_name = f"clip_{index:02d}{suffix}"
            shutil.copy2(clip.path, clips_dir / clip_name)
            clip_names.append(clip_name)

        created_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": display_name,
            "created_at": created_at,
            "updated_at": created_at,
            "clips": clip_names,
            "channel_id": str(channel_id) if channel_id else None,
            "channel_name": str(channel_name) if channel_name else None,
            "channel_url": str(channel_url) if channel_url else None,
        }
        (profile_dir / "profile.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return SpeakerProfile(
            name=display_name,
            clip_paths=[clips_dir / clip_name for clip_name in clip_names],
            created_at=created_at,
            profile_dir=profile_dir,
            updated_at=created_at,
            channel_id=str(channel_id) if channel_id else None,
            channel_name=str(channel_name) if channel_name else None,
            channel_url=str(channel_url) if channel_url else None,
        )

    def load_profile(self, name: str) -> SpeakerProfile:
        """Load a stored profile by name.

        Raises:
            SpeakerVerifierError: If the profile or any of its clips is
                missing or its manifest is malformed.
        """
        safe_name = _sanitize_filename_component(name)
        profile_dir = self.profiles_dir / safe_name
        manifest_path = profile_dir / "profile.json"
        if not manifest_path.is_file():
            raise SpeakerVerifierError(
                f"Profile {safe_name!r} not found under {self.profiles_dir}"
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            display_name = str(manifest.get("name") or safe_name).strip()
            clip_names = list(manifest["clips"])
            created_at = str(manifest.get("created_at", ""))
            updated_at = str(manifest.get("updated_at") or created_at)
            channel_id = manifest.get("channel_id")
            channel_name = manifest.get("channel_name")
            channel_url = manifest.get("channel_url")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SpeakerVerifierError(
                f"Malformed profile manifest: {manifest_path}"
            ) from exc

        clip_paths = [profile_dir / "clips" / clip_name for clip_name in clip_names]
        missing = [str(path) for path in clip_paths if not path.is_file()]
        if not clip_paths or missing:
            raise SpeakerVerifierError(
                f"Profile {safe_name!r} has missing clips: {missing or 'none listed'}"
            )

        return SpeakerProfile(
            name=display_name,
            clip_paths=clip_paths,
            created_at=created_at,
            profile_dir=profile_dir,
            updated_at=updated_at,
            channel_id=str(channel_id) if channel_id else None,
            channel_name=str(channel_name) if channel_name else None,
            channel_url=str(channel_url) if channel_url else None,
        )

    def add_clips(self, name: str, clips: list[Audio]) -> SpeakerProfile:
        """Append clean reference clips to an existing global profile.

        Args:
            name: Existing speaker profile name.
            clips: Additional single-speaker reference clips.

        Returns:
            The updated profile.

        Raises:
            SpeakerVerifierError: If the profile is missing or ``clips`` is empty.
            FileNotFoundError: If an input clip file does not exist.
        """
        if not clips:
            raise SpeakerVerifierError("add_clips requires at least one clip")
        profile = self.load_profile(name)
        for clip in clips:
            if not Path(clip.path).is_file():
                raise FileNotFoundError(f"Clip file does not exist: {clip.path}")

        manifest_path = profile.profile_dir / "profile.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        clip_names = list(manifest["clips"])
        clips_dir = profile.profile_dir / "clips"
        next_index = 0
        for clip in clips:
            suffix = Path(clip.path).suffix or ".wav"
            while (clips_dir / f"clip_{next_index:02d}{suffix}").exists():
                next_index += 1
            clip_name = f"clip_{next_index:02d}{suffix}"
            shutil.copy2(clip.path, clips_dir / clip_name)
            clip_names.append(clip_name)
            next_index += 1

        manifest["schema_version"] = PROFILE_SCHEMA_VERSION
        manifest["clips"] = clip_names
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.load_profile(profile.name)

    def remove_clip(self, name: str, clip_name: str) -> SpeakerProfile:
        """Remove one reference clip while keeping at least one enrollment clip.

        Args:
            name: Existing speaker profile name.
            clip_name: Basename of a clip listed by the profile.

        Returns:
            The updated profile.

        Raises:
            SpeakerVerifierError: If the clip is missing or is the last clip.
        """
        profile = self.load_profile(name)
        safe_clip_name = Path(clip_name).name
        if safe_clip_name != clip_name:
            raise SpeakerVerifierError(f"Invalid clip name: {clip_name!r}")

        manifest_path = profile.profile_dir / "profile.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        clip_names = list(manifest["clips"])
        if safe_clip_name not in clip_names:
            raise SpeakerVerifierError(
                f"Clip {safe_clip_name!r} is not part of profile {profile.name!r}"
            )
        if len(clip_names) == 1:
            raise SpeakerVerifierError(
                "A speaker profile must keep at least one clip; delete the profile instead"
            )

        clip_path = profile.profile_dir / "clips" / safe_clip_name
        clip_path.unlink()
        manifest["schema_version"] = PROFILE_SCHEMA_VERSION
        manifest["clips"] = [item for item in clip_names if item != safe_clip_name]
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.load_profile(profile.name)

    def list_profiles(self) -> list[str]:
        """Return the names of all stored profiles, sorted."""
        if not self.profiles_dir.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.profiles_dir.iterdir()
            if (entry / "profile.json").is_file()
        )

    def delete_profile(self, name: str) -> None:
        """Delete a stored profile and its clips.

        Raises:
            SpeakerVerifierError: If the profile does not exist.
        """
        safe_name = _sanitize_filename_component(name)
        profile_dir = self.profiles_dir / safe_name
        if not (profile_dir / "profile.json").is_file():
            raise SpeakerVerifierError(
                f"Profile {safe_name!r} not found under {self.profiles_dir}"
            )
        shutil.rmtree(profile_dir)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the pyannote embedding model onto the target device."""
        import torch
        from pyannote.audio import Inference, Model

        token = self.token if self.token is not None else os.getenv("HF_TOKEN")
        model = Model.from_pretrained(self.model_id, token=token)
        if model is None:
            raise SpeakerVerifierError(
                f"Embedding model could not be loaded: {self.model_id!r}"
            )

        if self.device == "auto":
            target_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            target_device = torch.device(self.device)

        inference = Inference(model, window="whole")
        if target_device.type != "cpu":
            inference.to(target_device)
        self._inference = inference

    def _unload(self) -> None:
        """Release the embedding model and cached CUDA allocations."""
        self._inference = None
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Scoring and filtering
    # ------------------------------------------------------------------

    def score(
        self,
        audio: Audio,
        result: DiarizationResult,
        profile: SpeakerProfile,
    ) -> TargetSpeakerResult:
        """Score every diarization turn against the profile centroid.

        Each turn is embedded independently and compared to the L2-normalized
        mean embedding of the profile clips via cosine similarity. Turns too
        short to embed (or failing embedding) get similarity ``-1.0`` so they
        are never selected. ``overlaps_other_speaker`` is set for turns that
        intersect a turn of a different speaker.

        Args:
            audio: The audio item the diarization result belongs to.
            result: Diarization turns from any backend.
            profile: Enrolled target speaker.

        Returns:
            All turns with similarity scores (no filtering applied).

        Raises:
            RuntimeError: If the model is not loaded.
            FileNotFoundError: If the audio file does not exist.
            SpeakerVerifierError: If no profile clip can be embedded.
        """
        if not self.is_loaded or self._inference is None:
            raise RuntimeError(
                "SpeakerVerifier is not loaded. Call load() before score(), "
                "or use it as a context manager."
            )
        source_path = Path(audio.path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {source_path}")

        centroid = self._profile_centroid(profile)

        import numpy as np

        segments: list[ScoredSegment] = []
        for index, turn in enumerate(result.turns):
            overlaps = any(
                other.speaker_id != turn.speaker_id
                and other.start_s < turn.end_s
                and other.end_s > turn.start_s
                for other in result.turns
            )

            similarity = -1.0
            if turn.end_s - turn.start_s >= MIN_EMBEDDING_DURATION_S:
                try:
                    vector = self._embed(source_path, turn.start_s, turn.end_s)
                    similarity = float(np.clip(np.dot(centroid, vector), -1.0, 1.0))
                except Exception:
                    similarity = -1.0

            segments.append(
                ScoredSegment(
                    speaker_id=turn.speaker_id,
                    start_s=turn.start_s,
                    end_s=turn.end_s,
                    similarity=similarity,
                    overlaps_other_speaker=overlaps,
                )
            )

        return TargetSpeakerResult(
            schema_version="1.0",
            audio_id=audio.source_id,
            profile_name=profile.name,
            segments=segments,
            model=DiarizationModelInfo(
                backend="pyannote-embedding",
                model_id=self.model_id,
            ),
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
        )

    @staticmethod
    def filter(
        scored: TargetSpeakerResult,
        *,
        threshold: float,
        min_duration_s: float = 1.5,
        exclude_overlap: bool = True,
    ) -> TargetSpeakerResult:
        """Keep only segments confidently matching the target speaker.

        Pure post-processing: no model needed, so different thresholds can be
        tried cheaply on one ``score`` result.

        Args:
            scored: Output of :meth:`score`.
            threshold: Minimum cosine similarity to keep a segment.
            min_duration_s: Minimum segment length in seconds. Short segments
                have unreliable embeddings.
            exclude_overlap: Drop segments overlapping another speaker's turn.

        Returns:
            A new result containing only the kept segments.
        """
        kept = [
            segment
            for segment in scored.segments
            if segment.similarity >= threshold
            and segment.duration_s >= min_duration_s
            and not (exclude_overlap and segment.overlaps_other_speaker)
        ]
        return TargetSpeakerResult(
            schema_version=scored.schema_version,
            audio_id=scored.audio_id,
            profile_name=scored.profile_name,
            segments=kept,
            model=scored.model,
            channel_id=scored.channel_id,
            channel_name=scored.channel_name,
            channel_url=scored.channel_url,
        )

    def _profile_centroid(self, profile: SpeakerProfile) -> Any:
        """L2-normalized mean embedding of the profile clips."""
        import numpy as np

        vectors = []
        failures = []
        for clip_path in profile.clip_paths:
            try:
                vectors.append(self._embed(clip_path))
            except Exception as exc:
                failures.append(
                    f"{clip_path.name}: {type(exc).__name__}: {exc}"
                )
        if not vectors:
            details = "; ".join(failures)
            raise SpeakerVerifierError(
                f"None of {len(profile.clip_paths)} profile clips could be embedded "
                f"for {profile.name!r}. {details}"
            )

        centroid = np.mean(np.stack(vectors), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            raise SpeakerVerifierError(
                f"Profile {profile.name!r} produced a zero centroid embedding"
            )
        return centroid / norm

    def _embed(
        self,
        path: Path,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> Any:
        """L2-normalized embedding of a whole file or a time slice of it."""
        import numpy as np
        import soundfile as sf
        import torch

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        info = sf.info(str(path))
        start_frame = 0
        stop_frame = info.frames
        if start_s is not None:
            if end_s is None or end_s <= start_s:
                raise SpeakerVerifierError(
                    f"Invalid embedding interval for {path}: {start_s}–{end_s}"
                )
            start_frame = max(0, round(start_s * info.samplerate))
            stop_frame = min(info.frames, round(end_s * info.samplerate))

        if stop_frame <= start_frame:
            raise SpeakerVerifierError(
                f"Embedding interval is empty for {path}: {start_s}–{end_s}"
            )

        waveform, sample_rate = sf.read(
            str(path),
            start=start_frame,
            stop=stop_frame,
            dtype="float32",
            always_2d=True,
        )
        if waveform.shape[0] == 0:
            raise SpeakerVerifierError(f"Cannot embed empty audio file: {path}")

        if waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1, keepdims=True)

        raw = self._inference(
            {
                "waveform": torch.from_numpy(waveform.T.copy()),
                "sample_rate": int(sample_rate),
            }
        )

        vector = np.asarray(raw, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            raise SpeakerVerifierError(f"Invalid embedding for {path}")
        return vector / norm
