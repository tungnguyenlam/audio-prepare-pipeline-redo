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
PROFILE_SCHEMA_VERSION = "1.0"
MIN_EMBEDDING_DURATION_S = 0.15

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class SpeakerVerifierError(RuntimeError):
    """Raised when enrollment, profile access, or scoring fails."""


@dataclass
class SpeakerProfile:
    """A named target speaker backed by reference clips on disk.

    Clips are the source of truth; embeddings are recomputed at scoring time
    so profiles stay valid across embedding-model changes and machines.
    """

    name: str
    clip_paths: list[Path]
    created_at: str
    profile_dir: Path


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
    ) -> SpeakerProfile:
        """Create a speaker profile from manually cut reference clips.

        Each clip must contain only the target speaker. Two or three clips of
        roughly 10 seconds from different videos work well.

        Args:
            name: Profile name (sanitized to a filesystem-safe identifier).
            clips: Single-speaker reference clips (file-backed).
            overwrite: Replace an existing profile with the same name.

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
            "name": safe_name,
            "created_at": created_at,
            "clips": clip_names,
        }
        (profile_dir / "profile.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return SpeakerProfile(
            name=safe_name,
            clip_paths=[clips_dir / clip_name for clip_name in clip_names],
            created_at=created_at,
            profile_dir=profile_dir,
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
            clip_names = list(manifest["clips"])
            created_at = str(manifest.get("created_at", ""))
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
            name=safe_name,
            clip_paths=clip_paths,
            created_at=created_at,
            profile_dir=profile_dir,
        )

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
        )

    def _profile_centroid(self, profile: SpeakerProfile) -> Any:
        """L2-normalized mean embedding of the profile clips."""
        import numpy as np

        vectors = []
        for clip_path in profile.clip_paths:
            try:
                vectors.append(self._embed(clip_path))
            except Exception:
                continue
        if not vectors:
            raise SpeakerVerifierError(
                f"No profile clip could be embedded for {profile.name!r}"
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
        from pyannote.core import Segment

        if start_s is None:
            raw = self._inference(str(path))
        else:
            raw = self._inference.crop(str(path), Segment(start_s, end_s))

        vector = np.asarray(raw, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise SpeakerVerifierError(f"Zero embedding for {path}")
        return vector / norm
