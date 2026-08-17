"""ModelScope 3D-Speaker (speakerlab) diarization backend.

Cascaded audio-only pipeline from
https://github.com/modelscope/3D-Speaker : FSMN VAD, CAM++ speaker embeddings,
then spectral clustering, with optional pyannote overlap refinement. Requires
the isolated environment pinned in ``requirements-3dspeaker.txt``. The
``speakerlab`` sources are shallow-cloned into ``.data/3d-speaker`` on first
load when missing (``speakerlab`` is not published as a package).
"""

from __future__ import annotations

import gc
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.base.model import ManagedModel
from src.diarization.BaseDiarizer import BaseDiarizer
from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    Speaker,
    SpeakerTurn,
)
from src.utils.audio_utils import normalize_wav, probe_wav
from src.utils.AudioClass import Audio

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_ID = "iic/speech_campplus_sv_zh_en_16k-common_advanced"
DEFAULT_VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
DEFAULT_MODEL_ID = f"{DEFAULT_VAD_MODEL_ID}+{DEFAULT_EMBEDDING_MODEL_ID}"
SAMPLE_RATE = 16000
THREEDSPEAKER_GIT_URL = "https://github.com/modelscope/3D-Speaker.git"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class ThreeDSpeakerDiarizer(BaseDiarizer, ManagedModel):
    """Diarize audio with ModelScope 3D-Speaker's ``Diarization3Dspeaker``.

    Speaker labels from clustering are converted to result-local ``spk_NN``
    identifiers. Those identifiers are meaningful only within one result.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        num_speakers: int | None = None,
        include_overlap: bool = False,
        token: str | None = None,
        model_cache_dir: str | Path | None = None,
        speakerlab_root: str | Path | None = None,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        """Initialize the 3D-Speaker diarizer.

        Args:
            device: Compute device (``"auto"``, ``"cuda"``, ``"cpu"``, etc.).
            num_speakers: Exact speaker count when known in advance.
            include_overlap: Enable pyannote segmentation-based overlap
                refinement. Requires a Hugging Face token.
            token: Hugging Face token used when ``include_overlap`` is True.
                Falls back to ``HF_TOKEN`` when unset.
            model_cache_dir: Directory for ModelScope pretrained downloads.
                Defaults to ``.data/modelscope`` under the repository root.
            speakerlab_root: Path to a cloned
                https://github.com/modelscope/3D-Speaker checkout. Defaults to
                ``THREEDSPEAKER_ROOT`` or ``.data/3d-speaker``. Missing checkouts
                are shallow-cloned on first ``load()``.
            ffmpeg_bin: ``ffmpeg`` executable used to normalize input audio.
        """
        ManagedModel.__init__(self)
        if num_speakers is not None and num_speakers < 1:
            raise ValueError("num_speakers must be at least 1")
        if include_overlap:
            resolved_token = token if token is not None else os.getenv("HF_TOKEN")
            if not resolved_token:
                raise ValueError(
                    "include_overlap=True requires token=... or HF_TOKEN in the "
                    "environment"
                )
            token = resolved_token

        self.device = str(device)
        self.num_speakers = num_speakers
        self.include_overlap = bool(include_overlap)
        self.token = token
        self.model_cache_dir = (
            Path(model_cache_dir).expanduser()
            if model_cache_dir is not None
            else _REPO_ROOT / ".data" / "modelscope"
        )
        configured_root = (
            speakerlab_root
            if speakerlab_root is not None
            else os.getenv("THREEDSPEAKER_ROOT")
        )
        self.speakerlab_root = (
            Path(configured_root).expanduser()
            if configured_root is not None
            else _REPO_ROOT / ".data" / "3d-speaker"
        )
        self.ffmpeg_bin = ffmpeg_bin

        self._pipeline: Any | None = None
        self._target_device: Any | None = None

    @staticmethod
    def resolve_speaker_settings(
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> int | None:
        """Map UI speaker bounds onto an oracle speaker count when possible.

        Args:
            num_speakers: Exact speaker count when known.
            min_speakers: Optional lower bound from the caller UI.
            max_speakers: Optional upper bound from the caller UI.

        Returns:
            Oracle speaker count when an exact value is known, otherwise
            ``None`` so clustering estimates the count.
        """
        if num_speakers is not None:
            return int(num_speakers)
        if (
            min_speakers is not None
            and max_speakers is not None
            and min_speakers == max_speakers
        ):
            return int(min_speakers)
        return None

    def _ensure_speakerlab_path(self) -> Path:
        root = self.speakerlab_root.expanduser()
        if not (root / "speakerlab").is_dir():
            self._clone_speakerlab(root)
        root = root.resolve()
        if not (root / "speakerlab").is_dir():
            raise RuntimeError(
                f"3D-Speaker checkout is incomplete at {root}. Expected a "
                "speakerlab/ package directory."
            )
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return root

    @staticmethod
    def _clone_speakerlab(root: Path) -> None:
        """Shallow-clone modelscope/3D-Speaker into ``root`` when missing."""
        if shutil.which("git") is None:
            raise RuntimeError(
                f"3D-Speaker checkout not found at {root} and git is not "
                f"available to clone {THREEDSPEAKER_GIT_URL}. Install git or "
                "set THREEDSPEAKER_ROOT to an existing checkout."
            )

        root.parent.mkdir(parents=True, exist_ok=True)
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                f"3D-Speaker checkout not found at {root}: the directory "
                "exists but does not contain speakerlab/. Remove it or set "
                "THREEDSPEAKER_ROOT to a valid checkout."
            )

        logger.info("Cloning 3D-Speaker into %s", root)
        staging = root.with_name(f"{root.name}.partial")
        if staging.exists():
            shutil.rmtree(staging)
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                THREEDSPEAKER_GIT_URL,
                str(staging),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(
                f"Failed to clone 3D-Speaker into {root}: "
                f"{detail[:2000] or 'no output'}"
            )
        if root.exists():
            root.rmdir()
        staging.rename(root)

    def _resolve_device(self, torch: Any) -> Any:
        if self.device != "auto":
            target = torch.device(self.device)
            if target.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if target.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS was requested but PyTorch cannot initialize it"
                )
            return target

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self) -> None:
        """Load CAM++ embeddings, FSMN VAD, and optional overlap segmentation."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "3D-Speaker diarizer dependencies are unavailable. Install the "
                "pinned requirements-3dspeaker.txt dependencies in an isolated "
                "environment."
            ) from exc

        self._ensure_speakerlab_path()
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MODELSCOPE_CACHE", str(self.model_cache_dir))

        try:
            from speakerlab.bin.infer_diarization import Diarization3Dspeaker
        except ImportError as exc:
            raise RuntimeError(
                "speakerlab is unavailable. Clone "
                "https://github.com/modelscope/3D-Speaker into .data/3d-speaker "
                "or set THREEDSPEAKER_ROOT, and install "
                "requirements-3dspeaker.txt."
            ) from exc

        target_device = self._resolve_device(torch)
        pipeline = Diarization3Dspeaker(
            device=target_device,
            include_overlap=self.include_overlap,
            hf_access_token=self.token,
            speaker_num=self.num_speakers,
            model_cache_dir=str(self.model_cache_dir),
        )
        self._target_device = target_device
        self._pipeline = pipeline

    def _unload(self) -> None:
        """Release VAD, embedding, clustering, and accelerator caches."""
        self._pipeline = None
        self._target_device = None
        gc.collect()

        try:
            import torch
        except ImportError:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def diarize(self, audio: Audio) -> DiarizationResult:
        """Diarize ``audio`` while preserving the schema 1.0 result contract.

        Args:
            audio: File-backed audio item to diarize.

        Returns:
            Speaker identities and turns with local ``spk_NN`` labels.

        Raises:
            RuntimeError: If the model is not loaded or inference fails.
            FileNotFoundError: If ``audio.path`` does not exist.
            ValueError: If the normalized audio duration is empty.
        """
        if not self.is_loaded or self._pipeline is None:
            raise RuntimeError(
                "ThreeDSpeakerDiarizer is not loaded. Call load() before "
                "diarize(), or use it as a context manager."
            )

        source_path = Path(audio.path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {source_path}")

        with tempfile.TemporaryDirectory(prefix="3d-speaker-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            normalized_path = temp_dir / "normalized.wav"
            normalize_wav(
                source_path,
                normalized_path,
                sample_rate=SAMPLE_RATE,
                channels=1,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            if probe_wav(normalized_path)[1] <= 0:
                raise ValueError(f"Audio source is empty: {source_path}")

            try:
                segments = self._pipeline(
                    str(normalized_path),
                    wav_fs=SAMPLE_RATE,
                    speaker_num=self.num_speakers,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"3D-Speaker diarization failed: {exc}"
                ) from exc

        turns, speakers = self._turns_from_segments(segments)
        return DiarizationResult(
            schema_version="1.0",
            audio_id=audio.source_id,
            speakers=speakers,
            turns=turns,
            model=DiarizationModelInfo(
                backend="3d-speaker",
                model_id=DEFAULT_MODEL_ID,
            ),
        )

    @staticmethod
    def _turns_from_segments(
        segments: Any,
    ) -> tuple[list[SpeakerTurn], list[Speaker]]:
        """Convert ``[[start, end, speaker_id], ...]`` into schema turns."""
        if segments is None:
            return [], []

        raw_turns: list[tuple[str, float, float]] = []
        for item in segments:
            try:
                start_s = float(item[0])
                end_s = float(item[1])
                label = str(item[2])
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(start_s) or not math.isfinite(end_s):
                continue
            if end_s <= start_s:
                continue
            raw_turns.append((label, max(0.0, start_s), end_s))

        raw_turns.sort(key=lambda item: (item[1], item[2], item[0]))
        label_to_speaker_id: dict[str, str] = {}
        speakers: list[Speaker] = []
        turns: list[SpeakerTurn] = []
        for label, start_s, end_s in raw_turns:
            speaker_id = label_to_speaker_id.get(label)
            if speaker_id is None:
                speaker_id = f"spk_{len(label_to_speaker_id):02d}"
                label_to_speaker_id[label] = speaker_id
                speakers.append(Speaker(speaker_id=speaker_id))
            turns.append(
                SpeakerTurn(
                    speaker_id=speaker_id,
                    start_s=float(start_s),
                    end_s=float(end_s),
                    confidence=None,
                )
            )
        return turns, speakers
