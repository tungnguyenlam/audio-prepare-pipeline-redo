"""Speaker Diarization implementation with PyAnnote Audio 3.1 / 4.0.

Stage 4 in SOTA Audio Processing Pipeline:
Identifies speaker turns, clusters voices, and filters out overlapping speech.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import torch

from src.diarization.base import BaseDiarizer, SpeakerTurn, refine_and_merge_turns

logger = logging.getLogger(__name__)

# Configure local project HF cache directory to avoid root permission issues
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
LOCAL_HF_CACHE.mkdir(parents=True, exist_ok=True)
TORCH_CACHE = PROJECT_ROOT / ".cache" / "torch"
TORCH_CACHE.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["TORCH_HOME"] = str(TORCH_CACHE)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


class PyannoteDiarizer(BaseDiarizer):
    """PyAnnote 3.1/4.0 SOTA Neural Diarization Pipeline."""

    def __init__(
        self,
        auth_token: Optional[str] = None,
        device: str = "cuda",
        model_name: str = "pyannote/speaker-diarization-3.1",
    ):
        self.auth_token = auth_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.model_name = model_name
        self._pipeline = None

    def _init_pipeline(self, auth_token: Optional[str] = None):
        """Lazy loader for PyAnnote pipeline."""
        token = auth_token or self.auth_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        # If pipeline already loaded with same token, reuse
        if self._pipeline is not None and (not auth_token or auth_token == self.auth_token):
            return self._pipeline

        try:
            from pyannote.audio import Pipeline

            logger.info("Initializing PyAnnote Pipeline (%s) on %s...", self.model_name, self.device)
            
            # PyAnnote 3.1 / 4.0 uses token=..., older versions used use_auth_token=...
            try:
                pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=token,
                    cache_dir=str(LOCAL_HF_CACHE / "hub"),
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=token,
                    cache_dir=str(LOCAL_HF_CACHE / "hub"),
                )

            if pipeline is None:
                raise ValueError(
                    f"Không thể tải mô hình PyAnnote '{self.model_name}'. "
                    "Hãy đảm bảo bạn đã cung cấp Hugging Face Access Token hợp lệ và đã bấm 'Accept conditions' tại "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1 và https://huggingface.co/pyannote/segmentation-3.0"
                )

            # Move pipeline to torch device (using 'cuda:0' or 'cpu' to prevent unpacking errors)
            target_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
            pipeline.to(torch.device(target_device))
            self._pipeline = pipeline
            self.auth_token = token
            return self._pipeline
        except Exception as exc:
            logger.error("Failed to load PyAnnote pipeline: %s", exc)
            raise RuntimeError(f"Lỗi khởi tạo PyAnnote Diarization: {exc}") from exc

    def check_status(self) -> dict:
        """Check if pyannote is ready to run."""
        try:
            import pyannote.audio
            has_token = bool(self.auth_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
            return {
                "available": True,
                "version": getattr(pyannote.audio, "__version__", "4.x"),
                "device": self.device,
                "has_token": has_token,
                "model": self.model_name,
            }
        except ImportError as e:
            return {
                "available": False,
                "message": f"Chưa cài pyannote.audio: {e}",
                "model": self.model_name,
            }

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        filter_overlap: bool = True,
        min_duration_s: float = 0.5,
        auth_token: Optional[str] = None,
    ) -> List[SpeakerTurn]:
        """Perform speaker diarization on audio file.

        Args:
            audio_path: Path to .wav audio
            num_speakers: Exact number of speakers if known
            min_speakers: Lower bound on speakers
            max_speakers: Upper bound on speakers
            filter_overlap: Whether to flag/filter overlapping segments
            min_duration_s: Minimum duration threshold for a turn
            auth_token: Optional override for HuggingFace token

        Returns:
            List of SpeakerTurn objects
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file không tồn tại: {audio_path}")

        pipeline = self._init_pipeline(auth_token=auth_token)

        # Prepare kwargs for pipeline
        params = {}
        if num_speakers is not None and num_speakers > 0:
            params["num_speakers"] = int(num_speakers)
        else:
            if min_speakers is not None and min_speakers > 0:
                params["min_speakers"] = int(min_speakers)
            if max_speakers is not None and max_speakers > 0:
                params["max_speakers"] = int(max_speakers)

        logger.info("Executing PyAnnote Diarization on %s with params: %s", audio_path.name, params)
        diarization_out = pipeline(str(audio_path), **params)

        # PyAnnote 4.x returns DiarizeOutput(speaker_diarization, exclusive_speaker_diarization, ...)
        # PyAnnote 3.x returns Annotation directly
        if hasattr(diarization_out, "speaker_diarization"):
            annotation = (
                diarization_out.exclusive_speaker_diarization
                if (filter_overlap and getattr(diarization_out, "exclusive_speaker_diarization", None) is not None)
                else diarization_out.speaker_diarization
            )
        else:
            annotation = diarization_out

        raw_turns: List[SpeakerTurn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            raw_turns.append(SpeakerTurn(
                start_s=float(turn.start),
                end_s=float(turn.end),
                speaker_id=str(speaker),
                is_overlap=False
            ))

        # Detect overlaps: segments that collide in time with other speakers
        turns_with_overlap: List[SpeakerTurn] = []
        for i, t in enumerate(raw_turns):
            dur = t.end_s - t.start_s
            if dur < min_duration_s:
                continue

            # Check overlap with any other turn
            is_ov = False
            for j, o in enumerate(raw_turns):
                if i != j and o.speaker_id != t.speaker_id:
                    # Overlap interval check
                    overlap_start = max(t.start_s, o.start_s)
                    overlap_end = min(t.end_s, o.end_s)
                    if overlap_end - overlap_start > 0.15:  # Overlap greater than 150ms
                        is_ov = True
                        break

            t.is_overlap = is_ov
            if filter_overlap and is_ov:
                # Skip overlap turns if strict filter is requested
                continue
            turns_with_overlap.append(t)

        # Sort turns by start time
        turns_with_overlap.sort(key=lambda x: x.start_s)

        # Apply boundary collar trimming and merge consecutive turns of the same speaker (gap <= 1.0s)
        final_turns = refine_and_merge_turns(
            turns=turns_with_overlap,
            max_merge_gap_s=1.0,
            boundary_collar_s=0.08,
            min_duration_s=min_duration_s,
        )

        logger.info(
            "PyAnnote detected %d raw turns (%d after overlap filter, %d after merging & trimming).",
            len(raw_turns),
            len(turns_with_overlap),
            len(final_turns)
        )
        return final_turns if final_turns else turns_with_overlap
