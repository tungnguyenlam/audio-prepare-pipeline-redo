"""NVIDIA Sortformer speaker diarization backend with long-audio stitching."""

from __future__ import annotations

import gc
import importlib
import inspect
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.base.model import ManagedModel
from src.diarization.BaseDiarizer import BaseDiarizer
from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    Speaker,
    SpeakerTurn,
)
from src.utils.audio_utils import normalize_wav
from src.utils.AudioClass import Audio

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "nvidia/diar_sortformer_4spk-v1"
DEFAULT_MODEL_FILENAME = "diar_sortformer_4spk-v1.nemo"
DEFAULT_MODEL_REVISION = "f059506485424eb68a90a7af84c8e63e67f381fd"
FRAME_DURATION_S = 0.08
MAX_LOCAL_SPEAKERS = 4
DEFAULT_ONSET = 0.74
DEFAULT_OFFSET = 0.64
DEFAULT_PAD_ONSET_S = 0.12
DEFAULT_PAD_OFFSET_S = 0.20


@dataclass(frozen=True)
class _LocalTurn:
    speaker_index: int
    start_s: float
    end_s: float


@dataclass
class _WindowResult:
    start_s: float
    end_s: float
    waveform: np.ndarray
    probabilities: np.ndarray
    turns: list[_LocalTurn]
    local_to_global: dict[int, int] = field(default_factory=dict)
    embeddings: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class _SpeakerProfile:
    centroid: np.ndarray
    observations: int = 1

    def update(self, embedding: np.ndarray) -> None:
        combined = self.centroid * self.observations + embedding
        norm = float(np.linalg.norm(combined))
        if norm > 0:
            self.centroid = combined / norm
            self.observations += 1


class SortformerDiarizer(BaseDiarizer, ManagedModel):
    """Diarize audio with NVIDIA's four-channel offline Sortformer model.

    Long recordings are processed in overlapping windows. Speaker slots are
    aligned first from activity in the shared region and then, when enabled,
    from TitaNet speaker embeddings. The four-speaker restriction therefore
    applies to each inference window, not to the final result.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        *,
        model_filename: str = DEFAULT_MODEL_FILENAME,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        token: str | None = None,
        batch_size: int = 1,
        window_duration_s: float = 360.0,
        overlap_duration_s: float = 60.0,
        oom_retry_window_s: float | None = 180.0,
        embedding_model_id: str = "titanet_large",
        enable_speaker_similarity: bool = True,
        embedding_similarity_threshold: float = 0.70,
        overlap_match_threshold: float = 0.35,
        onset: float = DEFAULT_ONSET,
        offset: float = DEFAULT_OFFSET,
        pad_onset_s: float = DEFAULT_PAD_ONSET_S,
        pad_offset_s: float = DEFAULT_PAD_OFFSET_S,
        min_duration_on_s: float = 0.10,
        min_duration_off_s: float = 0.15,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        ManagedModel.__init__(self)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be an integer of at least 1")
        if window_duration_s <= 0:
            raise ValueError("window_duration_s must be positive")
        if overlap_duration_s < 0 or overlap_duration_s >= window_duration_s:
            raise ValueError(
                "overlap_duration_s must be non-negative and smaller than "
                "window_duration_s"
            )
        if oom_retry_window_s is not None and oom_retry_window_s <= overlap_duration_s:
            raise ValueError(
                "oom_retry_window_s must be larger than overlap_duration_s"
            )
        for name, value in {
            "embedding_similarity_threshold": embedding_similarity_threshold,
            "overlap_match_threshold": overlap_match_threshold,
            "onset": onset,
            "offset": offset,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if onset < offset:
            raise ValueError(
                "onset must be greater than or equal to offset for hysteresis"
            )
        for name, value in {
            "pad_onset_s": pad_onset_s,
            "pad_offset_s": pad_offset_s,
            "min_duration_on_s": min_duration_on_s,
            "min_duration_off_s": min_duration_off_s,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        self.model_id = model_id
        self.revision = revision
        self.model_filename = model_filename
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        self.device = str(device)
        self.token = token
        self.batch_size = batch_size
        self.window_duration_s = float(window_duration_s)
        self.overlap_duration_s = float(overlap_duration_s)
        self.oom_retry_window_s = (
            float(oom_retry_window_s) if oom_retry_window_s is not None else None
        )
        self.embedding_model_id = embedding_model_id
        self.enable_speaker_similarity = bool(enable_speaker_similarity)
        self.embedding_similarity_threshold = float(embedding_similarity_threshold)
        self.overlap_match_threshold = float(overlap_match_threshold)
        self.onset = float(onset)
        self.offset = float(offset)
        self.pad_onset_s = float(pad_onset_s)
        self.pad_offset_s = float(pad_offset_s)
        self.min_duration_on_s = float(min_duration_on_s)
        self.min_duration_off_s = float(min_duration_off_s)
        self.ffmpeg_bin = ffmpeg_bin

        self._model: Any | None = None
        self._embedding_model: Any | None = None
        self._target_device: Any | None = None

    def _load(self) -> None:
        """Download and restore the pinned Sortformer checkpoint."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Sortformer dependencies are unavailable. Install the pinned "
                "requirements-sortformer.txt dependencies in an isolated NeMo "
                "environment."
            ) from exc

        target_device = self._resolve_device(torch)
        if target_device.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        try:
            from huggingface_hub import hf_hub_download

            nemo_models = importlib.import_module("nemo.collections.asr.models")
            sortformer_model_type = nemo_models.SortformerEncLabelModel
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Sortformer dependencies are unavailable. Install the pinned "
                "requirements-sortformer.txt dependencies in an isolated NeMo "
                "environment."
            ) from exc

        if self.checkpoint_path is not None:
            checkpoint = self.checkpoint_path.resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Sortformer checkpoint does not exist: {checkpoint}"
                )
        else:
            token = self.token if self.token is not None else os.getenv("HF_TOKEN")
            checkpoint = Path(
                hf_hub_download(
                    repo_id=self.model_id,
                    filename=self.model_filename,
                    revision=self.revision,
                    token=token,
                )
            )

        model = sortformer_model_type.restore_from(
            restore_path=str(checkpoint),
            map_location=target_device,
            strict=False,
        )
        model = model.to(target_device).eval()

        self._target_device = target_device
        self._model = model

    def _resolve_device(self, torch: Any) -> Any:
        if self.device != "auto":
            target = torch.device(self.device)
            if target.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if target.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS was requested but PyTorch cannot initialize it. The installed "
                    "wheel may include MPS while the current macOS/PyTorch combination "
                    "still reports the device unavailable."
                )
            return target

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _unload(self) -> None:
        """Release Sortformer, TitaNet, and accelerator caches."""
        self._embedding_model = None
        self._model = None
        self._target_device = None
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _ensure_embedding_model(self) -> Any:
        if self._embedding_model is not None:
            return self._embedding_model
        if self._target_device is None:
            raise RuntimeError("SortformerDiarizer has no configured target device")

        try:
            nemo_models = importlib.import_module("nemo.collections.asr.models")
            embedding_model_type = nemo_models.EncDecSpeakerLabelModel
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "NeMo speaker-recognition support is required for cross-window "
                "speaker similarity"
            ) from exc

        logger.info("Loading speaker embedding model %s", self.embedding_model_id)
        model = embedding_model_type.from_pretrained(
            model_name=self.embedding_model_id,
            map_location=self._target_device,
        )
        self._embedding_model = model.to(self._target_device).eval()
        return self._embedding_model

    def diarize(
        self,
        audio: Audio,
        *,
        enrollment_name: str | None = None,
        enrollment_clips: list[str | Path] | None = None,
    ) -> DiarizationResult:
        """Diarize audio with an optional pre-inference speaker enrollment.

        The enrollment clips are embedded with this pipeline's TitaNet model
        before target-audio inference. Their centroid seeds global speaker zero
        during window stitching, so the known identity participates in speaker
        assignment instead of being scored after diarization.

        Args:
            audio: File-backed target audio.
            enrollment_name: Global identity to assign to the enrolled speaker.
            enrollment_clips: Clean single-speaker reference clip paths.

        Returns:
            Full diarization with the matched speaker carrying
            ``global_speaker_id=enrollment_name``.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError(
                "SortformerDiarizer is not loaded. Call load() before diarize(), "
                "or use it as a context manager."
            )

        source_path = Path(audio.path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {source_path}")
        if bool(enrollment_name) != bool(enrollment_clips):
            raise ValueError(
                "enrollment_name and enrollment_clips must be supplied together"
            )

        logger.warning(
            "Sortformer supports at most %d distinct speakers per inference "
            "window; processing will continue, but additional speakers in one "
            "window may be missed or merged.",
            MAX_LOCAL_SPEAKERS,
        )

        with tempfile.TemporaryDirectory(prefix="sortformer-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            normalized_path = temp_dir / "normalized.wav"
            normalize_wav(
                source_path,
                normalized_path,
                sample_rate=16000,
                channels=1,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            waveform, sample_rate = sf.read(
                normalized_path,
                dtype="float32",
                always_2d=False,
            )
            if sample_rate != 16000:
                raise RuntimeError(
                    f"Normalized Sortformer input has unexpected rate {sample_rate}"
                )
            if waveform.ndim != 1:
                raise RuntimeError("Normalized Sortformer input is not mono")
            if waveform.size == 0:
                raise ValueError(f"Audio source is empty: {source_path}")

            enrollment_centroid = (
                self._build_enrollment_centroid(enrollment_clips or [], temp_dir)
                if enrollment_name
                else None
            )

            try:
                windows = self._process_windows(
                    waveform,
                    sample_rate,
                    self.window_duration_s,
                    temp_dir,
                    extract_embeddings=enrollment_centroid is not None,
                )
            except RuntimeError as exc:
                retry_window = self.oom_retry_window_s
                if (
                    retry_window is None
                    or retry_window >= self.window_duration_s
                    or not self._is_out_of_memory(exc)
                ):
                    raise
                logger.warning(
                    "Sortformer ran out of memory with %.0f-second windows; "
                    "retrying with %.0f-second windows.",
                    self.window_duration_s,
                    retry_window,
                )
                self._clear_accelerator_cache()
                windows = self._process_windows(
                    waveform,
                    sample_rate,
                    retry_window,
                    temp_dir,
                    extract_embeddings=enrollment_centroid is not None,
                )

            turns = self._stitch_windows(windows, enrollment_centroid)

        speakers = self._speakers_from_turns(turns, enrollment_name)
        return DiarizationResult(
            schema_version="2.0",
            audio_id=audio.source_id,
            speakers=speakers,
            turns=turns,
            source_audio=audio,
            model=DiarizationModelInfo(
                backend="nemo-sortformer",
                model_id=self.model_id,
                revision=self.revision,
            ),
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
        )

    def _process_windows(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        window_duration_s: float,
        temp_dir: Path,
        *,
        extract_embeddings: bool = False,
    ) -> list[_WindowResult]:
        window_samples = max(1, round(window_duration_s * sample_rate))
        overlap_samples = round(self.overlap_duration_s * sample_rate)
        stride_samples = window_samples - overlap_samples
        if stride_samples <= 0:
            raise ValueError("Window duration must be larger than overlap duration")

        windows: list[_WindowResult] = []
        start_sample = 0
        index = 0
        while start_sample < waveform.size:
            end_sample = min(start_sample + window_samples, waveform.size)
            chunk = np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
            chunk_path = temp_dir / f"window_{index:05d}.wav"
            sf.write(chunk_path, chunk, sample_rate, subtype="PCM_16")

            raw_output = self._run_model(chunk_path)
            probabilities = self._extract_probability_matrix(raw_output)
            duration_s = chunk.size / sample_rate
            if probabilities is not None:
                expected_frames = max(
                    1,
                    int(np.ceil(duration_s / FRAME_DURATION_S)),
                )
                probabilities = probabilities[:expected_frames]
                turns = self._probabilities_to_turns(probabilities, duration_s)
            else:
                turns = self._parse_native_turns(raw_output, duration_s)
                probabilities = self._turns_to_probabilities(turns, duration_s)

            windows.append(
                _WindowResult(
                    start_s=start_sample / sample_rate,
                    end_s=end_sample / sample_rate,
                    waveform=chunk,
                    probabilities=probabilities,
                    turns=turns,
                )
            )
            if end_sample >= waveform.size:
                break
            start_sample += stride_samples
            index += 1

        if extract_embeddings or (
            len(windows) > 1 and self.enable_speaker_similarity
        ):
            self._ensure_embedding_model()
            for index, window in enumerate(windows):
                window.embeddings = self._extract_window_embeddings(
                    window,
                    sample_rate,
                    temp_dir,
                    index,
                )
        return windows

    def _build_enrollment_centroid(
        self,
        clip_paths: list[str | Path],
        temp_dir: Path,
    ) -> np.ndarray:
        """Build the TitaNet enrollment anchor before target-audio inference."""
        model = self._ensure_embedding_model()
        vectors: list[np.ndarray] = []
        for index, raw_path in enumerate(clip_paths):
            source_path = Path(raw_path)
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Enrollment clip does not exist: {source_path}"
                )
            normalized_path = temp_dir / f"enrollment_{index:03d}.wav"
            normalize_wav(
                source_path,
                normalized_path,
                sample_rate=16000,
                channels=1,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            embedding = model.get_embedding(str(normalized_path))
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm > 0 and np.isfinite(vector).all():
                vectors.append(vector / norm)
        if not vectors:
            raise RuntimeError("No enrollment clip produced a valid TitaNet embedding")
        centroid = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0 or not np.isfinite(centroid).all():
            raise RuntimeError("Enrollment clips produced an invalid TitaNet centroid")
        return centroid / norm

    def _run_model(self, chunk_path: Path) -> Any:
        if self._model is None:
            raise RuntimeError("Sortformer model is unavailable")

        candidates: dict[str, Any] = {
            "audio": str(chunk_path),
            "batch_size": self.batch_size,
            "include_tensor_outputs": True,
            "num_workers": 0,
            "verbose": False,
        }
        try:
            signature = inspect.signature(self._model.diarize)
        except (TypeError, ValueError):
            # Some NeMo/decorator combinations do not expose an inspectable
            # signature. These are the stable Sortformer diarize arguments.
            return self._model.diarize(
                audio=str(chunk_path),
                batch_size=self.batch_size,
                include_tensor_outputs=True,
            )

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        available = signature.parameters
        kwargs = {
            name: value
            for name, value in candidates.items()
            if accepts_kwargs or name in available
        }
        return self._model.diarize(**kwargs)

    def _extract_probability_matrix(self, output: Any) -> np.ndarray | None:
        def visit(value: Any) -> np.ndarray | None:
            if isinstance(value, str) or value is None:
                return None
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                array = value
                while array.ndim > 2 and array.shape[0] == 1:
                    array = array[0]
                if not (
                    array.ndim == 2
                    and array.shape[0] > 0
                    and array.shape[1] == MAX_LOCAL_SPEAKERS
                ):
                    return None
                try:
                    matrix = array.astype(np.float32, copy=False)
                except (TypeError, ValueError):
                    return None
                if np.isfinite(matrix).all():
                    return np.clip(matrix, 0.0, 1.0)
                return None
            if isinstance(value, (list, tuple)):
                for item in reversed(value):
                    result = visit(item)
                    if result is not None:
                        return result
            return None

        return visit(output)

    def _probabilities_to_turns(
        self,
        probabilities: np.ndarray,
        duration_s: float,
    ) -> list[_LocalTurn]:
        turns: list[_LocalTurn] = []
        for speaker_index in range(probabilities.shape[1]):
            intervals = self._hysteresis_intervals(
                probabilities[:, speaker_index],
                duration_s,
            )
            turns.extend(
                _LocalTurn(speaker_index, start_s, end_s)
                for start_s, end_s in intervals
            )
        turns.sort(key=lambda turn: (turn.start_s, turn.end_s, turn.speaker_index))
        return turns

    def _hysteresis_intervals(
        self,
        probabilities: np.ndarray,
        duration_s: float,
    ) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        active = False
        start_frame = 0
        for frame_index, probability in enumerate(probabilities):
            if not active and probability > self.onset:
                active = True
                start_frame = frame_index
            elif active and probability < self.offset:
                intervals.append(
                    (
                        max(0.0, start_frame * FRAME_DURATION_S - self.pad_onset_s),
                        min(
                            duration_s,
                            frame_index * FRAME_DURATION_S + self.pad_offset_s,
                        ),
                    )
                )
                active = False
        if active:
            intervals.append(
                (
                    max(0.0, start_frame * FRAME_DURATION_S - self.pad_onset_s),
                    duration_s,
                )
            )

        intervals = [
            interval
            for interval in intervals
            if interval[1] - interval[0] >= self.min_duration_on_s
        ]
        if not intervals:
            return []

        merged = [intervals[0]]
        for start_s, end_s in intervals[1:]:
            previous_start, previous_end = merged[-1]
            if start_s - previous_end < self.min_duration_off_s:
                merged[-1] = (previous_start, max(previous_end, end_s))
            else:
                merged.append((start_s, end_s))
        return [(start_s, end_s) for start_s, end_s in merged if end_s > start_s]

    def _parse_native_turns(
        self,
        output: Any,
        duration_s: float,
    ) -> list[_LocalTurn]:
        lines: list[str] = []
        segments: list[tuple[Any, Any, Any]] = []
        saw_segment_data = False

        def collect(value: Any) -> None:
            nonlocal saw_segment_data
            if isinstance(value, str):
                lines.append(value)
            elif isinstance(value, (list, tuple)):
                if self._looks_like_native_segment(value):
                    saw_segment_data = True
                    segments.append((value[0], value[1], value[2]))
                    return
                for item in value:
                    collect(item)

        collect(output[0] if isinstance(output, tuple) and output else output)
        turns: list[_LocalTurn] = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                if parts[0].upper() == "SPEAKER" and len(parts) >= 8:
                    start_s = float(parts[3])
                    end_s = start_s + float(parts[4])
                    label = parts[7]
                elif len(parts) >= 3:
                    start_s = float(parts[0])
                    end_s = float(parts[1])
                    label = parts[2]
                else:
                    continue
            except ValueError:
                continue

            speaker_index = self._parse_speaker_index(label)
            if speaker_index is None:
                continue
            if not math.isfinite(start_s) or not math.isfinite(end_s):
                continue
            start_s = min(max(0.0, start_s), duration_s)
            end_s = min(max(0.0, end_s), duration_s)
            if 0 <= speaker_index < MAX_LOCAL_SPEAKERS and end_s > start_s:
                turns.append(_LocalTurn(speaker_index, start_s, end_s))

        for raw_start, raw_end, raw_label in segments:
            try:
                start_s = float(raw_start)
                end_s = float(raw_end)
            except (TypeError, ValueError):
                continue
            speaker_index = self._parse_speaker_index(raw_label)
            if (
                speaker_index is None
                or not math.isfinite(start_s)
                or not math.isfinite(end_s)
            ):
                continue
            start_s = min(max(0.0, start_s), duration_s)
            end_s = min(max(0.0, end_s), duration_s)
            if 0 <= speaker_index < MAX_LOCAL_SPEAKERS and end_s > start_s:
                turns.append(_LocalTurn(speaker_index, start_s, end_s))

        if (lines or saw_segment_data) and not turns:
            raise RuntimeError(
                "NeMo returned diarization segments in an unknown format"
            )
        turns.sort(key=lambda turn: (turn.start_s, turn.end_s, turn.speaker_index))
        return turns

    @staticmethod
    def _looks_like_native_segment(value: list[Any] | tuple[Any, ...]) -> bool:
        if len(value) < 3:
            return False
        start, end, speaker = value[:3]
        if isinstance(start, bool) or isinstance(end, bool):
            return False
        try:
            float(start)
            float(end)
        except (TypeError, ValueError):
            return False
        return SortformerDiarizer._parse_speaker_index(speaker) is not None

    @staticmethod
    def _parse_speaker_index(label: Any) -> int | None:
        if isinstance(label, bool):
            return None
        if isinstance(label, (int, np.integer)):
            return int(label)
        if isinstance(label, (float, np.floating)):
            numeric_label = float(label)
            if math.isfinite(numeric_label) and numeric_label.is_integer():
                return int(numeric_label)
            return None
        if not isinstance(label, str):
            return None

        stripped = label.strip()
        match = re.search(
            r"(?:speaker|spk)[_-]?(\d+)",
            stripped,
            re.IGNORECASE,
        )
        if match is not None:
            return int(match.group(1))
        return int(stripped) if stripped.isdigit() else None

    def _turns_to_probabilities(
        self,
        turns: list[_LocalTurn],
        duration_s: float,
    ) -> np.ndarray:
        frame_count = max(1, int(np.ceil(duration_s / FRAME_DURATION_S)))
        probabilities = np.zeros(
            (frame_count, MAX_LOCAL_SPEAKERS),
            dtype=np.float32,
        )
        for turn in turns:
            start = max(0, int(np.floor(turn.start_s / FRAME_DURATION_S)))
            end = min(frame_count, int(np.ceil(turn.end_s / FRAME_DURATION_S)))
            probabilities[start:end, turn.speaker_index] = 1.0
        return probabilities

    def _extract_window_embeddings(
        self,
        window: _WindowResult,
        sample_rate: int,
        temp_dir: Path,
        window_index: int,
    ) -> dict[int, np.ndarray]:
        if self._embedding_model is None:
            return {}

        embeddings: dict[int, np.ndarray] = {}
        active_speakers = sorted({turn.speaker_index for turn in window.turns})
        probabilities = window.probabilities
        for speaker_index in active_speakers:
            target = probabilities[:, speaker_index]
            if probabilities.shape[1] > 1:
                other = np.max(
                    np.delete(probabilities, speaker_index, axis=1),
                    axis=1,
                )
            else:
                other = np.zeros_like(target)
            clean = (target >= max(self.onset, 0.70)) & (other < 0.50)
            runs = self._true_runs(clean)
            runs = [run for run in runs if (run[1] - run[0]) * FRAME_DURATION_S >= 1.0]
            runs.sort(key=lambda run: run[1] - run[0], reverse=True)

            observations: list[np.ndarray] = []
            for clip_index, (start_frame, end_frame) in enumerate(runs[:3]):
                start_sample = round(start_frame * FRAME_DURATION_S * sample_rate)
                end_sample = round(end_frame * FRAME_DURATION_S * sample_rate)
                end_sample = min(end_sample, start_sample + 12 * sample_rate)
                clip = window.waveform[start_sample:end_sample]
                if clip.size < sample_rate:
                    continue
                clip_path = temp_dir / (
                    f"embedding_{window_index:05d}_{speaker_index}_{clip_index}.wav"
                )
                sf.write(clip_path, clip, sample_rate, subtype="PCM_16")
                embedding = self._embedding_model.get_embedding(str(clip_path))
                if hasattr(embedding, "detach"):
                    embedding = embedding.detach().cpu().numpy()
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    observations.append(vector / norm)

            if observations:
                vector = np.mean(observations, axis=0)
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    embeddings[speaker_index] = vector / norm
        return embeddings

    @staticmethod
    def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        padded = np.pad(mask.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return list(zip(starts.tolist(), ends.tolist(), strict=True))

    def _stitch_windows(
        self,
        windows: list[_WindowResult],
        enrollment_centroid: np.ndarray | None = None,
    ) -> list[SpeakerTurn]:
        profiles: dict[int, _SpeakerProfile] = (
            {0: _SpeakerProfile(enrollment_centroid.copy())}
            if enrollment_centroid is not None
            else {}
        )
        enrollment_global_index = 0 if enrollment_centroid is not None else None
        next_global_index = 1 if enrollment_centroid is not None else 0

        for window_index, window in enumerate(windows):
            active_speakers = sorted({turn.speaker_index for turn in window.turns})
            if window_index == 0:
                assignment = self._enrollment_assignment(
                    active_speakers,
                    window.embeddings,
                    enrollment_centroid,
                )
                for local_index in active_speakers:
                    global_index = assignment.get(local_index)
                    if global_index is None:
                        global_index = next_global_index
                        next_global_index += 1
                    window.local_to_global[local_index] = global_index
            else:
                previous = windows[window_index - 1]
                scores = self._speaker_match_scores(previous, window, profiles)
                assignment = self._best_assignment(active_speakers, scores)
                for local_index in active_speakers:
                    global_index = assignment.get(local_index)
                    if global_index is None:
                        global_index = next_global_index
                        next_global_index += 1
                    window.local_to_global[local_index] = global_index

            for local_index, embedding in window.embeddings.items():
                global_index = window.local_to_global.get(local_index)
                if global_index is None:
                    continue
                if global_index == enrollment_global_index:
                    continue
                profile = profiles.get(global_index)
                if profile is None:
                    profiles[global_index] = _SpeakerProfile(embedding.copy())
                else:
                    profile.update(embedding)

        turns: list[SpeakerTurn] = []
        for window_index, window in enumerate(windows):
            ownership_start = window.start_s
            ownership_end = window.end_s
            if window_index > 0:
                previous = windows[window_index - 1]
                ownership_start = window.start_s + max(
                    0.0,
                    previous.end_s - window.start_s,
                ) / 2
            if window_index + 1 < len(windows):
                following = windows[window_index + 1]
                ownership_end = following.start_s + max(
                    0.0,
                    window.end_s - following.start_s,
                ) / 2

            for turn in window.turns:
                global_index = window.local_to_global.get(turn.speaker_index)
                if global_index is None:
                    continue
                start_s = max(window.start_s + turn.start_s, ownership_start)
                end_s = min(window.start_s + turn.end_s, ownership_end)
                if end_s > start_s:
                    turns.append(
                        SpeakerTurn(
                            speaker_id=f"spk_{global_index:02d}",
                            start_s=float(start_s),
                            end_s=float(end_s),
                            confidence=None,
                        )
                    )

        turns = self._merge_adjacent_turns(turns)
        turns.sort(key=lambda turn: (turn.start_s, turn.end_s, turn.speaker_id))
        return turns

    def _enrollment_assignment(
        self,
        local_speakers: list[int],
        embeddings: dict[int, np.ndarray],
        enrollment_centroid: np.ndarray | None,
    ) -> dict[int, int]:
        """Assign at most one first-window slot to the enrolled identity."""
        if enrollment_centroid is None:
            return {}
        candidates = [
            (float(np.dot(embedding, enrollment_centroid)), local_index)
            for local_index, embedding in embeddings.items()
            if local_index in local_speakers
        ]
        if not candidates:
            return {}
        similarity, local_index = max(candidates)
        if similarity < self.embedding_similarity_threshold:
            return {}
        return {local_index: 0}

    @staticmethod
    def _speakers_from_turns(
        turns: list[SpeakerTurn],
        enrollment_name: str | None = None,
    ) -> list[Speaker]:
        """Build speaker records from turns that survived ownership clipping."""
        speakers: list[Speaker] = []
        seen: set[str] = set()
        for turn in turns:
            if turn.speaker_id in seen:
                continue
            seen.add(turn.speaker_id)
            speakers.append(
                Speaker(
                    speaker_id=turn.speaker_id,
                    global_speaker_id=(
                        enrollment_name
                        if enrollment_name and turn.speaker_id == "spk_00"
                        else None
                    ),
                )
            )

        if enrollment_name and not any(
            speaker.speaker_id == "spk_00" for speaker in speakers
        ):
            speakers.append(
                Speaker(
                    speaker_id="spk_00",
                    global_speaker_id=enrollment_name,
                )
            )

        def sort_key(speaker: Speaker) -> tuple[int, str]:
            match = re.fullmatch(r"spk_(\d+)", speaker.speaker_id)
            if match is not None:
                return (int(match.group(1)), speaker.speaker_id)
            return (10**9, speaker.speaker_id)

        speakers.sort(key=sort_key)
        return speakers

    def _speaker_match_scores(
        self,
        previous: _WindowResult,
        current: _WindowResult,
        profiles: dict[int, _SpeakerProfile],
    ) -> dict[tuple[int, int], float]:
        scores: dict[tuple[int, int], float] = {}
        overlap_s = max(0.0, previous.end_s - current.start_s)
        frame_count = max(0, round(overlap_s / FRAME_DURATION_S))
        previous_start = max(
            0,
            round((current.start_s - previous.start_s) / FRAME_DURATION_S),
        )

        active_current = sorted({turn.speaker_index for turn in current.turns})
        candidate_globals = sorted(
            set(profiles).union(previous.local_to_global.values())
        )
        for local_index in active_current:
            for global_index in candidate_globals:
                accepted_score = -1.0
                overlap_score = self._overlap_score(
                    previous,
                    current,
                    previous_start,
                    frame_count,
                    local_index,
                    global_index,
                )
                if overlap_score >= self.overlap_match_threshold:
                    accepted_score = 2.0 + overlap_score

                embedding = current.embeddings.get(local_index)
                profile = profiles.get(global_index)
                if embedding is not None and profile is not None:
                    similarity = float(np.dot(embedding, profile.centroid))
                    if similarity >= self.embedding_similarity_threshold:
                        accepted_score = max(accepted_score, 1.0 + similarity)

                if accepted_score >= 0:
                    scores[(local_index, global_index)] = accepted_score
        return scores

    def _overlap_score(
        self,
        previous: _WindowResult,
        current: _WindowResult,
        previous_start: int,
        frame_count: int,
        current_local: int,
        global_index: int,
    ) -> float:
        if frame_count <= 0:
            return 0.0
        previous_locals = [
            local_index
            for local_index, assigned_global in previous.local_to_global.items()
            if assigned_global == global_index
        ]
        if not previous_locals:
            return 0.0

        previous_end = min(
            previous.probabilities.shape[0],
            previous_start + frame_count,
        )
        current_end = min(
            current.probabilities.shape[0],
            previous_end - previous_start,
        )
        count = min(previous_end - previous_start, current_end)
        if count <= 0:
            return 0.0

        current_activity = (
            current.probabilities[:count, current_local] >= self.onset
        )
        best = 0.0
        for previous_local in previous_locals:
            previous_activity = (
                previous.probabilities[
                    previous_start : previous_start + count,
                    previous_local,
                ]
                >= self.onset
            )
            total_active = int(previous_activity.sum() + current_activity.sum())
            if total_active < 4:
                continue
            intersection = int(
                np.logical_and(previous_activity, current_activity).sum()
            )
            best = max(best, 2.0 * intersection / total_active)
        return best

    def _best_assignment(
        self,
        local_speakers: list[int],
        scores: dict[tuple[int, int], float],
    ) -> dict[int, int | None]:
        candidates: dict[int, list[tuple[int | None, float]]] = {}
        for local_index in local_speakers:
            existing = [
                (global_index, score)
                for (candidate_local, global_index), score in scores.items()
                if candidate_local == local_index
            ]
            existing.sort(key=lambda candidate: (-candidate[1], candidate[0]))
            candidates[local_index] = [(None, 0.0), *existing]

        best_score = -1.0
        best: dict[int, int | None] = {}

        def search(
            position: int,
            used_globals: set[int],
            total_score: float,
            assignment: dict[int, int | None],
        ) -> None:
            nonlocal best_score, best
            if position == len(local_speakers):
                if total_score > best_score:
                    best_score = total_score
                    best = assignment.copy()
                return
            local_index = local_speakers[position]
            for global_index, score in candidates[local_index]:
                if global_index is not None and global_index in used_globals:
                    continue
                assignment[local_index] = global_index
                if global_index is not None:
                    used_globals.add(global_index)
                search(position + 1, used_globals, total_score + score, assignment)
                if global_index is not None:
                    used_globals.remove(global_index)
                del assignment[local_index]

        search(0, set(), 0.0, {})
        return best

    @staticmethod
    def _merge_adjacent_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
        by_speaker: dict[str, list[SpeakerTurn]] = {}
        for turn in turns:
            by_speaker.setdefault(turn.speaker_id, []).append(turn)

        merged: list[SpeakerTurn] = []
        for speaker_id, speaker_turns in by_speaker.items():
            if not speaker_turns:
                continue
            speaker_turns.sort(key=lambda turn: (turn.start_s, turn.end_s))
            current = speaker_turns[0]
            for turn in speaker_turns[1:]:
                if turn.start_s <= current.end_s + FRAME_DURATION_S:
                    current = SpeakerTurn(
                        speaker_id=speaker_id,
                        start_s=current.start_s,
                        end_s=max(current.end_s, turn.end_s),
                        confidence=None,
                    )
                else:
                    merged.append(current)
                    current = turn
            merged.append(current)
        return merged

    @staticmethod
    def _is_out_of_memory(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "mps backend out of memory" in message

    @staticmethod
    def _clear_accelerator_cache() -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
