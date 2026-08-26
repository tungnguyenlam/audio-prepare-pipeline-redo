"""DiariZen WavLM speaker diarization backend."""

from __future__ import annotations

import gc
import math
import os
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


DEFAULT_DIARIZEN_MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
SAMPLE_RATE = 16000


class DiariZenDiarizer(BaseDiarizer, ManagedModel):
    """Diarize audio with DiariZen's WavLM and VBx pipeline.

    DiariZen can emit simultaneous turns for overlapping speakers. Backend
    labels are converted to result-local ``spk_NN`` identifiers.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_DIARIZEN_MODEL_ID,
        *,
        device: str = "auto",
        token: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        batch_size: int = 1,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        """Initialize the DiariZen diarizer.

        Args:
            model_id: Hugging Face DiariZen model repository ID.
            device: Compute device (``"auto"``, ``"cuda"``, ``"cuda:N"``,
                or ``"cpu"``).
            token: Optional Hugging Face token. Falls back to ``HF_TOKEN``.
            num_speakers: Optional exact speaker count.
            min_speakers: Optional minimum speaker count.
            max_speakers: Optional maximum speaker count.
            batch_size: Segmentation and embedding inference batch size.
            ffmpeg_bin: ``ffmpeg`` executable used to normalize input audio.
        """
        ManagedModel.__init__(self)
        self._validate_speaker_settings(
            num_speakers,
            min_speakers,
            max_speakers,
        )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be an integer of at least 1")
        self.model_id = model_id
        self.device = str(device)
        self.token = token
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.batch_size = batch_size
        self.ffmpeg_bin = ffmpeg_bin
        self._pipeline: Any | None = None
        self._target_device: Any | None = None

    @staticmethod
    def _validate_speaker_settings(
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> None:
        for name, value in (
            ("num_speakers", num_speakers),
            ("min_speakers", min_speakers),
            ("max_speakers", max_speakers),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be an integer of at least 1")
        if (
            min_speakers is not None
            and max_speakers is not None
            and min_speakers > max_speakers
        ):
            raise ValueError("min_speakers must be <= max_speakers")
        if num_speakers is not None:
            if min_speakers is not None and num_speakers < min_speakers:
                raise ValueError("num_speakers must be >= min_speakers")
            if max_speakers is not None and num_speakers > max_speakers:
                raise ValueError("num_speakers must be <= max_speakers")

    def _resolve_device(self, torch: Any) -> Any:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = torch.device(self.device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if target.type not in {"cuda", "cpu"}:
            raise RuntimeError(
                "DiariZen supports CUDA and CPU devices; "
                f"received {self.device!r}"
            )
        return target

    def _load(self) -> None:
        """Load the released DiariZen pipeline and move it to the device."""
        try:
            import torch
            from diarizen.pipelines.inference import DiariZenPipeline
        except ImportError as exc:
            raise RuntimeError(
                "DiariZen dependencies are unavailable. Create the isolated "
                ".venv-diarizen environment from requirements-diarizen.txt."
            ) from exc

        token = self.token if self.token is not None else os.getenv("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token

        target_device = self._resolve_device(torch)
        try:
            pipeline = DiariZenPipeline.from_pretrained(self.model_id)
            pipeline.segmentation_batch_size = self.batch_size
            pipeline.embedding_batch_size = self.batch_size
            pipeline.to(target_device)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load DiariZen model {self.model_id!r}: {exc}"
            ) from exc

        self._target_device = target_device
        self._pipeline = pipeline

    def _unload(self) -> None:
        """Release DiariZen models and accelerator allocations."""
        self._pipeline = None
        self._target_device = None
        gc.collect()

        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def diarize(
        self,
        audio: Audio,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """Diarize ``audio`` and return backend-independent speaker turns.

        Args:
            audio: File-backed audio item to diarize.
            num_speakers: Per-call exact speaker count override.
            min_speakers: Per-call minimum speaker count override.
            max_speakers: Per-call maximum speaker count override.

        Returns:
            Speaker identities and turns with local ``spk_NN`` labels.

        Raises:
            RuntimeError: If the model is not loaded or inference fails.
            FileNotFoundError: If the input audio does not exist.
            ValueError: If speaker settings are invalid or audio is empty.
        """
        if not self.is_loaded or self._pipeline is None:
            raise RuntimeError(
                "DiariZenDiarizer is not loaded. Call load() before diarize(), "
                "or use it as a context manager."
            )

        source_path = Path(audio.path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {source_path}")

        exact = num_speakers if num_speakers is not None else self.num_speakers
        minimum = min_speakers if min_speakers is not None else self.min_speakers
        maximum = max_speakers if max_speakers is not None else self.max_speakers
        self._validate_speaker_settings(exact, minimum, maximum)
        if exact is not None:
            minimum = exact
            maximum = exact

        pipeline = self._pipeline
        original_min = pipeline.min_speakers
        original_max = pipeline.max_speakers
        if minimum is not None:
            pipeline.min_speakers = minimum
        if maximum is not None:
            pipeline.max_speakers = maximum

        try:
            with tempfile.TemporaryDirectory(prefix="diarizen-") as temp_name:
                normalized_path = Path(temp_name) / "normalized.wav"
                normalize_wav(
                    source_path,
                    normalized_path,
                    sample_rate=SAMPLE_RATE,
                    channels=1,
                    ffmpeg_bin=self.ffmpeg_bin,
                )
                if probe_wav(normalized_path)[1] <= 0:
                    raise ValueError(f"Audio source is empty: {source_path}")
                annotation = pipeline(str(normalized_path))
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"DiariZen diarization failed: {exc}") from exc
        finally:
            pipeline.min_speakers = original_min
            pipeline.max_speakers = original_max

        turns, speakers = self._turns_from_annotation(
            annotation,
            duration_s=audio.duration_s,
        )
        return DiarizationResult(
            schema_version="2.0",
            audio_id=audio.source_id,
            speakers=speakers,
            turns=turns,
            source_audio=audio,
            model=DiarizationModelInfo(
                backend="diarizen",
                model_id=self.model_id,
            ),
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
        )

    @staticmethod
    def _turns_from_annotation(
        annotation: Any,
        duration_s: float | None = None,
    ) -> tuple[list[SpeakerTurn], list[Speaker]]:
        """Convert a Pyannote Annotation into schema turns.

        Last-frame timestamps from WavLM/VBx can land a few tens of
        milliseconds past the source file. Clamp them to ``duration_s`` so
        ``DiarizationResult`` schema 2.0 does not reject the result.
        """
        raw_turns: list[tuple[str, float, float]] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            start_s = float(segment.start)
            end_s = float(segment.end)
            if not math.isfinite(start_s) or not math.isfinite(end_s):
                continue
            if duration_s is not None:
                start_s = min(max(0.0, start_s), duration_s)
                end_s = min(max(0.0, end_s), duration_s)
            else:
                start_s = max(0.0, start_s)
            if end_s <= start_s:
                continue
            raw_turns.append((str(label), start_s, end_s))

        raw_turns.sort(key=lambda item: (item[1], item[2], item[0]))
        label_to_id: dict[str, str] = {}
        speakers: list[Speaker] = []
        turns: list[SpeakerTurn] = []
        for label, start_s, end_s in raw_turns:
            speaker_id = label_to_id.get(label)
            if speaker_id is None:
                speaker_id = f"spk_{len(label_to_id):02d}"
                label_to_id[label] = speaker_id
                speakers.append(Speaker(speaker_id=speaker_id))
            turns.append(
                SpeakerTurn(
                    speaker_id=speaker_id,
                    start_s=start_s,
                    end_s=end_s,
                    confidence=None,
                )
            )
        return turns, speakers
