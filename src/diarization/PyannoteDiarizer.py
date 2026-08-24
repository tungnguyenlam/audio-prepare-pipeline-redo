"""Pyannote speaker diarization backend."""

from __future__ import annotations

import gc
import os
from typing import Any

from src.base.model import ManagedModel
from src.diarization.BaseDiarizer import BaseDiarizer
from src.diarization.schemas import (
    DiarizationModelInfo,
    DiarizationResult,
    Speaker,
    SpeakerTurn,
)
from src.utils.AudioClass import Audio


DEFAULT_PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"


class PyannoteDiarizer(BaseDiarizer, ManagedModel):
    """Speaker diarization using Pyannote's community pipeline.

    The speaker labels produced by Pyannote are converted to local ``spk_NN``
    identifiers. Those identifiers are meaningful only within one result.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_PYANNOTE_MODEL_ID,
        device: str = "auto",
        token: str | None = None,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        """Initialize PyannoteDiarizer.

        Args:
            model_id: Hugging Face repository ID for the diarization pipeline.
                Defaults to ``"pyannote/speaker-diarization-community-1"``.
            device: Compute device (``"auto"``, ``"cuda"``, ``"cpu"``, etc.).
            token: Optional Hugging Face access token (or ``HF_TOKEN`` env).
            num_speakers: Optional exact number of speakers to look for.
            min_speakers: Optional lower bound on the number of speakers.
            max_speakers: Optional upper bound on the number of speakers.
        """
        ManagedModel.__init__(self)
        self.model_id = model_id
        self.device = str(device)
        self.token = token
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline: Any | None = None

    def _load(self) -> None:
        """Load the Pyannote pipeline and move it to CUDA when requested."""
        from pyannote.audio import Pipeline
        import torch

        token = self.token if self.token is not None else os.getenv("HF_TOKEN")
        pipeline = Pipeline.from_pretrained(self.model_id, token=token)
        if pipeline is None:
            raise RuntimeError(
                f"Pyannote pipeline could not be loaded for model {self.model_id!r}"
            )

        if self.device == "auto":
            target_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            target_device = torch.device(self.device)
        if target_device.type != "cpu" or self.device not in {"auto", "cpu"}:
            pipeline.to(target_device)

        # Assign only after the complete load, including device placement,
        # succeeds.  ManagedModel marks the instance loaded afterwards.
        self._pipeline = pipeline

    def _unload(self) -> None:
        """Release the Pyannote pipeline and any cached CUDA allocations."""
        self._pipeline = None
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def diarize(
        self,
        audio: Audio,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        hook: Any | None = None,
    ) -> DiarizationResult:
        """Diarize ``audio`` and return backend-independent speaker turns.

        Args:
            audio: Audio instance to diarize.
            num_speakers: Exact number of speakers, if known in advance.
            min_speakers: Minimum number of speakers to look for.
            max_speakers: Maximum number of speakers to look for.
            hook: Optional callback hook for progress tracking.

        Returns:
            A DiarizationResult containing detected speakers and turns.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If the audio file is empty.
        """
        if not self.is_loaded or self._pipeline is None:
            raise RuntimeError(
                "PyannoteDiarizer is not loaded. Call load() before diarize(), "
                "or use it as a context manager."
            )

        import soundfile as sf
        import torch

        waveform, sample_rate = sf.read(
            str(audio.path),
            dtype="float32",
            always_2d=True,
        )
        if waveform.shape[0] == 0:
            raise ValueError(f"Cannot diarize empty audio file: {audio.path}")

        # Pyannote expects (channel, time). Downmix stereo to mono if needed.
        if waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1, keepdims=True)

        pipeline_input = {
            "waveform": torch.from_numpy(waveform.T.copy()),
            "sample_rate": int(sample_rate),
        }

        pipeline_kwargs: dict[str, Any] = {}
        target_num_speakers = (
            num_speakers if num_speakers is not None else self.num_speakers
        )
        target_min_speakers = (
            min_speakers if min_speakers is not None else self.min_speakers
        )
        target_max_speakers = (
            max_speakers if max_speakers is not None else self.max_speakers
        )

        if target_num_speakers is not None:
            pipeline_kwargs["num_speakers"] = int(target_num_speakers)
        if target_min_speakers is not None:
            pipeline_kwargs["min_speakers"] = int(target_min_speakers)
        if target_max_speakers is not None:
            pipeline_kwargs["max_speakers"] = int(target_max_speakers)
        if hook is not None:
            pipeline_kwargs["hook"] = hook

        output = self._pipeline(pipeline_input, **pipeline_kwargs)

        if hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization
        elif isinstance(output, dict) and "speaker_diarization" in output:
            annotation = output["speaker_diarization"]
        else:
            annotation = output

        label_to_speaker_id: dict[Any, str] = {}
        speakers: list[Speaker] = []
        turns: list[SpeakerTurn] = []

        for segment, _, pyannote_label in annotation.itertracks(yield_label=True):
            start_s = float(segment.start)
            end_s = float(segment.end)
            if end_s <= start_s:
                continue

            speaker_id = label_to_speaker_id.get(pyannote_label)
            if speaker_id is None:
                speaker_id = f"spk_{len(label_to_speaker_id):02d}"
                label_to_speaker_id[pyannote_label] = speaker_id
                speakers.append(Speaker(speaker_id=speaker_id))

            turns.append(
                SpeakerTurn(
                    speaker_id=speaker_id,
                    start_s=start_s,
                    end_s=end_s,
                    confidence=None,
                )
            )

        return DiarizationResult(
            schema_version="1.0",
            audio_id=audio.source_id,
            speakers=speakers,
            turns=turns,
            model=DiarizationModelInfo(
                backend="pyannote",
                model_id=self.model_id,
            ),
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
        )
