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


class PyannoteDiarizer(BaseDiarizer, ManagedModel):
    """Speaker diarization using Pyannote's community pipeline.

    The speaker labels produced by Pyannote are converted to local ``spk_NN``
    identifiers.  Those identifiers are meaningful only within one result.
    """

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-community-1",
        device: str = "auto",
        token: str | None = None,
    ) -> None:
        ManagedModel.__init__(self)
        self.model_id = model_id
        self.device = str(device)
        self.token = token
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

    def diarize(self, audio: Audio) -> DiarizationResult:
        """Diarize ``audio`` and return backend-independent speaker turns."""
        if not self.is_loaded or self._pipeline is None:
            raise RuntimeError(
                "PyannoteDiarizer is not loaded. Call load() before diarize(), "
                "or use it as a context manager."
            )

        output = self._pipeline(str(audio.path))
        annotation = output.speaker_diarization

        label_to_speaker_id: dict[Any, str] = {}
        speakers: list[Speaker] = []
        turns: list[SpeakerTurn] = []

        for segment, _, pyannote_label in annotation.itertracks(yield_label=True):
            speaker_id = label_to_speaker_id.get(pyannote_label)
            if speaker_id is None:
                speaker_id = f"spk_{len(label_to_speaker_id):02d}"
                label_to_speaker_id[pyannote_label] = speaker_id
                speakers.append(Speaker(speaker_id=speaker_id))

            turns.append(
                SpeakerTurn(
                    speaker_id=speaker_id,
                    start_s=float(segment.start),
                    end_s=float(segment.end),
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
        )
