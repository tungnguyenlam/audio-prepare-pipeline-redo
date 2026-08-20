"""NVIDIA NeMo clustering speaker diarization backend.

Cascaded pipeline from the NeMo speaker-diarization models documentation:
MarbleNet voice-activity detection, TitaNet speaker embeddings, then spectral
clustering. Requires the isolated NeMo environment pinned in
``requirements-sortformer.txt``.
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
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

logger = logging.getLogger(__name__)

DEFAULT_VAD_MODEL = "vad_multilingual_marblenet"
DEFAULT_SPEAKER_MODEL = "titanet_large"
DEFAULT_MAX_NUM_SPEAKERS = 8
SAMPLE_RATE = 16000


class ClusteringDiarizer(BaseDiarizer, ManagedModel):
    """Diarize audio with NeMo's MarbleNet + TitaNet clustering pipeline.

    Long-form clustering parameters follow NeMo's general inference config.
    Speaker labels are converted to result-local ``spk_NN`` identifiers.
    """

    def __init__(
        self,
        vad_model: str = DEFAULT_VAD_MODEL,
        speaker_model: str = DEFAULT_SPEAKER_MODEL,
        *,
        device: str = "auto",
        num_speakers: int | None = None,
        max_num_speakers: int = DEFAULT_MAX_NUM_SPEAKERS,
        batch_size: int = 64,
        num_workers: int = 0,
        ffmpeg_bin: str = "ffmpeg",
        vad_onset: float = 0.5,
        vad_offset: float = 0.3,
        vad_pad_onset_s: float = 0.2,
        vad_pad_offset_s: float = 0.2,
        vad_min_duration_on_s: float = 0.5,
        vad_min_duration_off_s: float = 0.5,
    ) -> None:
        """Initialize the NeMo clustering diarizer.

        Args:
            vad_model: MarbleNet VAD name or local ``.nemo`` path.
            speaker_model: TitaNet embedding name or local ``.nemo`` path.
            device: Compute device (``"auto"``, ``"cuda"``, ``"cpu"``, etc.).
            num_speakers: Exact speaker count when known in advance.
            max_num_speakers: Upper bound used when the count is estimated.
            batch_size: VAD and embedding extraction batch size.
            num_workers: Dataloader workers. ``0`` avoids fork issues in workers.
            ffmpeg_bin: ``ffmpeg`` executable used to normalize input audio.
            vad_onset: Speech-onset probability threshold.
            vad_offset: Speech-offset probability threshold.
            vad_pad_onset_s: Padding added before each detected speech region.
            vad_pad_offset_s: Padding added after each detected speech region.
            vad_min_duration_on_s: Minimum kept speech segment duration.
            vad_min_duration_off_s: Minimum kept non-speech gap duration.
        """
        ManagedModel.__init__(self)
        if max_num_speakers < 1:
            raise ValueError("max_num_speakers must be at least 1")
        if num_speakers is not None and num_speakers < 1:
            raise ValueError("num_speakers must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        self.vad_model = vad_model
        self.speaker_model = speaker_model
        self.device = str(device)
        self.num_speakers = num_speakers
        self.max_num_speakers = int(max_num_speakers)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.ffmpeg_bin = ffmpeg_bin
        self.vad_onset = float(vad_onset)
        self.vad_offset = float(vad_offset)
        self.vad_pad_onset_s = float(vad_pad_onset_s)
        self.vad_pad_offset_s = float(vad_pad_offset_s)
        self.vad_min_duration_on_s = float(vad_min_duration_on_s)
        self.vad_min_duration_off_s = float(vad_min_duration_off_s)

        self._model: Any | None = None
        self._target_device: Any | None = None

    @staticmethod
    def resolve_speaker_settings(
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
        *,
        default_max: int = DEFAULT_MAX_NUM_SPEAKERS,
    ) -> tuple[int | None, int]:
        """Map UI speaker bounds onto clustering oracle / max settings.

        Args:
            num_speakers: Exact speaker count when known.
            min_speakers: Optional lower bound from the caller UI.
            max_speakers: Optional upper bound from the caller UI.
            default_max: Fallback maximum when the caller does not set one.

        Returns:
            ``(oracle_num_speakers, max_num_speakers)``. Oracle is set when an
            exact count is given, or when min and max are equal.
        """
        oracle = num_speakers
        if (
            oracle is None
            and min_speakers is not None
            and max_speakers is not None
            and min_speakers == max_speakers
        ):
            oracle = min_speakers
        max_num = max_speakers if max_speakers is not None else default_max
        if oracle is not None:
            max_num = max(max_num, oracle)
        return oracle, max(1, int(max_num))

    def _load(self) -> None:
        """Restore MarbleNet VAD and TitaNet embedding models."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Clustering diarizer dependencies are unavailable. Install the "
                "pinned requirements-sortformer.txt dependencies in an isolated "
                "NeMo environment."
            ) from exc

        target_device = self._resolve_device(torch)
        if target_device.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        try:
            omega_conf = importlib.import_module("omegaconf")
            nemo_models = importlib.import_module("nemo.collections.asr.models")
            clustering_model_type = nemo_models.ClusteringDiarizer
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Clustering diarizer dependencies are unavailable. Install the "
                "pinned requirements-sortformer.txt dependencies in an isolated "
                "NeMo environment."
            ) from exc

        config = omega_conf.OmegaConf.create(self._config_dict(".", target_device))
        model = clustering_model_type(cfg=config)

        self._target_device = target_device
        self._model = model

    def _config_dict(self, out_dir: str, target_device: Any) -> dict[str, Any]:
        oracle = self.num_speakers is not None
        return {
            "num_workers": self.num_workers,
            "sample_rate": SAMPLE_RATE,
            "batch_size": self.batch_size,
            "device": str(target_device),
            "verbose": False,
            "diarizer": {
                "manifest_filepath": None,
                "out_dir": out_dir,
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": self.vad_model,
                    "external_vad_manifest": None,
                    "parameters": {
                        "window_length_in_sec": 0.63,
                        "shift_length_in_sec": 0.08,
                        "smoothing": False,
                        "overlap": 0.5,
                        "onset": self.vad_onset,
                        "offset": self.vad_offset,
                        "pad_onset": self.vad_pad_onset_s,
                        "pad_offset": self.vad_pad_offset_s,
                        "min_duration_on": self.vad_min_duration_on_s,
                        "min_duration_off": self.vad_min_duration_off_s,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": self.speaker_model,
                    "parameters": {
                        "window_length_in_sec": [1.9, 1.2, 0.5],
                        "shift_length_in_sec": [0.95, 0.6, 0.25],
                        "multiscale_weights": [1, 1, 1],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": oracle,
                        "max_num_speakers": self.max_num_speakers,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 10,
                        "maj_vote_spk_count": False,
                        "chunk_cluster_count": 50,
                        "embeddings_per_chunk": 10000,
                    },
                },
            },
        }

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
        """Release VAD, speaker-embedding, and accelerator caches."""
        self._model = None
        self._target_device = None
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def diarize(
        self,
        audio: Audio,
        *,
        num_speakers: int | None = None,
    ) -> DiarizationResult:
        """Diarize ``audio`` while preserving the schema 1.0 result contract.

        Args:
            audio: File-backed audio item to diarize.
            num_speakers: Optional per-call oracle speaker count. When omitted,
                uses the value configured at construction time.

        Returns:
            Speaker identities and turns with local ``spk_NN`` labels.

        Raises:
            RuntimeError: If the model is not loaded or NeMo clustering fails.
            FileNotFoundError: If ``audio.path`` does not exist.
            ValueError: If the normalized audio duration is empty.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError(
                "ClusteringDiarizer is not loaded. Call load() before diarize(), "
                "or use it as a context manager."
            )

        source_path = Path(audio.path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {source_path}")

        oracle_speakers = (
            num_speakers if num_speakers is not None else self.num_speakers
        )
        if oracle_speakers is not None and oracle_speakers < 1:
            raise ValueError("num_speakers must be at least 1")

        with tempfile.TemporaryDirectory(prefix="nemo-clustering-") as temp_dir_name:
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

            manifest_path = temp_dir / "manifest.json"
            self._write_manifest(manifest_path, normalized_path, oracle_speakers)
            self._prepare_inference(temp_dir, manifest_path, oracle_speakers)

            try:
                self._model.diarize()
            except Exception as exc:
                rttm_path = self._find_rttm(temp_dir)
                if rttm_path is None:
                    raise RuntimeError(
                        f"NeMo clustering diarization failed: {exc}"
                    ) from exc
                logger.warning(
                    "NeMo clustering scoring failed after writing RTTM; "
                    "continuing with predicted turns: %s",
                    exc,
                )

            rttm_path = self._find_rttm(temp_dir)
            if rttm_path is None:
                raise RuntimeError(
                    "NeMo clustering diarization finished without writing an "
                    "RTTM file under pred_rttms/"
                )
            turns, speakers = self._turns_from_rttm(rttm_path)

        return DiarizationResult(
            schema_version="1.0",
            audio_id=audio.source_id,
            speakers=speakers,
            turns=turns,
            model=DiarizationModelInfo(
                backend="nemo-clustering",
                model_id=f"{self.vad_model}+{self.speaker_model}",
            ),
        )

    def _write_manifest(
        self,
        manifest_path: Path,
        audio_path: Path,
        oracle_speakers: int | None,
    ) -> None:
        entry = {
            "audio_filepath": str(audio_path),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": oracle_speakers,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        manifest_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    def _prepare_inference(
        self,
        out_dir: Path,
        manifest_path: Path,
        oracle_speakers: int | None,
    ) -> None:
        if self._model is None:
            raise RuntimeError("Clustering model is unavailable")

        from omegaconf import open_dict

        params = self._model._diarizer_params
        cluster_params = self._model._cluster_params
        with open_dict(params):
            params.manifest_filepath = str(manifest_path)
            params.out_dir = str(out_dir)
        with open_dict(cluster_params):
            cluster_params.oracle_num_speakers = oracle_speakers is not None
            cluster_params.max_num_speakers = (
                max(self.max_num_speakers, oracle_speakers)
                if oracle_speakers is not None
                else self.max_num_speakers
            )

    @staticmethod
    def _find_rttm(out_dir: Path) -> Path | None:
        rttm_dir = out_dir / "pred_rttms"
        if not rttm_dir.is_dir():
            return None
        matches = sorted(rttm_dir.glob("*.rttm"))
        return matches[0] if matches else None

    @staticmethod
    def _turns_from_rttm(rttm_path: Path) -> tuple[list[SpeakerTurn], list[Speaker]]:
        raw_turns: list[tuple[str, float, float]] = []
        for line in rttm_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            try:
                start_s = float(parts[3])
                duration_s = float(parts[4])
            except ValueError:
                continue
            if not math.isfinite(start_s) or not math.isfinite(duration_s):
                continue
            end_s = start_s + duration_s
            if end_s <= start_s:
                continue
            raw_turns.append((parts[7], max(0.0, start_s), end_s))

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
