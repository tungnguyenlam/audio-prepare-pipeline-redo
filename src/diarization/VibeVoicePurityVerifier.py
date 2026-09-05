"""Speaker-purity verification from VibeVoice-ASR structured diarization.

The transcript is ignored. The verifier runs the whole candidate file through
VibeVoice-ASR, counts distinct speaker IDs, and classifies:

* exactly one speaker → ``pass``
* a second speaker with meaningful duration → ``reject``
* empty, unlabeled, or tiny secondary speech → ``uncertain``
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.base.model import ManagedModel
from src.diarization.schemas import (
    DiarizationModelInfo,
    VibeVoicePurityResult,
    VibeVoiceSpeakerTurn,
)
from src.utils.AudioClass import Audio

DEFAULT_VIBEVOICE_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
DEFAULT_MIN_SECONDARY_SPEECH_S = 0.25
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_VIBEVOICE_BATCH_SIZE = 1
VIBEVOICE_PURITY_SCHEMA_VERSION = "1.0"
VIBEVOICE_AUDIO_SKIP_MODULES = (
    "acoustic_tokenizer_encoder",
    "semantic_tokenizer_encoder",
    "acoustic_projection",
    "semantic_projection",
    "lm_head",
)
VIBEVOICE_MODEL_CHOICES: tuple[dict[str, Any], ...] = (
    {
        "id": DEFAULT_VIBEVOICE_MODEL_ID,
        "label": "Full BF16 (~17 GB VRAM)",
        "quantized": False,
    },
    {
        "id": "Dubedo/VibeVoice-ASR-HF-INT8",
        "label": "INT8 (~10–11 GB VRAM)",
        "quantized": True,
    },
    {
        "id": "Dubedo/VibeVoice-ASR-HF-NF4",
        "label": "NF4 4-bit (~7–8 GB VRAM)",
        "quantized": True,
    },
)

logger = logging.getLogger(__name__)


class VibeVoicePurityError(RuntimeError):
    """Raised when VibeVoice-ASR cannot be loaded or the source file is unusable."""


class VibeVoicePurityVerifier(ManagedModel):
    """Verify single-speaker purity from VibeVoice-ASR speaker-count output.

    Uses Transformers-native ``microsoft/VibeVoice-ASR-HF`` or a selective
    bitsandbytes checkpoint from :data:`VIBEVOICE_MODEL_CHOICES` (requires
    ``transformers>=5.3.0``). Call ``load()`` first or use a context manager.
    Inference is run on the dedicated model server, not the development
    machine. Quantized checkpoints need CUDA and ``bitsandbytes>=0.48.1``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VIBEVOICE_MODEL_ID,
        *,
        device: str = "auto",
        token: str | None = None,
        min_secondary_speech_s: float = DEFAULT_MIN_SECONDARY_SPEECH_S,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        batch_size: int = DEFAULT_VIBEVOICE_BATCH_SIZE,
        attn_implementation: str = "eager",
    ) -> None:
        """Initialize the verifier without loading weights.

        Args:
            model_id: Hugging Face repository ID. Studio offers the
                :data:`VIBEVOICE_MODEL_CHOICES` catalog; any Transformers-native
                VibeVoice-ASR checkpoint also loads.
            device: ``auto``, ``cuda``, ``cuda:N``, ``mps``, or ``cpu``.
            token: Optional Hugging Face token (else ``HF_TOKEN``).
            min_secondary_speech_s: Reject only when non-dominant speaker
                duration meets this threshold. Smaller secondary turns are
                ``uncertain``.
            max_new_tokens: Generation budget for the structured transcript.
            batch_size: Maximum number of candidate files per model forward pass.
            attn_implementation: Transformers attention backend.
        """
        ManagedModel.__init__(self)
        self.model_id = str(model_id).strip()
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        self.device = str(device)
        self.token = token
        self.min_secondary_speech_s = _validate_min_secondary_speech_s(
            min_secondary_speech_s
        )
        self.max_new_tokens = _validate_max_new_tokens(max_new_tokens)
        self.batch_size = _validate_batch_size(batch_size)
        self.attn_implementation = str(attn_implementation).strip() or "eager"
        self._processor: Any = None
        self._model: Any = None
        self._torch_device: Any = None
        self._effective_attn_implementation = self.attn_implementation

    def _load(self) -> None:
        """Load the VibeVoice-ASR processor and model."""
        try:
            import torch
            from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
        except ImportError as exc:
            raise VibeVoicePurityError(
                "VibeVoice-ASR requires transformers>=5.3.0 with "
                "VibeVoiceAsrForConditionalGeneration. Create .venv-vibevoice "
                "from requirements-vibevoice.txt or set VIBEVOICE_PYTHON."
            ) from exc

        token = self.token if self.token is not None else os.getenv("HF_TOKEN")
        self._torch_device = _resolve_torch_device(self.device, torch)
        quant_config = _peek_quantization_config(self.model_id, token)
        quantized = _is_bitsandbytes_quantization(
            quant_config
        ) or vibevoice_model_is_quantized(self.model_id)
        if quantized and self._torch_device.type != "cuda":
            raise VibeVoicePurityError(
                f"Quantized VibeVoice checkpoint {self.model_id!r} requires "
                "CUDA (bitsandbytes does not run on CPU or MPS)."
            )
        dtype = (
            torch.bfloat16
            if self._torch_device.type == "cuda"
            else torch.float32
        )
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": self.attn_implementation,
        }
        if quantized:
            device_index = (
                self._torch_device.index
                if self._torch_device.index is not None
                else 0
            )
            load_kwargs["device_map"] = {"": device_index}
            if quant_config:
                _warn_if_missing_audio_skip_modules(self.model_id, quant_config)
        if token:
            load_kwargs["token"] = token
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, token=token
            )
            try:
                self._model = (
                    VibeVoiceAsrForConditionalGeneration.from_pretrained(
                        self.model_id, **load_kwargs
                    )
                )
                self._effective_attn_implementation = self.attn_implementation
            except Exception as exc:
                if not _is_unsupported_attention_error(exc):
                    raise
                logger.warning(
                    "VibeVoice attention=%s is unsupported; retrying once "
                    "with eager attention for every nested backbone",
                    self.attn_implementation,
                )
                self._model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                eager_kwargs = {
                    **load_kwargs,
                    "attn_implementation": {"": "eager"},
                }
                self._model = (
                    VibeVoiceAsrForConditionalGeneration.from_pretrained(
                        self.model_id, **eager_kwargs
                    )
                )
                self._effective_attn_implementation = "eager"
            if not _model_is_quantized(self._model):
                self._model = self._model.to(self._torch_device)
            self._model.eval()
            logger.info(
                "Loaded VibeVoice-ASR model=%s device=%s attention=%s quantized=%s",
                self.model_id,
                self._model_runtime_device(),
                self._effective_attn_implementation,
                _model_is_quantized(self._model),
            )
        except Exception as exc:
            self._processor = None
            self._model = None
            self._torch_device = None
            raise VibeVoicePurityError(
                f"Could not load VibeVoice-ASR from {self.model_id!r}: {exc}"
            ) from exc

    def _unload(self) -> None:
        """Release processor, model, and accelerator cache."""
        self._processor = None
        self._model = None
        self._torch_device = None
        self._effective_attn_implementation = self.attn_implementation
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    def verify(self, audio: Audio) -> VibeVoicePurityResult:
        """Run VibeVoice-ASR on the whole candidate and classify speaker count.

        Args:
            audio: File-backed candidate segment. Do not pre-slice into
                sub-second windows; the model uses the full clip as context.

        Returns:
            A purity decision. Inference failures are ``uncertain`` with
            ``reason="inference_error"`` rather than raising, so a batch can
            continue. Missing files still raise ``FileNotFoundError``.

        Raises:
            RuntimeError: If ``load()`` has not completed.
            FileNotFoundError: If ``audio.path`` is missing.
        """
        if not self.is_loaded or self._model is None or self._processor is None:
            raise RuntimeError(
                "VibeVoice purity verifier is not loaded. Call load() first "
                "or use it as a context manager."
            )
        path = Path(audio.path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        try:
            segments = self._infer(path)
        except Exception as exc:
            return _result(
                audio_id=audio.source_id,
                decision="uncertain",
                reason="inference_error",
                num_speakers=0,
                secondary_speech_s=0.0,
                speaker_turns=(),
                model=self._model_info(),
                error=f"{type(exc).__name__}: {exc}",
            )
        return classify_vibevoice_segments(
            segments,
            audio_id=audio.source_id,
            min_secondary_speech_s=self.min_secondary_speech_s,
            model=self._model_info(),
        )

    def verify_batch(self, audios: list[Audio]) -> list[VibeVoicePurityResult]:
        """Verify candidates in configurable model batches.

        Args:
            audios: File-backed candidate segments.

        Returns:
            One result per input, in the same order.
        """
        if not self.is_loaded or self._model is None or self._processor is None:
            raise RuntimeError(
                "VibeVoice purity verifier is not loaded. Call load() first "
                "or use it as a context manager."
            )
        for audio in audios:
            if not Path(audio.path).is_file():
                raise FileNotFoundError(f"Audio file does not exist: {audio.path}")

        results: list[VibeVoicePurityResult] = []
        batch_offsets = list(range(0, len(audios), self.batch_size))
        iterator = tqdm(batch_offsets, desc="VibeVoice verification", unit="batch") if len(batch_offsets) > 1 else batch_offsets
        for offset in iterator:
            batch = audios[offset : offset + self.batch_size]
            try:
                segments_by_audio = self._infer_batch(
                    [Path(audio.path) for audio in batch]
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                results.extend(
                    _result(
                        audio_id=audio.source_id,
                        decision="uncertain",
                        reason="inference_error",
                        num_speakers=0,
                        secondary_speech_s=0.0,
                        speaker_turns=(),
                        model=self._model_info(),
                        error=error,
                    )
                    for audio in batch
                )
                continue
            results.extend(
                classify_vibevoice_segments(
                    segments,
                    audio_id=audio.source_id,
                    min_secondary_speech_s=self.min_secondary_speech_s,
                    model=self._model_info(),
                )
                for audio, segments in zip(batch, segments_by_audio, strict=True)
            )
        return results

    def _infer(self, path: Path) -> list[dict[str, Any]]:
        """Generate structured VibeVoice-ASR segments for one file."""
        import torch

        processor = self._processor
        model = self._model
        inputs = processor.apply_transcription_request(audio=str(path))
        inputs = inputs.to(self._model_runtime_device(), model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = processor.decode(generated_ids, return_format="parsed")
        return _normalize_segments(_unwrap_parsed_transcription(parsed))

    def _infer_batch(self, paths: list[Path]) -> list[list[dict[str, Any]]]:
        """Generate structured segments for multiple files in one forward pass."""
        import torch

        processor = self._processor
        model = self._model
        inputs = processor.apply_transcription_request(
            audio=[str(path) for path in paths]
        )
        inputs = inputs.to(self._model_runtime_device(), model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed_batch = processor.decode(generated_ids, return_format="parsed")
        if len(paths) == 1:
            return [_normalize_segments(_unwrap_parsed_transcription(parsed_batch))]
        if not isinstance(parsed_batch, list) or len(parsed_batch) != len(paths):
            raise VibeVoicePurityError(
                "VibeVoice-ASR returned an unexpected batch result shape"
            )
        return [_normalize_segments(parsed) for parsed in parsed_batch]

    def _model_runtime_device(self) -> Any:
        """Device tensors should land on after a full or quantized load."""
        model = self._model
        device = getattr(model, "device", None)
        if device is not None:
            return device
        try:
            return next(model.parameters()).device
        except (StopIteration, TypeError, AttributeError):
            return self._torch_device

    def _model_info(self) -> DiarizationModelInfo:
        return DiarizationModelInfo(
            backend="vibevoice-asr",
            model_id=self.model_id,
        )


def classify_vibevoice_segments(
    segments: list[dict[str, Any]],
    *,
    audio_id: str,
    min_secondary_speech_s: float = DEFAULT_MIN_SECONDARY_SPEECH_S,
    model: DiarizationModelInfo | None = None,
) -> VibeVoicePurityResult:
    """Classify speaker-count purity from already-parsed VibeVoice segments.

    Args:
        segments: Normalized dicts with ``start_time``, ``end_time``, and
            ``speaker_id``. Transcript text is ignored.
        audio_id: Candidate identity copied onto the result.
        min_secondary_speech_s: Secondary-speech reject threshold.
        model: Optional backend metadata.

    Returns:
        The purity decision for this candidate.
    """
    min_secondary_speech_s = _validate_min_secondary_speech_s(min_secondary_speech_s)
    turns, duration_by_speaker = _speaker_durations(segments)
    if not turns:
        return _result(
            audio_id=audio_id,
            decision="uncertain",
            reason="empty_output" if not segments else "no_speaker_labels",
            num_speakers=0,
            secondary_speech_s=0.0,
            speaker_turns=(),
            model=model,
        )
    num_speakers = len(duration_by_speaker)
    dominant_speaker_id = max(duration_by_speaker, key=duration_by_speaker.get)
    dominant_s = duration_by_speaker[dominant_speaker_id]
    secondary_speech_s = sum(duration_by_speaker.values()) - dominant_s
    if num_speakers == 1:
        return _result(
            audio_id=audio_id,
            decision="pass",
            reason="single_speaker",
            num_speakers=1,
            secondary_speech_s=0.0,
            speaker_turns=turns,
            dominant_speaker_id=dominant_speaker_id,
            model=model,
        )
    if secondary_speech_s >= min_secondary_speech_s:
        return _result(
            audio_id=audio_id,
            decision="reject",
            reason="multiple_speakers",
            num_speakers=num_speakers,
            secondary_speech_s=secondary_speech_s,
            speaker_turns=turns,
            dominant_speaker_id=dominant_speaker_id,
            model=model,
        )
    return _result(
        audio_id=audio_id,
        decision="uncertain",
        reason="tiny_secondary_speaker",
        num_speakers=num_speakers,
        secondary_speech_s=secondary_speech_s,
        speaker_turns=turns,
        dominant_speaker_id=dominant_speaker_id,
        model=model,
    )


def _unwrap_parsed_transcription(parsed: Any) -> Any:
    """Drop the batch wrapper from ``processor.decode(..., return_format="parsed")``.

    Transformers returns ``[[{Start, End, Speaker, Content}, ...]]`` for a
    single file. A failed parse may be a string; that is left for
    ``_normalize_segments`` to treat as empty.
    """
    if not isinstance(parsed, list) or not parsed:
        return parsed
    if isinstance(parsed[0], list):
        return parsed[0]
    return parsed


def _normalize_segments(parsed: Any) -> list[dict[str, Any]]:
    """Map VibeVoice parsed output onto start_time/end_time/speaker_id."""
    if parsed is None or isinstance(parsed, str):
        return []
    if isinstance(parsed, dict):
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = _first_present(item, ("start_time", "Start", "start", "Start time"))
        end = _first_present(item, ("end_time", "End", "end", "End time"))
        speaker = _first_present(
            item, ("speaker_id", "Speaker", "speaker", "Speaker ID")
        )
        speaker_id = _parse_speaker_id(speaker)
        try:
            start_s = float(start)
            end_s = float(end)
        except (TypeError, ValueError):
            continue
        if speaker_id is None:
            continue
        normalized.append(
            {"start_time": start_s, "end_time": end_s, "speaker_id": speaker_id}
        )
    return normalized


def _speaker_durations(
    segments: list[dict[str, Any]],
) -> tuple[tuple[VibeVoiceSpeakerTurn, ...], dict[int, float]]:
    """Drop invalid intervals and accumulate duration per speaker."""
    turns: list[VibeVoiceSpeakerTurn] = []
    duration_by_speaker: dict[int, float] = {}
    for segment in segments:
        speaker_id = _parse_speaker_id(segment.get("speaker_id"))
        try:
            start_s = float(segment["start_time"])
            end_s = float(segment["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if speaker_id is None or end_s <= start_s:
            continue
        turns.append(
            VibeVoiceSpeakerTurn(
                start_s=start_s, end_s=end_s, speaker_id=speaker_id
            )
        )
        duration_by_speaker[speaker_id] = (
            duration_by_speaker.get(speaker_id, 0.0) + (end_s - start_s)
        )
    return tuple(turns), duration_by_speaker


def _result(
    *,
    audio_id: str,
    decision: str,
    reason: str,
    num_speakers: int,
    secondary_speech_s: float,
    speaker_turns: tuple[VibeVoiceSpeakerTurn, ...],
    model: DiarizationModelInfo | None = None,
    dominant_speaker_id: int | None = None,
    error: str | None = None,
) -> VibeVoicePurityResult:
    return VibeVoicePurityResult(
        schema_version=VIBEVOICE_PURITY_SCHEMA_VERSION,
        audio_id=audio_id,
        decision=decision,
        reason=reason,
        num_speakers=num_speakers,
        secondary_speech_s=secondary_speech_s,
        speaker_turns=speaker_turns,
        dominant_speaker_id=dominant_speaker_id,
        model=model,
        error=error,
    )


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def _parse_speaker_id(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(char for char in text if char.isdigit() or char == "-")
    if not digits or digits == "-":
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def vibevoice_studio_models() -> list[dict[str, str]]:
    """Return ``id`` / ``label`` pairs for the SonicStudio checkpoint select."""
    return [
        {"id": str(choice["id"]), "label": str(choice["label"])}
        for choice in VIBEVOICE_MODEL_CHOICES
    ]


def vibevoice_model_is_quantized(model_id: str) -> bool:
    """Return whether ``model_id`` is a known bitsandbytes catalog checkpoint."""
    requested = str(model_id).strip()
    for choice in VIBEVOICE_MODEL_CHOICES:
        if str(choice["id"]) == requested:
            return bool(choice["quantized"])
    return False


def _peek_quantization_config(model_id: str, token: str | None) -> dict[str, Any]:
    """Read ``quantization_config`` from the checkpoint without loading weights."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return {}
    try:
        config = AutoConfig.from_pretrained(model_id, token=token)
    except Exception as exc:
        logger.warning(
            "Could not read VibeVoice config for %s: %s", model_id, exc
        )
        return {}
    return _as_quantization_dict(getattr(config, "quantization_config", None))


def _as_quantization_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _is_bitsandbytes_quantization(quant_config: dict[str, Any]) -> bool:
    method = str(quant_config.get("quant_method") or "").strip().lower()
    if method == "bitsandbytes":
        return True
    return bool(
        quant_config.get("load_in_4bit") or quant_config.get("load_in_8bit")
    )


def _warn_if_missing_audio_skip_modules(
    model_id: str, quant_config: dict[str, Any]
) -> None:
    skip_modules = quant_config.get("llm_int8_skip_modules") or []
    if isinstance(skip_modules, str):
        skip_names = {skip_modules}
    else:
        try:
            skip_names = {str(name) for name in skip_modules}
        except TypeError:
            skip_names = set()
    missing = [
        name for name in VIBEVOICE_AUDIO_SKIP_MODULES if name not in skip_names
    ]
    if missing:
        logger.warning(
            "VibeVoice checkpoint %s is quantized without skip_modules %s; "
            "speaker labels may collapse",
            model_id,
            missing,
        )


def _model_is_quantized(model: Any) -> bool:
    return bool(
        getattr(model, "is_quantized", False)
        or getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "quantization_method", None)
    )


def _resolve_torch_device(device: str, torch: Any) -> Any:
    requested = str(device).strip()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise VibeVoicePurityError(
            f"Requested device {requested!r} but CUDA is unavailable"
        )
    if resolved.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise VibeVoicePurityError(
                f"Requested device {requested!r} but MPS is unavailable"
            )
    return resolved


def _is_unsupported_attention_error(exc: BaseException) -> bool:
    """Return whether Transformers rejected an attention backend for the model."""
    message = str(exc).lower()
    return (
        "does not support an attention implementation" in message
        and "scaled_dot_product_attention" in message
    )


def _validate_min_secondary_speech_s(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("min_secondary_speech_s must be a number")
    seconds = float(value)
    if seconds < 0:
        raise ValueError("min_secondary_speech_s must be non-negative")
    return seconds


def _validate_max_new_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_new_tokens must be an integer")
    if value <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    return value


def _validate_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("batch_size must be an integer")
    if value < 1 or value > 256:
        raise ValueError("batch_size must be from 1 to 256")
    return value
