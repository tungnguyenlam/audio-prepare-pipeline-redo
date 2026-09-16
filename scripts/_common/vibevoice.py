"""Shared VibeVoice-ASR checkpoint selection and loading for ASR and verifier."""
from __future__ import annotations

import argparse
import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

DEFAULT_VIBEVOICE_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
QUANTIZATION_NONE = "none"
QUANTIZATION_INT8 = "int8"
QUANTIZATION_NF4 = "nf4"
QUANTIZATION_ALIASES = {
    "none": QUANTIZATION_NONE,
    "bf16": QUANTIZATION_NONE,
    "int8": QUANTIZATION_INT8,
    "8bit": QUANTIZATION_INT8,
    "nf4": QUANTIZATION_NF4,
    "int4": QUANTIZATION_NF4,
    "4bit": QUANTIZATION_NF4,
}
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
        "quantization": QUANTIZATION_NONE,
        "quantized": False,
        "label": "Full BF16 (~17 GB VRAM)",
    },
    {
        "id": "Dubedo/VibeVoice-ASR-HF-INT8",
        "quantization": QUANTIZATION_INT8,
        "quantized": True,
        "label": "INT8 (~10–11 GB VRAM)",
    },
    {
        "id": "Dubedo/VibeVoice-ASR-HF-NF4",
        "quantization": QUANTIZATION_NF4,
        "quantized": True,
        "label": "NF4 4-bit (~7–8 GB VRAM)",
    },
)
QUANTIZATION_MODELS = {
    str(choice["quantization"]): str(choice["id"]) for choice in VIBEVOICE_MODEL_CHOICES
}


class LoadedVibeVoice(NamedTuple):
    processor: Any
    model: Any
    model_id: str
    quantization: str
    device: str
    quantized: bool


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add --model-id and --quantization flags used by both VibeVoice commands."""
    parser.add_argument(
        "-m",
        "--model-id",
        default=DEFAULT_VIBEVOICE_MODEL_ID,
        help=(
            "VibeVoice model ID on Hugging Face or local checkpoint directory "
            f"(default: {DEFAULT_VIBEVOICE_MODEL_ID}; --quantization int8/nf4 "
            "replaces this default with the matching Dubedo checkpoint)"
        ),
    )
    parser.add_argument(
        "-q",
        "--quantization",
        default=QUANTIZATION_NONE,
        type=normalize_quantization,
        choices=(QUANTIZATION_NONE, QUANTIZATION_INT8, QUANTIZATION_NF4),
        help=(
            'Checkpoint catalog: "none" full BF16 (default), "int8" '
            "Dubedo/VibeVoice-ASR-HF-INT8, \"nf4\" (alias \"int4\") "
            "Dubedo/VibeVoice-ASR-HF-NF4. Quantized checkpoints need NVIDIA CUDA "
            "and bitsandbytes>=0.48.1"
        ),
    )


def normalize_quantization(value: str) -> str:
    """Map CLI aliases such as int4 onto the catalog names none/int8/nf4."""
    key = str(value).strip().lower()
    if key not in QUANTIZATION_ALIASES:
        allowed = ", ".join(sorted(QUANTIZATION_ALIASES))
        raise ValueError(f"Unknown VibeVoice quantization {value!r}; expected one of: {allowed}")
    return QUANTIZATION_ALIASES[key]


def catalog_quantization(model_id: str) -> str:
    """Return the catalog quantization for a known checkpoint, else none."""
    requested = str(model_id).strip()
    for choice in VIBEVOICE_MODEL_CHOICES:
        if str(choice["id"]) == requested:
            return str(choice["quantization"])
    return QUANTIZATION_NONE


def catalog_is_quantized(model_id: str) -> bool:
    """Return whether model_id is a known bitsandbytes catalog checkpoint."""
    requested = str(model_id).strip()
    for choice in VIBEVOICE_MODEL_CHOICES:
        if str(choice["id"]) == requested:
            return bool(choice["quantized"])
    return False


def resolve_checkpoint(model_id: str, quantization: str) -> tuple[str, str]:
    """Resolve the Hugging Face ID and catalog quantization name.

    ``--quantization int8|nf4`` selects the matching Dubedo checkpoint when
    ``--model-id`` is still the default BF16 repo. A custom ``--model-id`` is
    kept, and quantized loading still applies when the flag or catalog says so.
    """
    requested = str(model_id).strip() or DEFAULT_VIBEVOICE_MODEL_ID
    quant = normalize_quantization(quantization)
    if quant in {QUANTIZATION_INT8, QUANTIZATION_NF4}:
        catalog_id = QUANTIZATION_MODELS[quant]
        if requested == DEFAULT_VIBEVOICE_MODEL_ID:
            requested = catalog_id
        return requested, quant
    inferred = catalog_quantization(requested)
    return requested, inferred


def load_vibevoice(
    model_id: str,
    *,
    device: str = "auto",
    quantization: str = QUANTIZATION_NONE,
    token: str | None = None,
    attn_implementation: str = "eager",
) -> LoadedVibeVoice:
    """Load processor + model, using device_map for bitsandbytes checkpoints."""
    import torch
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    resolved_id, quant = resolve_checkpoint(model_id, quantization)
    resolved_device = _resolve_device_string(device, torch)
    hf_token = token
    quant_config = _peek_quantization_config(resolved_id, hf_token)
    quantized = (
        quant != QUANTIZATION_NONE
        or catalog_is_quantized(resolved_id)
        or _is_bitsandbytes_quantization(quant_config)
    )
    if quantized and not _bitsandbytes_cuda_ok(torch, resolved_device):
        raise RuntimeError(
            f"Quantized VibeVoice checkpoint {resolved_id!r} requires NVIDIA CUDA "
            "(bitsandbytes does not run on CPU, MPS, or ROCm/HIP). "
            "Use --quantization none on this machine."
        )
    dtype = torch.bfloat16 if str(resolved_device).startswith("cuda") else torch.float32
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if hf_token:
        load_kwargs["token"] = hf_token
    if quantized:
        load_kwargs["device_map"] = {"": _cuda_device_index(resolved_device)}
        if quant_config:
            _warn_if_missing_audio_skip_modules(resolved_id, quant_config)

    processor = AutoProcessor.from_pretrained(resolved_id, token=hf_token)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(resolved_id, **load_kwargs)
    if not _model_is_quantized(model):
        model = model.to(resolved_device)
    model.eval()
    runtime = _model_runtime_device(model, resolved_device)
    logger.info(
        "Loaded VibeVoice-ASR model=%s device=%s quantization=%s quantized=%s",
        resolved_id,
        runtime,
        quant,
        quantized or _model_is_quantized(model),
    )
    return LoadedVibeVoice(
        processor=processor,
        model=model,
        model_id=resolved_id,
        quantization=quant,
        device=str(runtime),
        quantized=bool(quantized or _model_is_quantized(model)),
    )


def _resolve_device_string(device: str, torch: Any) -> str:
    requested = str(device).strip() or "auto"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return requested


def _bitsandbytes_cuda_ok(torch: Any, device: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    hip = getattr(torch.version, "hip", None)
    return not bool(hip)


def _cuda_device_index(device: str) -> int:
    if ":" not in str(device):
        return 0
    try:
        return int(str(device).rsplit(":", 1)[-1])
    except ValueError:
        return 0


def _model_runtime_device(model: Any, fallback: str) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (StopIteration, TypeError, AttributeError):
        return fallback


def _peek_quantization_config(model_id: str, token: str | None) -> dict[str, Any]:
    try:
        from transformers import AutoConfig
    except ImportError:
        return {}
    try:
        config = AutoConfig.from_pretrained(model_id, token=token)
    except Exception as exc:
        logger.warning("Could not read VibeVoice config for %s: %s", model_id, exc)
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
    return bool(quant_config.get("load_in_4bit") or quant_config.get("load_in_8bit"))


def _warn_if_missing_audio_skip_modules(model_id: str, quant_config: dict[str, Any]) -> None:
    skip_modules = quant_config.get("llm_int8_skip_modules") or []
    if isinstance(skip_modules, str):
        skip_names = {skip_modules}
    else:
        try:
            skip_names = {str(name) for name in skip_modules}
        except TypeError:
            skip_names = set()
    missing = [name for name in VIBEVOICE_AUDIO_SKIP_MODULES if name not in skip_names]
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
