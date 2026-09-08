from __future__ import annotations

from typing import Any

from src.diarization.verifiers.BaseVerifier import (
    BaseVerifier,
    DEFAULT_ACOUSTIC_PROMPT,
    extract_json_payload,
    load_audio_waveform,
)
from src.diarization.verifiers.DefaultHFVerifier import DefaultHFVerifier
from src.diarization.verifiers.MossAudioVerifier import MossAudioVerifier
from src.diarization.verifiers.MiniCPMVerifier import MiniCPMVerifier
from src.diarization.verifiers.KimiAudioVerifier import KimiAudioVerifier
from src.diarization.verifiers.GeminiVerifier import GeminiVerifier
from src.diarization.verifiers.EndpointVerifier import EndpointVerifier
from src.diarization.verifiers.UnslothVerifier import UnslothVerifier


def get_verifier(
    backend: str = "hf_local",
    model: str = "google/gemma-4-E2B-it",
    device: str = "auto",
    adapter_path: str | None = None,
    endpoint: str = "http://localhost:8000/v1/chat/completions",
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    reasoning_effort: str = "medium",
    api_key: str | None = None,
    hf_token: str | None = None,
    gguf_variant: str | None = None,
    **kwargs: Any,
) -> BaseVerifier:
    """Factory helper to instantiate the appropriate speech acoustic verifier.

    Routes to specialized verifiers for models requiring special loaders (MOSS-Audio,
    MiniCPM-o, Kimi-Audio) or defaults to DefaultHFVerifier, GeminiVerifier,
    UnslothVerifier, or EndpointVerifier.

    Args:
        backend: Evaluation backend ('hf_local', 'gemini', 'endpoint', 'unsloth').
        model: Model name, path, or Hugging Face repository ID.
        device: Target execution device ('auto', 'cuda:0', 'cpu', 'mps').
        adapter_path: Optional LoRA adapter path or Hugging Face Hub ID.
        endpoint: API endpoint URL for backend='endpoint' or 'unsloth'.
        trust_remote_code: Whether to allow remote HF code execution.
        torch_dtype: Model tensor precision ('bfloat16', 'float16', 'float32').
        reasoning_effort: Thinking level for Gemini ('none', 'low', 'medium', 'high').
        api_key: Optional API key override (Gemini, Endpoint, or Unsloth).
        hf_token: Optional Hugging Face authentication token.
        gguf_variant: Optional GGUF quantization variant for Unsloth models (e.g. 'Q8_0').

    Returns:
        Configured BaseVerifier instance.
    """
    backend_lower = backend.lower()
    if backend_lower == "gemini":
        return GeminiVerifier(
            model=model,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

    if backend_lower in {"unsloth", "unsloth_local", "unsloth_studio"}:
        return UnslothVerifier(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            gguf_variant=gguf_variant,
            **kwargs,
        )

    if backend_lower == "endpoint":
        # If the target or key clearly indicates Unsloth Studio, use UnslothVerifier for maximum compatibility
        if (
            "unsloth" in (model or "").lower()
            or "unsloth" in (endpoint or "").lower()
            or (api_key and "unsloth" in api_key.lower())
            or gguf_variant is not None
        ):
            return UnslothVerifier(
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                gguf_variant=gguf_variant,
                **kwargs,
            )
        return EndpointVerifier(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            **kwargs,
        )

    # Local model backend
    model_lower = model.lower()
    if "moss" in model_lower:
        return MossAudioVerifier(
            model_id=model,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            adapter_path=adapter_path,
            hf_token=hf_token,
            **kwargs,
        )

    if "kimi" in model_lower:
        return KimiAudioVerifier(
            model_id=model,
            device=device,
            adapter_path=adapter_path,
            **kwargs,
        )

    if "minicpm" in model_lower:
        return MiniCPMVerifier(
            model_id=model,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            adapter_path=adapter_path,
            hf_token=hf_token,
            **kwargs,
        )

    return DefaultHFVerifier(
        model_id=model,
        device=device,
        adapter_path=adapter_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        hf_token=hf_token,
        **kwargs,
    )


__all__ = [
    "BaseVerifier",
    "DefaultHFVerifier",
    "MossAudioVerifier",
    "MiniCPMVerifier",
    "KimiAudioVerifier",
    "GeminiVerifier",
    "EndpointVerifier",
    "UnslothVerifier",
    "DEFAULT_ACOUSTIC_PROMPT",
    "extract_json_payload",
    "load_audio_waveform",
    "get_verifier",
]
