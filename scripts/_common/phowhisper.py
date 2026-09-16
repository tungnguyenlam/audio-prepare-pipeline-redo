"""PhoWhisper size aliases shared by ASR commands."""
from __future__ import annotations

PHOWHISPER_SIZES = ("tiny", "base", "small", "medium", "large")
DEFAULT_PHOWHISPER_SIZE = "large"
DEFAULT_PHOWHISPER_MODEL_ID = f"vinai/PhoWhisper-{DEFAULT_PHOWHISPER_SIZE}"
_SIZE_HELP = ", ".join(PHOWHISPER_SIZES)


def resolve_model_id(model: str) -> str:
    """Map a PhoWhisper size name to `vinai/PhoWhisper-<size>`; pass other ids through."""
    raw = model.strip()
    lowered = raw.lower()
    size = lowered.removeprefix("vinai/phowhisper-").removeprefix("phowhisper-")
    if size in PHOWHISPER_SIZES:
        return f"vinai/PhoWhisper-{size}"
    return raw


def model_flag_help(*, default_size: str = DEFAULT_PHOWHISPER_SIZE) -> str:
    return (
        f"PhoWhisper size ({_SIZE_HELP}; default: {default_size}) "
        "or a Hugging Face / OpenAI Whisper identifier"
    )
