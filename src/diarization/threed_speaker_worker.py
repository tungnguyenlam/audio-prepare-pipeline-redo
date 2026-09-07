"""JSON-lines worker entrypoint for isolated 3D-Speaker diarization."""

from __future__ import annotations

import os

# Prevent inheriting Jupyter inline backend in headless worker
os.environ["MPLBACKEND"] = "Agg"

import torch
import soundfile as sf
import torchaudio
if not hasattr(torchaudio, "AudioMetaData"):
    from dataclasses import dataclass
    @dataclass
    class AudioMetaData:
        sample_rate: int = 16000
        num_frames: int = 0
        num_channels: int = 1
        bits_per_sample: int = 16
        encoding: str = "PCM_S"
    torchaudio.AudioMetaData = AudioMetaData
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

_orig_load = getattr(torchaudio, "load", None)
def _safe_load(filepath, *args, **kwargs):
    try:
        data, sr = sf.read(filepath, dtype="float32")
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.t()
        return tensor, sr
    except Exception:
        if _orig_load is not None:
            return _orig_load(filepath, *args, **kwargs)
        raise
torchaudio.load = _safe_load

import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

from src.diarization.ThreeDSpeakerDiarizer import ThreeDSpeakerDiarizer
from src.utils.AudioClass import Audio

_PROTOCOL_PREFIX = "@@THREEDSPEAKER_RPC@@"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _respond(*, result: Any = None, error: BaseException | None = None) -> None:
    if error is None:
        payload = {"ok": True, "result": result}
    else:
        payload = {
            "ok": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    print(
        _PROTOCOL_PREFIX
        + json.dumps(payload, ensure_ascii=False, default=_json_default),
        flush=True,
    )


def _audio_from_dict(payload: dict[str, Any]) -> Audio:
    return Audio(
        path=Path(payload["path"]),
        source_id=payload["source_id"],
        title=payload.get("title"),
        source_url=payload.get("source_url"),
        channel_id=payload.get("channel_id"),
        channel_name=payload.get("channel_name"),
        channel_url=payload.get("channel_url"),
        sample_rate=payload.get("sample_rate"),
        duration_s=payload.get("duration_s"),
        channels=payload.get("channels"),
        format=payload.get("format", "wav"),
        native_sample_rate=payload.get("native_sample_rate"),
        history=tuple(payload.get("history", [])),
    )


def main() -> None:
    """Serve load, diarize, and close commands over standard input/output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    diarizer: ThreeDSpeakerDiarizer | None = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                action = request.get("action")
                if action == "load":
                    if diarizer is not None:
                        raise RuntimeError("3D-Speaker worker is already loaded")
                    diarizer = ThreeDSpeakerDiarizer(**request["config"])
                    diarizer.load()
                    _respond(result={"status": "loaded"})
                elif action == "diarize":
                    if diarizer is None or not diarizer.is_loaded:
                        raise RuntimeError("3D-Speaker worker is not loaded")
                    kwargs: dict[str, Any] = {}
                    if "num_speakers" in request:
                        kwargs["num_speakers"] = request["num_speakers"]
                    result = diarizer.diarize(
                        _audio_from_dict(request["audio"]),
                        **kwargs,
                    )
                    _respond(result=result.to_dict())
                elif action == "close":
                    if diarizer is not None:
                        diarizer.unload()
                        diarizer = None
                    _respond(result={"status": "closed"})
                    return
                else:
                    raise ValueError(f"Unknown 3D-Speaker worker action: {action!r}")
            except Exception as exc:
                _respond(error=exc)
    finally:
        if diarizer is not None:
            diarizer.unload()


if __name__ == "__main__":
    main()
