"""JSON-lines worker entrypoint for isolated NeMo clustering diarization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

from src.diarization.ClusteringDiarizer import ClusteringDiarizer
from src.utils.AudioClass import Audio

_PROTOCOL_PREFIX = "@@CLUSTERING_RPC@@"


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
    diarizer: ClusteringDiarizer | None = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                action = request.get("action")
                if action == "load":
                    if diarizer is not None:
                        raise RuntimeError("Clustering worker is already loaded")
                    diarizer = ClusteringDiarizer(**request["config"])
                    diarizer.load()
                    _respond(result={"status": "loaded"})
                elif action == "diarize":
                    if diarizer is None or not diarizer.is_loaded:
                        raise RuntimeError("Clustering worker is not loaded")
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
                    raise ValueError(f"Unknown clustering worker action: {action!r}")
            except Exception as exc:
                _respond(error=exc)
    finally:
        if diarizer is not None:
            diarizer.unload()


if __name__ == "__main__":
    main()
