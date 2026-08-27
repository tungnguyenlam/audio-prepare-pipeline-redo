"""JSON-lines worker entrypoint for isolated VibeVoice-ASR purity inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

from src.diarization.VibeVoicePurityVerifier import VibeVoicePurityVerifier
from src.utils.AudioClass import Audio

_PROTOCOL_PREFIX = "@@VIBEVOICE_PURITY_RPC@@"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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
    """Serve load, verify, and close commands over standard input/output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    verifier: VibeVoicePurityVerifier | None = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                action = request.get("action")
                if action == "load":
                    if verifier is not None:
                        raise RuntimeError("VibeVoice purity worker is already loaded")
                    verifier = VibeVoicePurityVerifier(**request["config"])
                    verifier.load()
                    _respond(result={"status": "loaded"})
                elif action == "verify":
                    if verifier is None:
                        raise RuntimeError("VibeVoice purity worker is not loaded")
                    result = verifier.verify(_audio_from_dict(request["audio"]))
                    _respond(result=result.to_dict())
                elif action == "verify_batch":
                    if verifier is None:
                        raise RuntimeError("VibeVoice purity worker is not loaded")
                    results = verifier.verify_batch(
                        [_audio_from_dict(item) for item in request["audios"]]
                    )
                    _respond(result=[result.to_dict() for result in results])
                elif action == "close":
                    if verifier is not None:
                        verifier.unload()
                        verifier = None
                    _respond(result={"status": "closed"})
                    return
                else:
                    raise ValueError(f"Unknown VibeVoice purity worker action: {action}")
            except Exception as exc:
                _respond(error=exc)
    finally:
        if verifier is not None:
            verifier.unload()


if __name__ == "__main__":
    main()
