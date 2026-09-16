"""Verify single-speaker purity from VibeVoice-ASR speaker counts."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _cli import resolved_parameters, run_verifier
from _common.files import destinations, parser
from _common.vibevoice import DEFAULT_VIBEVOICE_MODEL_ID, add_checkpoint_args, load_vibevoice

logger = logging.getLogger(__name__)

DEFAULT_MIN_SECONDARY_SPEECH_S = 0.25
DEFAULT_MAX_NEW_TOKENS = 2048


def _unwrap_parsed(parsed: Any) -> Any:
    if hasattr(parsed, "transcription"):
        return getattr(parsed, "transcription")
    if isinstance(parsed, dict) and "transcription" in parsed:
        return parsed["transcription"]
    return parsed


def _normalize_segments(transcription: Any) -> list[dict[str, Any]]:
    if not isinstance(transcription, list):
        return []
    segments = []
    for item in transcription:
        if not isinstance(item, dict):
            continue
        start = item.get("start_time") or item.get("start") or 0.0
        end = item.get("end_time") or item.get("end") or 0.0
        spk = item.get("speaker_id") or item.get("speaker") or "UNKNOWN"
        segments.append({"start_time": float(start), "end_time": float(end), "speaker_id": str(spk)})
    return segments


def classify_segments(segments: list[dict[str, Any]], min_secondary_speech_s: float = DEFAULT_MIN_SECONDARY_SPEECH_S) -> dict[str, Any]:
    if not segments:
        return {"decision": "uncertain", "reason": "empty_output", "num_speakers": 0, "secondary_speech_s": 0.0}
    durations: dict[str, float] = {}
    for s in segments:
        dur = max(0.0, s["end_time"] - s["start_time"])
        durations[s["speaker_id"]] = durations.get(s["speaker_id"], 0.0) + dur
    if not durations:
        return {"decision": "uncertain", "reason": "no_speaker_labels", "num_speakers": 0, "secondary_speech_s": 0.0}
    num_speakers = len(durations)
    dominant_id = max(durations, key=durations.get)
    dominant_s = durations[dominant_id]
    secondary_s = sum(durations.values()) - dominant_s
    if num_speakers == 1:
        return {"decision": "pass", "reason": "single_speaker", "num_speakers": 1,
                "secondary_speech_s": 0.0, "dominant_speaker_id": dominant_id}
    if secondary_s >= min_secondary_speech_s:
        return {"decision": "reject", "reason": "multiple_speakers", "num_speakers": num_speakers,
                "secondary_speech_s": round(secondary_s, 3), "dominant_speaker_id": dominant_id}
    return {"decision": "uncertain", "reason": "tiny_secondary_speaker", "num_speakers": num_speakers,
            "secondary_speech_s": round(secondary_s, 3), "dominant_speaker_id": dominant_id}


class VibeVoiceVerifier:
    def __init__(self, model_id: str = DEFAULT_VIBEVOICE_MODEL_ID, device: str = "auto",
                 quantization: str = "none", max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 token: str | None = None) -> None:
        loaded = load_vibevoice(
            model_id, device=device, quantization=quantization, token=token or os.getenv("HF_TOKEN"),
        )
        self.processor = loaded.processor
        self.model = loaded.model
        self.model_id = loaded.model_id
        self.quantization = loaded.quantization
        self.device = loaded.device
        self.max_new_tokens = max_new_tokens

    def verify(self, audio_path: Path, min_secondary_speech_s: float = DEFAULT_MIN_SECONDARY_SPEECH_S) -> dict[str, Any]:
        import torch
        t0 = time.time()
        inputs = self.processor.apply_transcription_request(audio=str(audio_path))
        inputs = inputs.to(self.device, self.model.dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.processor.decode(generated_ids, return_format="parsed")
        transcription = _unwrap_parsed(parsed)
        segments = _normalize_segments(transcription)
        verdict = classify_segments(segments, min_secondary_speech_s)
        try:
            raw_response = json.dumps(
                transcription,
                ensure_ascii=False,
                default=lambda value: (
                    value.model_dump()
                    if hasattr(value, "model_dump")
                    else vars(value)
                    if hasattr(value, "__dict__")
                    else str(value)
                ),
            )
        except (TypeError, ValueError):
            raw_response = str(transcription)
        verdict["_raw_response"] = raw_response
        verdict["_response_kind"] = "structured"
        verdict["_latency_s"] = round(time.time() - t0, 3)
        return verdict


def main() -> int:
    p = parser('Verify audio with VibeVoice-ASR speaker counts; writes verdicts without filtering audio.', 's4-agent/verifier', 'vibevoice')
    add_checkpoint_args(p)
    p.add_argument('--device', default='auto', help='Inference device ("auto", "cpu", "cuda", or "hip")')
    p.add_argument('--max-new-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS, help='Maximum number of tokens to generate')
    p.add_argument('--min-secondary-speech-s', type=float, default=DEFAULT_MIN_SECONDARY_SPEECH_S, help='Minimum duration in seconds of secondary speaker speech to trigger rejection')
    args = p.parse_args()

    pairs = destinations(args, '_vibevoice', '.json')
    parameters = {
        'model_id': args.model_id, 'quantization': args.quantization, 'device': args.device,
        'max_new_tokens': args.max_new_tokens,
        'min_secondary_speech_s': args.min_secondary_speech_s
    }
    with contextlib.redirect_stdout(sys.stderr):
        verifier = VibeVoiceVerifier(
            model_id=args.model_id, device=args.device, quantization=args.quantization,
            max_new_tokens=args.max_new_tokens,
        )

    parameters = resolved_parameters(parameters, verifier)
    return run_verifier(
        args=args, pairs=pairs, backend='vibevoice', parameters=parameters,
        verify=lambda source: verifier.verify(
            source, min_secondary_speech_s=args.min_secondary_speech_s
        ),
    )


if __name__ == '__main__':
    raise SystemExit(main())
