"""VibeVoice-ASR transcription with optional Whisper forced word alignment."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (
    batch,
    destinations,
    identity,
    parser,
    probe,
    progress,
    read_json,
    request,
    write_json,
)

logger = logging.getLogger(__name__)

DEFAULT_VIBEVOICE_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
DEFAULT_MAX_NEW_TOKENS = 2048


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.ss string."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def _unwrap_parsed(parsed: Any) -> Any:
    """Extract transcription payload from parsed decoder output."""
    if hasattr(parsed, "transcription"):
        return getattr(parsed, "transcription")
    if isinstance(parsed, dict) and "transcription" in parsed:
        return parsed["transcription"]
    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
        return parsed[0]
    return parsed


def _extract_segments(transcription: Any) -> list[dict[str, Any]]:
    """Normalize raw parsed speaker turns from VibeVoice processor."""
    if not isinstance(transcription, list):
        return []
    segments = []
    for item in transcription:
        if not isinstance(item, dict):
            continue
        start = item.get("Start", item.get("start_time", item.get("start", 0.0)))
        end = item.get("End", item.get("end_time", item.get("end", 0.0)))
        speaker = item.get("Speaker", item.get("speaker_id", item.get("speaker", "0")))
        content = item.get("Content", item.get("text", item.get("content", "")))
        try:
            start_f = float(start)
        except (ValueError, TypeError):
            start_f = 0.0
        try:
            end_f = float(end)
        except (ValueError, TypeError):
            end_f = 0.0
        segments.append({
            "speaker_id": str(speaker),
            "start_time": start_f,
            "end_time": end_f,
            "text": str(content).strip(),
        })
    return segments


def _write_text_file(target_path: Path, lines: list[str]) -> None:
    """Atomically write text lines to disk."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target_path.parent, prefix="." + target_path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        os.replace(tmp_name, target_path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def main() -> int:
    p = parser(__doc__, "asr", "vibevoice")
    p.add_argument(
        "--model-id",
        default=DEFAULT_VIBEVOICE_MODEL_ID,
        help="VibeVoice model ID on Hugging Face or local checkpoint directory (default: microsoft/VibeVoice-ASR-HF)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help='Compute device for model inference ("auto", "cpu", "cuda", or "hip") (default: auto)',
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate (default: 2048)",
    )
    p.add_argument(
        "--align-words",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align word-level timestamps using Whisper cross-attention forced alignment (default: True)",
    )
    p.add_argument(
        "--align-model",
        default="base",
        help="Whisper model variant for word-level alignment (default: base)",
    )
    p.add_argument(
        "--align-language",
        default=None,
        help='Language code hint for Whisper tokenizer alignment (e.g. "vi", "en", default: None/multilingual)',
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print live transcription turns and aligned words to stderr (default: False)",
    )
    args = p.parse_args()

    pairs = destinations(args, suffix="_vibevoice", extension=".json")
    if not pairs:
        progress("empty", "No audio files found matching input criteria.")
        return 0

    import torch
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    hf_token = os.getenv("HF_TOKEN")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    progress("load", f"Loading VibeVoice model '{args.model_id}' on {device} ({dtype})...")
    with contextlib.redirect_stdout(sys.stderr):
        processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            args.model_id, dtype=dtype, token=hf_token, attn_implementation="eager"
        ).to(device).eval()

    whisper_model = None
    whisper_tokenizer = None
    if args.align_words:
        import whisper
        import whisper.timing
        progress("load", f"Loading Whisper alignment model '{args.align_model}' on {device}...")
        with contextlib.redirect_stdout(sys.stderr):
            whisper_model = whisper.load_model(args.align_model, device=device)
            whisper_tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True,
                language=args.align_language or "vi",
                task="transcribe",
            )

    parameters = {
        "model_id": args.model_id,
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "align_words": bool(args.align_words),
        "align_model": args.align_model if args.align_words else None,
        "align_language": args.align_language if args.align_words else None,
    }

    def process(src: Path, dest: Path) -> None:
        src_identity = identity(src)
        wanted = request(src_identity, "asr", parameters, "vibevoice")

        # Skip if cached and matching parameters
        if not args.overwrite and dest.is_file():
            try:
                old = read_json(dest)
                if (
                    old.get("schema_version") == 1
                    and old.get("source", {}).get("sha256") == src_identity["sha256"]
                    and old.get("parameters") == parameters
                ):
                    progress("skip", f"{src.name} -> {dest.name}")
                    print(dest)
                    return
            except Exception:
                pass

        t0 = time.time()
        progress("transcribing", src.name)

        # 1. Run VibeVoice inference
        inputs = processor.apply_transcription_request(audio=str(src))
        inputs = inputs.to(device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = processor.decode(generated_ids, return_format="parsed")
        transcription = _unwrap_parsed(parsed)
        raw_segments = _extract_segments(transcription)

        # 2. Extract audio metadata & waveform for Whisper forced alignment
        audio_info = probe(src)
        audio_duration_s = float(audio_info.get("duration_s", 0.0))

        waveform = None
        if args.align_words and whisper_model is not None and raw_segments:
            import whisper
            try:
                waveform = whisper.load_audio(str(src))
                if len(waveform) > 0 and audio_duration_s <= 0.0:
                    audio_duration_s = float(len(waveform)) / float(whisper.audio.SAMPLE_RATE)
            except Exception as load_err:
                logger.warning("Failed to load audio waveform for Whisper alignment: %s", load_err)

        # 3. Process segments and align words
        processed_turns: list[dict[str, Any]] = []
        for seg in raw_segments:
            st = max(0.0, float(seg["start_time"]))
            et = min(audio_duration_s if audio_duration_s > 0 else float("inf"), float(seg["end_time"]))
            if et <= st:
                et = st + 0.1
            text = seg["text"]
            words: list[dict[str, Any]] = []

            if waveform is not None and text and args.align_words and whisper_model is not None:
                import whisper
                import whisper.timing
                sr = whisper.audio.SAMPLE_RATE
                start_sample = int(st * sr)
                end_sample = int(et * sr)
                # Small 50ms padding around the turn boundary
                pad = int(0.05 * sr)
                slice_start = max(0, start_sample - pad)
                slice_end = min(len(waveform), end_sample + pad)
                turn_audio = waveform[slice_start:slice_end]

                if len(turn_audio) > 160:
                    try:
                        turn_tensor = torch.from_numpy(turn_audio).to(device)
                        mel = whisper.log_mel_spectrogram(
                            whisper.pad_or_trim(turn_tensor),
                            n_mels=whisper_model.dims.n_mels,
                        )
                        num_frames = min(3000, int(len(turn_audio) / whisper.audio.HOP_LENGTH))
                        text_tokens = whisper_tokenizer.encode(text)
                        # Whisper decoder max positional context is 448 tokens
                        if 0 < len(text_tokens) <= 440:
                            timings = whisper.timing.find_alignment(
                                whisper_model, whisper_tokenizer, text_tokens, mel, num_frames
                            )
                            slice_base_s = float(slice_start) / float(sr)
                            for t in timings:
                                word_cleaned = t.word.strip()
                                if not word_cleaned:
                                    continue
                                w_start = round(slice_base_s + float(t.start), 3)
                                w_end = round(slice_base_s + float(t.end), 3)
                                words.append({
                                    "word": word_cleaned,
                                    "start": w_start,
                                    "end": w_end,
                                    "probability": round(float(t.probability), 3),
                                })
                    except Exception as align_err:
                        logger.warning("Forced alignment error for turn [%.2f, %.2f]: %s", st, et, align_err)

            processed_turns.append({
                "speaker_id": seg["speaker_id"],
                "start_time": round(st, 3),
                "end_time": round(et, 3),
                "duration_s": round(et - st, 3),
                "text": text,
                "words": words,
            })

        full_text = " ".join(t["text"] for t in processed_turns if t["text"]).strip()
        elapsed_s = round(time.time() - t0, 3)

        # 4. Verbose terminal logging
        if args.verbose:
            border = "=" * 70
            print(f"\n{border}", file=sys.stderr)
            print(f"[{src.name}] TRANSCRIBED OUTPUT (latency: {elapsed_s:.2f}s):", file=sys.stderr)
            print("-" * 70, file=sys.stderr)
            for turn in processed_turns:
                spk = turn["speaker_id"]
                st_str = format_timestamp(turn["start_time"])
                et_str = format_timestamp(turn["end_time"])
                print(f"[{st_str} -> {et_str}] [Speaker {spk}]: {turn['text']}", file=sys.stderr)
                if turn.get("words"):
                    words_summary = " ".join(
                        f"[{w['word']} {w['start']:.2f}-{w['end']:.2f}]" for w in turn["words"]
                    )
                    print(f"  └─ Words: {words_summary}", file=sys.stderr)
            print("-" * 70, file=sys.stderr)
            print(f"FULL TEXT: {full_text}", file=sys.stderr)
            print(f"{border}\n", file=sys.stderr, flush=True)

        # 5. Write artifacts
        output_payload = {
            **wanted,
            "duration_s": round(audio_duration_s, 3),
            "text": full_text,
            "turns": processed_turns,
        }
        write_json(dest, output_payload)

        # Write companion plain-text transcript
        txt_dest = dest.with_suffix(".txt")
        txt_lines = [
            f"[{format_timestamp(t['start_time'])} -> {format_timestamp(t['end_time'])}] [Speaker {t['speaker_id']}]: {t['text']}"
            for t in processed_turns
        ]
        txt_lines.append("")
        txt_lines.append("--- FULL TRANSCRIPT ---")
        txt_lines.append(full_text)
        _write_text_file(txt_dest, txt_lines)

        progress("done", f"{src.name} -> {dest.name}", elapsed_s=elapsed_s)
        print(dest)

    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
