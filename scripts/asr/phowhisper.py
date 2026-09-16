"""PhoWhisper ASR transcription with word-level forced alignment."""
from __future__ import annotations

import argparse
import contextlib
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

DEFAULT_MODEL_ID = "vinai/PhoWhisper-small"


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.ss string."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


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


def _load_phowhisper_model(model_name: str, device: Any) -> Any:
    """Load PhoWhisper (or Whisper) model with whisper_timestamped."""
    import whisper

    try:
        import whisper_timestamped as whisperts
    except ImportError:
        whisperts = None

    whisper_model = None
    if whisperts is not None:
        try:
            whisper_model = whisperts.load_model(model_name, device=device)
        except Exception as wt_err:
            logger.debug("whisper_timestamped loader failed for %s: %s", model_name, wt_err)
            whisper_model = None

    if whisper_model is None:
        whisper_model = whisper.load_model(model_name, device=device)

    # Register alignment heads if not present
    if not hasattr(whisper_model, "alignment_heads") or getattr(whisper_model, "alignment_heads") is None:
        try:
            import whisper_timestamped as whisperts

            g = whisperts.transcribe_timestamped.__globals__
            _get_heads = g.get("_get_alignment_heads")
            if _get_heads is not None:
                n_layers = getattr(whisper_model.dims, "n_text_layer", 12)
                n_heads = getattr(whisper_model.dims, "n_text_head", 12)
                size_map = {
                    (4, 6): "tiny",
                    (6, 8): "base",
                    (12, 12): "small",
                    (24, 16): "medium",
                    (32, 20): "large-v3",
                }
                variant = size_map.get((n_layers, n_heads), "small")
                heads = _get_heads(variant, n_layers, n_heads).to(device)
                whisper_model.register_buffer("alignment_heads", heads, persistent=False)
        except Exception as head_err:
            logger.warning("Could not set alignment heads for %s: %s", model_name, head_err)

    return whisper_model


def main() -> int:
    p = parser(__doc__, "asr", "phowhisper")
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"PhoWhisper/Whisper model name or Hugging Face repository (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--device",
        default="auto",
        help='Compute device for model inference ("auto", "cpu", "cuda", or "hip") (default: auto)',
    )
    p.add_argument(
        "--language",
        default="vi",
        help='Language code for transcription and tokenizer alignment (default: "vi")',
    )
    p.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Number of beams in beam search (default: 1 for greedy decoding)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    p.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Silero VAD for pre-segmentation in whisper-timestamped (default: True)",
    )
    p.add_argument(
        "--align-words",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate word-level timestamps and confidence scores (default: True)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print live transcription turns and aligned words to stderr (default: False)",
    )
    args = p.parse_args()

    pairs = destinations(args, suffix="_phowhisper", extension=".json")
    if not pairs:
        progress("empty", "No audio files found matching input criteria.")
        return 0

    import torch

    # Resolve device
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    progress("load", f"Loading PhoWhisper model '{args.model_id}' on {device}...")
    with contextlib.redirect_stdout(sys.stderr):
        model = _load_phowhisper_model(args.model_id, device=device)

    import whisper_timestamped as whisperts

    parameters = {
        "model_id": args.model_id,
        "device": str(device),
        "language": args.language,
        "beam_size": args.beam_size,
        "temperature": args.temperature,
        "vad": bool(args.vad),
        "align_words": bool(args.align_words),
    }

    def process(src: Path, dest: Path) -> None:
        src_identity = identity(src)
        wanted = request(src_identity, "asr", parameters, "phowhisper")

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

        # 1. Run transcription with word timestamps
        transcribe_opts: dict[str, Any] = {
            "language": args.language or "vi",
            "vad": args.vad,
            "detect_disfluencies": True,
            "temperature": args.temperature,
        }
        if args.beam_size > 1:
            transcribe_opts["beam_size"] = args.beam_size

        try:
            asr_res = whisperts.transcribe(model, str(src), **transcribe_opts)
        except Exception as exc:
            err_msg = str(exc).lower()
            if ("silero" in err_msg or "vad" in err_msg or "untrusted" in err_msg) and transcribe_opts.get("vad"):
                logger.warning("VAD error (%s). Retrying transcription with vad=False...", exc)
                transcribe_opts["vad"] = False
                asr_res = whisperts.transcribe(model, str(src), **transcribe_opts)
            else:
                raise

        # 2. Extract audio duration metadata
        audio_info = probe(src)
        audio_duration_s = float(audio_info.get("duration_s", 0.0))

        # 3. Format turns and word timestamps
        raw_segments = asr_res.get("segments", [])
        processed_turns: list[dict[str, Any]] = []

        for seg in raw_segments:
            st = max(0.0, float(seg.get("start", 0.0)))
            et = float(seg.get("end", st + 0.1))
            if audio_duration_s > 0:
                et = min(audio_duration_s, et)
            if et <= st:
                et = st + 0.1

            text = str(seg.get("text", "")).strip()

            words: list[dict[str, Any]] = []
            if args.align_words:
                for w in seg.get("words", []):
                    w_text = str(w.get("text") or w.get("word") or "").strip()
                    if not w_text:
                        continue
                    w_start = float(w.get("start", st))
                    w_end = float(w.get("end", et))
                    confidence = float(w.get("confidence", w.get("probability", 1.0)))
                    words.append({
                        "word": w_text,
                        "start": round(w_start, 3),
                        "end": round(w_end, 3),
                        "probability": round(confidence, 3),
                    })

            processed_turns.append({
                "speaker_id": "0",
                "start_time": round(st, 3),
                "end_time": round(et, 3),
                "duration_s": round(et - st, 3),
                "text": text,
                "words": words,
            })

        full_text = asr_res.get("text", "").strip()
        if not full_text:
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
                    print(f"  └── Words: {words_summary}", file=sys.stderr)
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
