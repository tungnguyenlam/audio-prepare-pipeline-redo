#!/usr/bin/env python3
"""Generate an extended distillation dataset for acoustic boundary verification.

Slices new conversational segments from available Vietnamese audio, creates
realistic acoustic corruptions (onset truncation, coda truncation, head/tail bleed),
and annotates them using Google Gemini 3.8 Flash as the teacher.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import wave
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.diarization.OverlapVerifier import GeminiOverlapVerifier, OVERLAP_PROMPT
from src.utils.AudioClass import Audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extended_distill")

OUTPUT_DIR = Path(".data/distillation/extended")
AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_JSONL = OUTPUT_DIR / "extended_samples.jsonl"
COMBINED_TRAIN_JSONL = Path(".data/distillation/train_extended.jsonl")
COMBINED_VAL_JSONL = Path(".data/distillation/val_extended.jsonl")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def cut_wave(src_path: Path, dst_path: Path, start_s: float, end_s: float) -> Path:
    """Extract slice from WAV file with sample precision."""
    with wave.open(str(src_path), "rb") as wf:
        rate = wf.getframerate()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        start_frame = max(0, int(start_s * rate))
        end_frame = min(wf.getnframes(), int(end_s * rate))
        wf.setpos(start_frame)
        frames = wf.readframes(max(1, end_frame - start_frame))

    with wave.open(str(dst_path), "wb") as out_wf:
        out_wf.setnchannels(nchannels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(rate)
        out_wf.writeframes(frames)
    return dst_path


def append_audio_snippet(base_path: Path, donor_path: Path, dst_path: Path, donor_duration_s: float = 0.3) -> Path:
    """Append a short trailing snippet from another speaker to simulate tail bleed."""
    with wave.open(str(base_path), "rb") as wf1, wave.open(str(donor_path), "rb") as wf2:
        rate = wf1.getframerate()
        frames1 = wf1.readframes(wf1.getnframes())
        donor_frames = int(donor_duration_s * rate)
        frames2 = wf2.readframes(min(wf2.getnframes(), donor_frames))

    with wave.open(str(dst_path), "wb") as out_wf:
        out_wf.setnchannels(wf1.getnchannels())
        out_wf.setsampwidth(wf1.getsampwidth())
        out_wf.setframerate(rate)
        out_wf.writeframes(frames1 + frames2)
    return dst_path


def prepend_audio_snippet(base_path: Path, donor_path: Path, dst_path: Path, donor_duration_s: float = 0.25) -> Path:
    """Prepend a short leading snippet from another speaker to simulate head bleed."""
    with wave.open(str(base_path), "rb") as wf1, wave.open(str(donor_path), "rb") as wf2:
        rate = wf1.getframerate()
        frames1 = wf1.readframes(wf1.getnframes())
        donor_frames = int(donor_duration_s * rate)
        frames2 = wf2.readframes(min(wf2.getnframes(), donor_frames))

    with wave.open(str(dst_path), "wb") as out_wf:
        out_wf.setnchannels(wf1.getnchannels())
        out_wf.setsampwidth(wf1.getsampwidth())
        out_wf.setframerate(rate)
        out_wf.writeframes(frames2 + frames1)
    return dst_path


MIN_AUDIO_DURATION_S = 2.0
MAX_AUDIO_DURATION_S = 15.0


def is_valid_duration(dur: float) -> bool:
    """Hardened rule: All training audio must be between 2.0s and 15.0s."""
    return MIN_AUDIO_DURATION_S <= dur <= MAX_AUDIO_DURATION_S


def generate_candidate_cuts(source_slice: Path) -> list[tuple[Path, str]]:
    """Generate candidate audio slices covering diverse boundary conditions strictly within 2.0s - 15.0s."""
    candidates = []

    # Slices from 180s recording with varied speaker interactions
    segments = [
        ("kv_clean_01", 2.2, 5.2, "clean"),
        ("kv_clean_02", 8.6, 10.4, "clean"),
        ("kv_clean_03", 23.1, 28.0, "clean"),
        ("kv_clean_04", 36.3, 40.6, "clean"),
        ("kv_clean_05", 48.4, 51.2, "clean"),
        ("kv_clean_07", 99.4, 101.7, "clean"),
        ("kv_clean_08", 115.4, 118.5, "clean"),
        ("kv_clean_09", 119.3, 121.2, "clean"),
        ("kv_clean_10", 154.4, 157.6, "clean"),
        # Guest turns (duration >= 2.0s)
        ("guest2_clean_02", 43.8, 46.7, "clean"),
        ("guest2_clean_03", 80.5, 84.5, "clean"),
        ("guest1_clean_05", 158.5, 161.5, "clean"),
        # Longer segments (split or chosen to remain <= 15.0s)
        ("kv_long_01", 11.0, 19.5, "clean"),
        ("kv_long_02", 28.2, 35.8, "clean"),
        ("kv_long_03", 60.0, 74.5, "clean"),
        ("kv_long_04", 130.5, 139.5, "clean"),
    ]

    base_wavs = []
    for name, start, end, kind in segments:
        dur = end - start
        if is_valid_duration(dur):
            dst = AUDIO_DIR / f"{name}.wav"
            cut_wave(source_slice, dst, start, end)
            candidates.append((dst, kind))
            base_wavs.append((dst, dur))

    # Generate synthetic edge degradations strictly adhering to [2.0s, 15.0s]
    logger.info("Generating varied edge-clip degradations within [2.0s, 15.0s]...")
    for idx, (b_wav, dur) in enumerate(base_wavs):
        # 1. Subtle coda cut (80ms)
        coda_sub_dur = dur - 0.08
        if is_valid_duration(coda_sub_dur):
            p_end_subtle = AUDIO_DIR / f"synth_clip_coda_subtle_{idx+1:02d}.wav"
            cut_wave(b_wav, p_end_subtle, 0.0, coda_sub_dur)
            candidates.append((p_end_subtle, "coda_subtle"))

        # 2. Moderate coda cut (180ms)
        coda_mod_dur = dur - 0.18
        if is_valid_duration(coda_mod_dur):
            p_end_mod = AUDIO_DIR / f"synth_clip_coda_mod_{idx+1:02d}.wav"
            cut_wave(b_wav, p_end_mod, 0.0, coda_mod_dur)
            candidates.append((p_end_mod, "coda_moderate"))

        # 3. Subtle onset cut (90ms)
        onset_sub_dur = dur - 0.09
        if is_valid_duration(onset_sub_dur):
            p_start_subtle = AUDIO_DIR / f"synth_clip_onset_subtle_{idx+1:02d}.wav"
            cut_wave(b_wav, p_start_subtle, 0.09, dur)
            candidates.append((p_start_subtle, "onset_subtle"))

        # 4. Severe onset cut (220ms)
        onset_sev_dur = dur - 0.22
        if is_valid_duration(onset_sev_dur):
            p_start_sev = AUDIO_DIR / f"synth_clip_onset_sev_{idx+1:02d}.wav"
            cut_wave(b_wav, p_start_sev, 0.22, dur)
            candidates.append((p_start_sev, "onset_severe"))

    # Generate synthetic tail and head speaker bleed
    logger.info("Generating speaker bleed (cross-talk / secondary speaker)...")
    for idx, (b_wav, dur) in enumerate(base_wavs[:15]):
        donor_candidates = [bw[0] for bw in base_wavs if bw[0] != b_wav]
        donor_wav = random.choice(donor_candidates)

        # Tail intrusion (adds 0.30s)
        tail_dur = dur + 0.30
        if is_valid_duration(tail_dur):
            p_tail = AUDIO_DIR / f"synth_tail_bleed_{idx+1:02d}.wav"
            append_audio_snippet(b_wav, donor_wav, p_tail, donor_duration_s=0.30)
            candidates.append((p_tail, "tail_bleed"))

        # Head intrusion (adds 0.25s)
        head_dur = dur + 0.25
        if is_valid_duration(head_dur):
            p_head = AUDIO_DIR / f"synth_head_bleed_{idx+1:02d}.wav"
            prepend_audio_snippet(b_wav, donor_wav, p_head, donor_duration_s=0.25)
            candidates.append((p_head, "head_bleed"))

    return candidates


def main() -> None:
    source_slice = Path(".data/experiment_khanhvy/khanhvy_180s_slice.wav")
    if not source_slice.is_file():
        logger.error("Source slice not found at %s!", source_slice)
        return

    logger.info("Initializing Gemini 3.8 Flash Teacher...")
    verifier = GeminiOverlapVerifier(
        model="gemini-3.8-flash",
        thinking_level="LOW",
        max_output_tokens=2048,
    )
    if not verifier.check_ready():
        logger.error("Gemini 3.8 Flash is not ready! Check GEMINI_API_KEY.")
        return

    logger.info("Extracting candidate cuts from %s...", source_slice)
    candidates = generate_candidate_cuts(source_slice)
    logger.info("Total candidate audio samples to annotate: %d", len(candidates))

    existing_paths = set()
    annotated_items = []
    if OUTPUT_JSONL.is_file():
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    annotated_items.append(obj)
                    existing_paths.add(obj.get("audio_path"))
        logger.info("Resuming: already have %d annotated samples.", len(annotated_items))

    for idx, (cand_wav, cand_kind) in enumerate(candidates):
        resolved_path = str(cand_wav.resolve())
        if resolved_path in existing_paths:
            continue

        logger.info("[%d/%d] Annotating %s (%s)...", idx + 1, len(candidates), cand_wav.name, cand_kind)
        try:
            aud = Audio.from_file(cand_wav)
            res = verifier.verify(aud)
            target_json = {
                "speaker_purity": res.get("speaker_purity", "unknown"),
                "word_completeness": res.get("word_completeness", "unknown"),
                "boundary_issue": res.get("boundary_issue", "unknown"),
                "failure_codes": res.get("failure_codes", []),
                "reason": res.get("reason", ""),
            }
            item = {
                "audio_path": resolved_path,
                "prompt": OVERLAP_PROMPT,
                "target_json": target_json,
                "decision": res.get("decision", "unknown"),
                "source": cand_kind,
            }
            annotated_items.append(item)
            existing_paths.add(resolved_path)

            with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logger.info("  -> Decision: %s | Codes: %s", res.get("decision"), res.get("failure_codes"))
        except Exception as exc:
            logger.error("  -> Failed to annotate %s: %s", cand_wav.name, exc)

        time.sleep(0.5)

    logger.info("Annotated %d samples successfully!", len(annotated_items))

    orig_train_path = Path(".data/distillation/train.jsonl")
    orig_val_path = Path(".data/distillation/val.jsonl")

    all_train = []
    all_val = []

    if orig_train_path.is_file():
        with open(orig_train_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    all_train.append(json.loads(l))

    if orig_val_path.is_file():
        with open(orig_val_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    all_val.append(json.loads(l))

    random.seed(1337)
    random.shuffle(annotated_items)
    n_val = max(5, int(len(annotated_items) * 0.15))

    new_val = annotated_items[:n_val]
    new_train = annotated_items[n_val:]

    combined_train = all_train + new_train
    combined_val = all_val + new_val

    with open(COMBINED_TRAIN_JSONL, "w", encoding="utf-8") as f:
        for it in combined_train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(COMBINED_VAL_JSONL, "w", encoding="utf-8") as f:
        for it in combined_val:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    logger.info("Extended datasets written:")
    logger.info("  Train: %s (%d samples)", COMBINED_TRAIN_JSONL, len(combined_train))
    logger.info("  Val:   %s (%d samples)", COMBINED_VAL_JSONL, len(combined_val))


if __name__ == "__main__":
    main()
