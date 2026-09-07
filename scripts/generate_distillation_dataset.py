#!/usr/bin/env python3
"""Generate a balanced distillation dataset using Gemini 3.8 Flash as the teacher."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.diarization.OverlapVerifier import GeminiOverlapVerifier, OVERLAP_PROMPT
from src.utils.AudioClass import Audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distill_data")

DISTILL_DIR = Path(".data/distillation")
AUDIO_DIR = DISTILL_DIR / "audio"
DATASET_JSONL = DISTILL_DIR / "dataset.jsonl"
TRAIN_JSONL = DISTILL_DIR / "train.jsonl"
VAL_JSONL = DISTILL_DIR / "val.jsonl"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def cut_wave(src_path: Path, dst_path: Path, start_s: float, end_s: float) -> Path:
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


def append_audio_snippet(base_path: Path, snippet_path: Path, dst_path: Path, snippet_duration_s: float = 0.3) -> Path:
    with wave.open(str(base_path), "rb") as wf1, wave.open(str(snippet_path), "rb") as wf2:
        rate = wf1.getframerate()
        frames1 = wf1.readframes(wf1.getnframes())
        snippet_frames = int(snippet_duration_s * rate)
        frames2 = wf2.readframes(min(wf2.getnframes(), snippet_frames))

    with wave.open(str(dst_path), "wb") as out_wf:
        out_wf.setnchannels(wf1.getnchannels())
        out_wf.setsampwidth(wf1.getsampwidth())
        out_wf.setframerate(rate)
        out_wf.writeframes(frames1 + frames2)
    return dst_path


def main() -> None:
    # 1. Initialize Gemini Teacher
    verifier = GeminiOverlapVerifier(
        model="gemini-3.8-flash",
        thinking_level="LOW",
        max_output_tokens=2048,
    )
    logger.info("Gemini Teacher ready: %s", verifier.check_ready())

    # 2. Collect existing evaluated turns from results.json
    results_json = Path(".data/experiment_khanhvy/results.json")
    with open(results_json, "r", encoding="utf-8") as f:
        existing_records = json.load(f)

    dataset_items = []
    clean_passes = []
    logger.info("Processing %d existing evaluated turns...", len(existing_records))

    for r in existing_records:
        gem = r.get("gemini", {})
        if gem.get("decision") in {"pass", "reject"}:
            wav_src = Path(r["wav_path"])
            if not wav_src.is_file():
                wav_src = Path(".data/experiment_khanhvy/cuts") / f"{r['turn_id']}.wav"
            if wav_src.is_file():
                dst_audio = AUDIO_DIR / f"orig_{r['turn_id']}.wav"
                dst_audio.write_bytes(wav_src.read_bytes())
                
                target_json = {
                    "speaker_purity": gem["speaker_purity"],
                    "word_completeness": gem["word_completeness"],
                    "boundary_issue": gem["boundary_issue"],
                    "failure_codes": gem["failure_codes"],
                    "reason": gem["reason"],
                }
                item = {
                    "audio_path": str(dst_audio.resolve()),
                    "prompt": OVERLAP_PROMPT,
                    "target_json": target_json,
                    "decision": gem["decision"],
                    "source": "khanhvy_orig",
                }
                dataset_items.append(item)
                if gem["decision"] == "pass":
                    clean_passes.append((dst_audio, r["duration_s"]))

    logger.info("Loaded %d original turns (%d clean passes, %d rejects).", len(dataset_items), len(clean_passes), len(dataset_items) - len(clean_passes))

    # 3. Create synthetic perturbations from clean passes
    logger.info("Creating synthetic boundary corruptions from %d clean turns...", len(clean_passes))
    synthetic_cuts = []

    for idx, (clean_wav, dur) in enumerate(clean_passes):
        stem = clean_wav.stem
        # A. Shave end (clipped_word_end)
        if dur > 1.0:
            shave_end_wav = AUDIO_DIR / f"synth_clip_end_{idx+1:03d}.wav"
            cut_wave(clean_wav, shave_end_wav, 0.0, max(0.5, dur - 0.20))
            synthetic_cuts.append((shave_end_wav, "clipped_end"))

        # B. Shave start (clipped_word_start)
        if dur > 1.0:
            shave_start_wav = AUDIO_DIR / f"synth_clip_start_{idx+1:03d}.wav"
            cut_wave(clean_wav, shave_start_wav, 0.20, dur)
            synthetic_cuts.append((shave_start_wav, "clipped_start"))

        # C. Tail intrusion
        other_passes = [p for p in clean_passes if p[0] != clean_wav]
        if other_passes:
            donor_wav, _ = random.choice(other_passes)
            tail_wav = AUDIO_DIR / f"synth_tail_intrusion_{idx+1:03d}.wav"
            append_audio_snippet(clean_wav, donor_wav, tail_wav, snippet_duration_s=0.35)
            synthetic_cuts.append((tail_wav, "tail_intrusion"))

    logger.info("Generated %d synthetic corrupted audio segments.", len(synthetic_cuts))

    # 4. Query Gemini 3.8 Flash to annotate the synthetic cuts
    logger.info("Annotating synthetic cuts with Gemini 3.8 Flash...")
    for s_wav, s_type in synthetic_cuts:
        logger.info("Evaluating %s (%s)...", s_wav.name, s_type)
        try:
            aud = Audio.from_file(s_wav)
            res = verifier.verify(aud)
            target_json = {
                "speaker_purity": res["speaker_purity"],
                "word_completeness": res["word_completeness"],
                "boundary_issue": res["boundary_issue"],
                "failure_codes": res["failure_codes"],
                "reason": res["reason"],
            }
            item = {
                "audio_path": str(s_wav.resolve()),
                "prompt": OVERLAP_PROMPT,
                "target_json": target_json,
                "decision": res["decision"],
                "source": f"synth_{s_type}",
            }
            dataset_items.append(item)
            logger.info("  -> Gemini decision: %s | Codes: %s", res["decision"], res["failure_codes"])
        except Exception as exc:
            logger.error("  -> Gemini annotation failed for %s: %s", s_wav.name, exc)
        time.sleep(0.5)

    # 5. Shuffle and split into train / val
    random.seed(42)
    random.shuffle(dataset_items)

    val_size = max(5, int(len(dataset_items) * 0.15))
    val_items = dataset_items[:val_size]
    train_items = dataset_items[val_size:]

    logger.info("Total dataset size: %d (Train: %d, Val: %d)", len(dataset_items), len(train_items), len(val_items))

    with open(DATASET_JSONL, "w", encoding="utf-8") as f:
        for item in dataset_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_JSONL, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Dataset generated successfully under %s!", DISTILL_DIR)


if __name__ == "__main__":
    main()
