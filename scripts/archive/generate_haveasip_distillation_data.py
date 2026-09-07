#!/usr/bin/env python3
"""Generate acoustic boundary distillation dataset from new Have A Sip video.

Uses Silero VAD to extract speech turns strictly in [2.0s, 15.0s], synthesizes
realistic Vietnamese acoustic boundary corruptions (coda/onset clipping, speaker bleeds),
and annotates them using Google Gemini 3.8 Flash with thinkingLevel='MEDIUM'.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import wave
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.diarization.OverlapVerifier import GeminiOverlapVerifier, OVERLAP_PROMPT
from src.utils.AudioClass import Audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("haveasip_distill")

SOURCE_WAV = Path(".data/new_video/haveasip_khanhvy_6min.wav")
OUTPUT_DIR = Path(".data/distillation/haveasip")
AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_JSONL = OUTPUT_DIR / "haveasip_samples.jsonl"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

MIN_AUDIO_DURATION_S = 2.0
MAX_AUDIO_DURATION_S = 15.0


def is_valid_duration(dur: float) -> bool:
    """Strict hardened rule: audio must be in [2.0s, 15.0s]."""
    return MIN_AUDIO_DURATION_S <= dur <= MAX_AUDIO_DURATION_S


def cut_wave(src_path: Path, dst_path: Path, start_s: float, end_s: float) -> Path:
    """Extract sample-accurate slice from WAV file."""
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


def append_audio_snippet(base_path: Path, donor_path: Path, dst_path: Path, donor_duration_s: float = 0.30) -> Path:
    """Append a short trailing snippet to simulate tail bleed."""
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
    """Prepend a short leading snippet to simulate head bleed."""
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


def extract_vad_speech_turns(wav_path: Path) -> list[tuple[float, float, float]]:
    """Use Silero VAD to detect speech turns strictly in [2.0s, 15.0s]."""
    logger.info("Extracting speech turns from %s using Silero VAD...", wav_path)
    data, sr = sf.read(str(wav_path))
    wav = torch.from_numpy(data).float()

    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils

    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=2000,
        max_speech_duration_s=15,
        min_silence_duration_ms=300,
    )

    valid_turns = []
    for t in timestamps:
        start_s = t["start"] / 16000.0
        end_s = t["end"] / 16000.0
        dur = end_s - start_s
        if is_valid_duration(dur):
            valid_turns.append((start_s, end_s, dur))

    logger.info("Found %d valid speech turns in [%.1fs, %.1fs].", len(valid_turns), MIN_AUDIO_DURATION_S, MAX_AUDIO_DURATION_S)
    return valid_turns


def generate_candidate_dataset(source_wav: Path, turns: list[tuple[float, float, float]]) -> list[tuple[Path, str]]:
    """Produce clean cuts and synthetic edge corruptions adhering to duration boundaries."""
    candidates = []
    base_cuts = []

    # 1. Clean turns
    for idx, (st, en, dur) in enumerate(turns):
        dst = AUDIO_DIR / f"clean_turn_{idx+1:03d}_{st:.2f}-{en:.2f}.wav"
        cut_wave(source_wav, dst, st, en)
        candidates.append((dst, "clean"))
        base_cuts.append((dst, dur))

    # 2. Synthetic corruptions from valid base cuts
    logger.info("Synthesizing acoustic boundary corruptions...")
    for idx, (b_wav, dur) in enumerate(base_cuts):
        # A. Subtle coda cut (80ms)
        coda_sub_dur = dur - 0.08
        if is_valid_duration(coda_sub_dur):
            p_end_sub = AUDIO_DIR / f"synth_clip_coda_subtle_{idx+1:03d}.wav"
            cut_wave(b_wav, p_end_sub, 0.0, coda_sub_dur)
            candidates.append((p_end_sub, "coda_subtle"))

        # B. Moderate coda cut (180ms)
        coda_mod_dur = dur - 0.18
        if is_valid_duration(coda_mod_dur):
            p_end_mod = AUDIO_DIR / f"synth_clip_coda_mod_{idx+1:03d}.wav"
            cut_wave(b_wav, p_end_mod, 0.0, coda_mod_dur)
            candidates.append((p_end_mod, "coda_moderate"))

        # C. Subtle onset cut (90ms)
        onset_sub_dur = dur - 0.09
        if is_valid_duration(onset_sub_dur):
            p_start_sub = AUDIO_DIR / f"synth_clip_onset_subtle_{idx+1:03d}.wav"
            cut_wave(b_wav, p_start_sub, 0.09, dur)
            candidates.append((p_start_sub, "onset_subtle"))

        # D. Severe onset cut (220ms)
        onset_sev_dur = dur - 0.22
        if is_valid_duration(onset_sev_dur):
            p_start_sev = AUDIO_DIR / f"synth_clip_onset_sev_{idx+1:03d}.wav"
            cut_wave(b_wav, p_start_sev, 0.22, dur)
            candidates.append((p_start_sev, "onset_severe"))

    # 3. Cross-talk speaker bleeds
    logger.info("Synthesizing cross-talk speaker bleeds...")
    for idx, (b_wav, dur) in enumerate(base_cuts[:25]):
        donor_candidates = [bw[0] for bw in base_cuts if bw[0] != b_wav]
        if donor_candidates:
            donor_wav = random.choice(donor_candidates)

            # Tail intrusion (adds 0.30s)
            tail_dur = dur + 0.30
            if is_valid_duration(tail_dur):
                p_tail = AUDIO_DIR / f"synth_tail_bleed_{idx+1:03d}.wav"
                append_audio_snippet(b_wav, donor_wav, p_tail, donor_duration_s=0.30)
                candidates.append((p_tail, "tail_bleed"))

            # Head intrusion (adds 0.25s)
            head_dur = dur + 0.25
            if is_valid_duration(head_dur):
                p_head = AUDIO_DIR / f"synth_head_bleed_{idx+1:03d}.wav"
                prepend_audio_snippet(b_wav, donor_wav, p_head, donor_duration_s=0.25)
                candidates.append((p_head, "head_bleed"))

    return candidates


def main() -> None:
    if not SOURCE_WAV.is_file():
        logger.error("Source WAV file not found: %s", SOURCE_WAV)
        return

    logger.info("Initializing Gemini 3.8 Flash with thinkingLevel='MEDIUM'...")
    verifier = GeminiOverlapVerifier(
        model="gemini-3.8-flash",
        thinking_level="MEDIUM",
        max_output_tokens=4096,
    )
    if not verifier.check_ready():
        logger.error("Gemini 3.8 Flash is not ready! Check GEMINI_API_KEY.")
        return

    turns = extract_vad_speech_turns(SOURCE_WAV)
    candidates = generate_candidate_dataset(SOURCE_WAV, turns)
    logger.info("Total candidate clips to annotate with Gemini (MEDIUM): %d", len(candidates))

    existing_paths = set()
    annotated_items = []
    if OUTPUT_JSONL.is_file():
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    annotated_items.append(item)
                    existing_paths.add(item.get("audio_path"))
        logger.info("Resuming: already have %d annotated samples.", len(annotated_items))

    pending_candidates = []
    for idx, (cand_wav, cand_kind) in enumerate(candidates):
        resolved = str(cand_wav.resolve())
        if resolved not in existing_paths:
            pending_candidates.append((idx + 1, cand_wav, cand_kind, resolved))

    logger.info("Total remaining clips to annotate: %d (concurrency=6)", len(pending_candidates))

    write_lock = threading.Lock()

    def process_candidate(task_tuple):
        idx_num, cand_wav, cand_kind, resolved = task_tuple
        start_t = time.time()
        try:
            aud = Audio.from_file(cand_wav)
            res = verifier.verify(aud)
            elapsed = time.time() - start_t

            target_json = {
                "speaker_purity": res.get("speaker_purity", "unknown"),
                "word_completeness": res.get("word_completeness", "unknown"),
                "boundary_issue": res.get("boundary_issue", "unknown"),
                "failure_codes": res.get("failure_codes", []),
                "reason": res.get("reason", ""),
            }
            item = {
                "audio_path": resolved,
                "prompt": OVERLAP_PROMPT,
                "target_json": target_json,
                "decision": res.get("decision", "unknown"),
                "source": cand_kind,
                "duration_s": round(aud.duration_s, 3),
                "teacher_latency_s": round(elapsed, 2),
            }
            with write_lock:
                annotated_items.append(item)
                existing_paths.add(resolved)
                with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logger.info("[%d/%d] Done %s (%s) -> %s | Codes: %s | Elapsed: %.2fs",
                        len(annotated_items), len(candidates), cand_wav.name, cand_kind,
                        res.get("decision"), res.get("failure_codes"), elapsed)
            return item
        except Exception as exc:
            logger.error("Failed to annotate %s: %s", cand_wav.name, exc)
            return None

    if pending_candidates:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(process_candidate, t) for t in pending_candidates]
            for f in as_completed(futures):
                _ = f.result()

    logger.info("Annotation complete! Total annotated samples: %d", len(annotated_items))


if __name__ == "__main__":
    main()
