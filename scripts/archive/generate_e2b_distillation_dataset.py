#!/usr/bin/env python3
"""Generate multi-factor acoustic distillation dataset for Gemma 4 E2B fine-tuning.

Pipeline:
1. Slices representative audio from crawled challenging channels (@TRANTHANHTOWN & @KhánhVyOFFICIAL).
2. Runs Mel-Band RoFormer vocal separation on AMD ROCm GPU.
3. Runs DiariZen Large and Pyannote Comm-1 diarization.
4. Clamps all turns strictly into [2.0s, 15.0s] via energy valley splitting.
5. Emits paired raw vs zero-contamination mitigated audio cuts.
6. Evaluates each clip with Gemini 3.8 Flash (MEDIUM reasoning) across:
   - Speaker Purity
   - Word Completeness
   - Audio Quality
7. Saves formatted train_e2b.jsonl and val_e2b.jsonl for Gemma 4 E2B QLoRA fine-tuning.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.diarization.DiariZenWorkerDiarizer import DiariZenWorkerDiarizer
from src.diarization.PyannoteDiarizer import PyannoteDiarizer
from src.diarization.schemas import DiarizationResult
from src.utils.AudioClass import Audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("e2b_distill_builder")

BASE_DIR = Path(".data/distillation_e2b")
SLICES_DIR = BASE_DIR / "slices"
VOCALS_DIR = BASE_DIR / "vocals"
AUDIO_DIR = BASE_DIR / "audio"
DISTILL_DIR = Path(".data/distillation")

for d in [BASE_DIR, SLICES_DIR, VOCALS_DIR, AUDIO_DIR, DISTILL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_RATE = 16000
MIN_DUR = 2.0
MAX_DUR = 15.0

PROMPT_VERIFY = """Listen to the supplied audio directly. Do not transcribe it.
Evaluate three strict acoustic dimensions required for clean Text-to-Speech (TTS) training:

1. SPEAKER PURITY:
   - "pure": Exactly one primary speaker throughout. No background chatter, secondary voices, or intruder laughter.
   - "secondary_speaker": Another speaker's voice is audible (even a short word, whisper, or breath).
   - "overlapping_speech": Multiple speakers speaking or laughing simultaneously.

2. WORD COMPLETENESS (Không lẹm chữ, đủ âm tiết):
   - "complete": All words start and finish on clean acoustic word boundaries with their full vowel decay and coda consonant closure.
   - "clipped_word_start": The initial word has its onset consonant abruptly cut off.
   - "clipped_word_end": The final word is cut off abruptly while vocal fold vibration or tone is still in flight.

3. AUDIO QUALITY:
   - "studio_clean": Clear vocal signal, minimal background artifacts, and no residual music bleed.
   - "music_bleed": Audible residual background music, beats, or synthetic melodies.
   - "noisy_reverberant": Severe room echo, reverb, or excessive environmental noise.
   - "distorted": Clipping distortion, phase artifacts, or muffled frequency response.

DECISION RULE:
- "pass" ONLY if speaker_purity == "pure" AND word_completeness == "complete" AND audio_quality == "studio_clean".
- Otherwise "reject".

Return strict JSON only (no markdown, no other text):
{
  "speaker_purity": "pure" | "secondary_speaker" | "overlapping_speech",
  "word_completeness": "complete" | "clipped_word_start" | "clipped_word_end",
  "audio_quality": "studio_clean" | "music_bleed" | "noisy_reverberant" | "distorted",
  "decision": "pass" | "reject",
  "failure_codes": ["clipped_word_start", "clipped_word_end", "secondary_speaker", "overlapping_speech", "music_bleed", "noisy_reverberant", "distorted"],
  "reason": "Concise English explanation of the acoustic decision."
}"""


# ---------------------------------------------------------------------------
# 1. Slice Extraction
# ---------------------------------------------------------------------------
def extract_slice(src_wav: Path, dst_wav: Path, start_s: float, dur_s: float) -> Path:
    if dst_wav.is_file():
        logger.info("Reusing existing slice: %s", dst_wav)
        return dst_wav

    data, sr = sf.read(str(src_wav))
    s_idx = int(start_s * sr)
    e_idx = min(len(data), int((start_s + dur_s) * sr))
    slice_data = data[s_idx:e_idx]
    sf.write(str(dst_wav), slice_data, sr)
    logger.info("Extracted slice: %s (%.1fs)", dst_wav, len(slice_data) / sr)
    return dst_wav


# ---------------------------------------------------------------------------
# 2. Mel-Band RoFormer Vocal Separation
# ---------------------------------------------------------------------------
def separate_vocals(slice_wav: Path, vocals_wav: Path) -> Path:
    if vocals_wav.is_file():
        logger.info("Reusing existing vocals: %s", vocals_wav)
        return vocals_wav

    unsloth_py = "/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python"
    worker_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, "{REPO_ROOT}")
from src.separation.MelRoFormer import MelRoFormer
from src.utils.AudioClass import Audio

sep = MelRoFormer(
    device="cuda",
    output_dir="{vocals_wav.parent / 'work'}",
    work_dir="{vocals_wav.parent / 'work'}",
    sample_rate=16000,
    channels=1,
)
sep.load()
aud = Audio.from_file("{slice_wav}")
res = sep.separate(aud)
res.save_to("{vocals_wav}")
sep.unload()
print("MEL_ROFORMER_DONE")
"""
    logger.info("Running Mel-Band RoFormer vocal separation for %s...", slice_wav.name)
    res = subprocess.run([unsloth_py, "-c", worker_code], capture_output=True, text=True)
    if res.returncode != 0 or not vocals_wav.is_file():
        logger.error("Vocal separation failed for %s: %s", slice_wav.name, res.stderr)
        raise RuntimeError(f"Vocal separation failed: {res.stderr}")
    logger.info("Separation complete: %s", vocals_wav)
    return vocals_wav


# ---------------------------------------------------------------------------
# 3. Energy Valley Turn Splitting & Boundary Mitigation
# ---------------------------------------------------------------------------
def find_energy_valleys(wav_data: np.ndarray, sr: int, min_gap_s: float = 0.15, threshold_db: float = -32.0) -> list[float]:
    frame_len = int(0.020 * sr)  # 20ms
    hop_len = int(0.010 * sr)    # 10ms
    n_frames = (len(wav_data) - frame_len) // hop_len + 1
    if n_frames <= 0:
        return []

    frames = np.lib.stride_tricks.sliding_window_view(wav_data[:n_frames * hop_len + frame_len], frame_len)[::hop_len]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    db = 20 * np.log10(rms / (np.max(rms) + 1e-9))

    is_silent = db < threshold_db
    valleys = []
    in_silence = False
    start_idx = 0

    for i, s in enumerate(is_silent):
        if s and not in_silence:
            in_silence = True
            start_idx = i
        elif not s and in_silence:
            in_silence = False
            dur_s = (i - start_idx) * hop_len / sr
            if dur_s >= min_gap_s:
                mid_t = (start_idx + i) / 2.0 * hop_len / sr
                valleys.append(mid_t)
    return valleys


def split_turn_intelligently(start_s: float, end_s: float, wav_data: np.ndarray, sr: int) -> list[tuple[float, float]]:
    dur = end_s - start_s
    if dur < MIN_DUR:
        return []
    if dur <= MAX_DUR:
        return [(start_s, end_s)]

    s_idx = int(start_s * sr)
    e_idx = int(end_s * sr)
    turn_audio = wav_data[s_idx:e_idx]

    valleys = find_energy_valleys(turn_audio, sr)
    if not valleys:
        # Fallback: uniform cut
        n_chunks = int(np.ceil(dur / 12.0))
        c_len = dur / n_chunks
        return [(start_s + i * c_len, start_s + (i + 1) * c_len) for i in range(n_chunks)]

    sub_cuts = []
    cur_start = 0.0
    for v in valleys:
        seg_dur = v - cur_start
        if seg_dur >= MIN_DUR:
            if seg_dur <= MAX_DUR:
                sub_cuts.append((round(start_s + cur_start, 3), round(start_s + v, 3)))
                cur_start = v
            else:
                n_sub = int(np.ceil(seg_dur / 10.0))
                s_step = seg_dur / n_sub
                for k in range(n_sub):
                    sub_cuts.append((round(start_s + cur_start + k * s_step, 3),
                                     round(start_s + cur_start + (k + 1) * s_step, 3)))
                cur_start = v

    final_dur = dur - cur_start
    if final_dur >= MIN_DUR:
        if final_dur <= MAX_DUR:
            sub_cuts.append((round(start_s + cur_start, 3), round(end_s, 3)))
        else:
            n_sub = int(np.ceil(final_dur / 10.0))
            s_step = final_dur / n_sub
            for k in range(n_sub):
                sub_cuts.append((round(start_s + cur_start + k * s_step, 3),
                                 round(start_s + cur_start + (k + 1) * s_step, 3)))
    return sub_cuts


def apply_boundary_mitigation(start_s: float, end_s: float, wav_data: np.ndarray, sr: int) -> tuple[float, float]:
    total_dur = len(wav_data) / sr
    lead_in_s = 0.050
    lead_out_s = 0.060
    mit_start = max(0.0, start_s - lead_in_s)
    mit_end = min(total_dur, end_s + lead_out_s)
    return (round(mit_start, 3), round(mit_end, 3))


# ---------------------------------------------------------------------------
# 4. Teacher Evaluation via Gemini 3.8 Flash (MEDIUM reasoning)
# ---------------------------------------------------------------------------
def verify_clip(cand_wav: Path, api_key: str) -> dict[str, Any]:
    dur_s = round(sf.info(str(cand_wav)).duration, 3)
    with open(cand_wav, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
                    {"text": PROMPT_VERIFY},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "MEDIUM"},
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                text = body["candidates"][0]["content"]["parts"][0]["text"]
                res_data = json.loads(text.strip())
                res_data["latency_s"] = round(time.time() - t0, 2)
                res_data["duration_s"] = dur_s
                return res_data
        except Exception as exc:
            if attempt == 2:
                logger.error("Gemini failed for %s: %s", cand_wav.name, exc)
                return {
                    "speaker_purity": "error",
                    "word_completeness": "error",
                    "audio_quality": "error",
                    "decision": "reject",
                    "failure_codes": ["api_error"],
                    "reason": str(exc),
                    "duration_s": dur_s,
                }
            time.sleep(2)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.error("GEMINI_API_KEY not found in environment!")
        return

    # 1. Load Crawled Manifest
    manifest_path = Path(".data/crawled/crawled_manifest.json")
    if not manifest_path.is_file():
        logger.error("No crawled manifest found at %s", manifest_path)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        crawled = json.load(f)

    # Pick 4 rich representative tracks: 2 Tran Thanh, 2 Khanh Vy
    target_ids = ["QBml8L3wS3Q", "j83rzAzRDAI", "Oa-mVxGS4cw", "H0VpjeULCck"]
    selected_tracks = [item for item in crawled if item["id"] in target_ids]

    logger.info("Selected %d tracks for distillation dataset generation:", len(selected_tracks))
    for t in selected_tracks:
        logger.info("  - [%s] %s (%s)", t["channel"], t["title"], t["id"])

    # 2. Extract Slices and Separate Vocals
    slices_and_vocals = []
    for trk in selected_tracks:
        src_wav = Path(trk["path"])
        v_id = trk["id"]
        # Take 180s slice from 45s to 225s
        slice_wav = SLICES_DIR / f"{v_id}_slice_180s.wav"
        extract_slice(src_wav, slice_wav, start_s=45.0, dur_s=180.0)

        vocals_wav = VOCALS_DIR / f"{v_id}_vocals.wav"
        separate_vocals(slice_wav, vocals_wav)
        slices_and_vocals.append((v_id, trk["title"], vocals_wav))

    # 3. Diarization Inference
    diar_cache = BASE_DIR / "diarization_cache.json"
    diar_data = {}
    if diar_cache.is_file():
        try:
            with open(diar_cache, "r", encoding="utf-8") as f:
                diar_data = json.load(f)
        except Exception:
            diar_data = {}

    dz_diarizer = None
    py_diarizer = None

    for v_id, title, voc_wav in slices_and_vocals:
        if v_id not in diar_data:
            diar_data[v_id] = {}

        aud = Audio.from_file(voc_wav)

        # DiariZen
        if "diarizen" not in diar_data[v_id]:
            if dz_diarizer is None:
                dz_diarizer = DiariZenWorkerDiarizer(model_id="BUT-FIT/diarizen-wavlm-large-s80-md-v2", device="cpu")
                dz_diarizer.load()
            logger.info("Running DiariZen on %s...", voc_wav.name)
            res: DiarizationResult = dz_diarizer.diarize(aud)
            diar_data[v_id]["diarizen"] = [
                {"speaker_id": t.speaker_id, "start_s": t.start_s, "end_s": t.end_s}
                for t in res.turns
            ]

        # Pyannote
        if "pyannote" not in diar_data[v_id]:
            if py_diarizer is None:
                py_diarizer = PyannoteDiarizer(model_id="pyannote/speaker-diarization-community-1", device="cpu")
                py_diarizer.load()
            logger.info("Running Pyannote on %s...", voc_wav.name)
            res_py: DiarizationResult = py_diarizer.diarize(aud)
            diar_data[v_id]["pyannote"] = [
                {"speaker_id": t.speaker_id, "start_s": t.start_s, "end_s": t.end_s}
                for t in res_py.turns
            ]

    if dz_diarizer is not None:
        dz_diarizer.unload()
    if py_diarizer is not None:
        py_diarizer.unload()

    with open(diar_cache, "w", encoding="utf-8") as f:
        json.dump(diar_data, f, indent=2, ensure_ascii=False)

    # 4. Generate Paired Cuts (Raw vs Mitigated)
    candidate_items = []
    for v_id, title, voc_wav in slices_and_vocals:
        wav_data, sr = sf.read(str(voc_wav))
        v_diar = diar_data.get(v_id, {})

        for model_name, turns in v_diar.items():
            split_turns = []
            for t in turns:
                sp = split_turn_intelligently(t["start_s"], t["end_s"], wav_data, sr)
                for s_st, s_en in sp:
                    split_turns.append((t["speaker_id"], s_st, s_en))

            # Sample top 6 longest diverse turns per diarizer per video
            split_turns = sorted(split_turns, key=lambda x: x[2] - x[1], reverse=True)[:6]

            for idx, (spk, st, en) in enumerate(split_turns):
                # Raw Cut
                raw_wav = AUDIO_DIR / f"{v_id}_{model_name}_turn_{idx+1:02d}_raw_{st:.2f}-{en:.2f}.wav"
                if not raw_wav.is_file():
                    s_i, e_i = int(st * sr), int(en * sr)
                    sf.write(str(raw_wav), wav_data[s_i:e_i], sr)

                candidate_items.append({
                    "video_id": v_id,
                    "title": title,
                    "model": model_name,
                    "strategy": "raw",
                    "speaker": spk,
                    "start_s": st,
                    "end_s": en,
                    "path": raw_wav,
                    "duration_s": round(en - st, 3),
                })

                # Mitigated Cut
                mit_st, mit_en = apply_boundary_mitigation(st, en, wav_data, sr)
                mit_wav = AUDIO_DIR / f"{v_id}_{model_name}_turn_{idx+1:02d}_mit_{mit_st:.2f}-{mit_en:.2f}.wav"
                if not mit_wav.is_file():
                    ms_i, me_i = int(mit_st * sr), int(mit_en * sr)
                    sf.write(str(mit_wav), wav_data[ms_i:me_i], sr)

                candidate_items.append({
                    "video_id": v_id,
                    "title": title,
                    "model": model_name,
                    "strategy": "mitigated",
                    "speaker": spk,
                    "start_s": mit_st,
                    "end_s": mit_en,
                    "path": mit_wav,
                    "duration_s": round(mit_en - mit_st, 3),
                })

    logger.info("Total paired candidate clips generated: %d", len(candidate_items))

    # 5. Teacher Verification with Gemini 3.8 Flash (MEDIUM)
    logger.info("Evaluating candidates with Gemini 3.8 Flash (MEDIUM reasoning)...")
    audited_entries = []

    def audit_worker(item):
        audit_res = verify_clip(item["path"], gemini_key)
        item["audit"] = audit_res
        logger.info("[%s | %s] %s -> %s (Codes: %s)",
                    item["video_id"], item["strategy"], item["path"].name,
                    audit_res["decision"], audit_res.get("failure_codes", []))
        return item

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(audit_worker, it) for it in candidate_items]
        for f in as_completed(futures):
            audited_entries.append(f.result())

    # Save complete audit manifest
    audit_manifest_path = BASE_DIR / "audit_manifest.json"
    clean_manifest = []
    for it in audited_entries:
        c_it = dict(it)
        c_it["path"] = str(c_it["path"].resolve())
        clean_manifest.append(c_it)

    with open(audit_manifest_path, "w", encoding="utf-8") as f:
        json.dump(clean_manifest, f, indent=2, ensure_ascii=False)
    logger.info("Saved complete audit manifest to %s", audit_manifest_path)

    # 6. Format Training JSONL for Gemma 4 E2B
    logger.info("Building Gemma 4 E2B train/val JSONL...")
    training_samples = []
    for it in audited_entries:
        aud = it["audit"]
        if aud.get("decision") not in ["pass", "reject"]:
            continue

        target_json = {
            "speaker_purity": aud.get("speaker_purity", "pure"),
            "word_completeness": aud.get("word_completeness", "complete"),
            "audio_quality": aud.get("audio_quality", "studio_clean"),
            "decision": aud.get("decision", "reject"),
            "failure_codes": aud.get("failure_codes", []),
            "reason": aud.get("reason", ""),
        }

        sample = {
            "audio_path": str(it["path"].resolve()),
            "prompt": PROMPT_VERIFY,
            "target_json": target_json,
            "decision": aud["decision"],
            "video_id": it["video_id"],
            "strategy": it["strategy"],
            "duration_s": it["duration_s"],
        }
        training_samples.append(sample)

    random.seed(42)
    random.shuffle(training_samples)

    split_idx = int(len(training_samples) * 0.8)
    train_set = training_samples[:split_idx]
    val_set = training_samples[split_idx:]

    train_jsonl = DISTILL_DIR / "train_e2b.jsonl"
    val_jsonl = DISTILL_DIR / "val_e2b.jsonl"

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_jsonl, "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("=== Dataset Generation Complete ===")
    logger.info("Total Labeled Clips: %d", len(training_samples))
    logger.info("Train set: %d -> %s", len(train_set), train_jsonl)
    logger.info("Val set: %d -> %s", len(val_set), val_jsonl)


if __name__ == "__main__":
    main()
