#!/usr/bin/env python3
"""Build a balanced training and validation dataset for Gemma 4 E2B distillation.

Combines:
1. Challenging real-world samples from crawled channels (Trấn Thành, Khánh Vy)
   with Mel-Band RoFormer separation and DiariZen/Pyannote segmentation.
2. Verified studio clean pass cuts (HaveASip podcast) mapped to the 3-factor schema.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("balance_e2b_data")

DISTILL_DIR = Path(".data/distillation")
TRAIN_E2B = DISTILL_DIR / "train_e2b.jsonl"
VAL_E2B = DISTILL_DIR / "val_e2b.jsonl"
TRAIN_COMBINED = DISTILL_DIR / "train_combined.jsonl"
VAL_COMBINED = DISTILL_DIR / "val_combined.jsonl"

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


def load_jsonl(path: Path) -> list[dict]:
    items = []
    if not path.is_file():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    # 1. Existing E2B data (challenging crawled vlog cuts)
    train_e2b = load_jsonl(TRAIN_E2B)
    val_e2b = load_jsonl(VAL_E2B)
    logger.info("Loaded crawled E2B data: %d train, %d val", len(train_e2b), len(val_e2b))

    # 2. Extract clean studio passes from earlier HaveASip & extended cuts
    clean_passes = []
    for src_file in [TRAIN_COMBINED, VAL_COMBINED]:
        for it in load_jsonl(src_file):
            if it.get("decision") == "pass" and Path(it["audio_path"]).is_file():
                # Map to new 3-factor schema
                mapped_item = {
                    "audio_path": it["audio_path"],
                    "prompt": PROMPT_VERIFY,
                    "target_json": {
                        "speaker_purity": "pure",
                        "word_completeness": "complete",
                        "audio_quality": "studio_clean",
                        "decision": "pass",
                        "failure_codes": [],
                        "reason": it.get("target_json", {}).get("reason", "Single clean speaker with natural acoustic boundaries and dry studio acoustics."),
                    },
                    "decision": "pass",
                    "source": "studio_clean",
                    "duration_s": it.get("duration_s", 0.0),
                }
                clean_passes.append(mapped_item)

    logger.info("Extracted %d clean studio pass items", len(clean_passes))
    random.seed(42)
    random.shuffle(clean_passes)

    # Split clean passes 80/20
    split_idx = int(len(clean_passes) * 0.8)
    clean_train = clean_passes[:split_idx]
    clean_val = clean_passes[split_idx:]

    # Merge
    final_train = train_e2b + clean_train
    final_val = val_e2b + clean_val

    random.shuffle(final_train)
    random.shuffle(final_val)

    # Write back to TRAIN_E2B and VAL_E2B
    with open(TRAIN_E2B, "w", encoding="utf-8") as f:
        for it in final_train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(VAL_E2B, "w", encoding="utf-8") as f:
        for it in final_val:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Statistics
    t_pass = sum(1 for x in final_train if x["decision"] == "pass")
    t_rej = len(final_train) - t_pass
    v_pass = sum(1 for x in final_val if x["decision"] == "pass")
    v_rej = len(final_val) - v_pass

    logger.info("==========================================")
    logger.info("Balanced Gemma 4 E2B Dataset Ready:")
    logger.info("  Train Set: %d total (%d pass [%.1f%%], %d reject [%.1f%%])",
                len(final_train), t_pass, (t_pass/len(final_train))*100, t_rej, (t_rej/len(final_train))*100)
    logger.info("  Val Set:   %d total (%d pass [%.1f%%], %d reject [%.1f%%])",
                len(final_val), v_pass, (v_pass/len(final_val))*100, v_rej, (v_rej/len(final_val))*100)
    logger.info("==========================================")


if __name__ == "__main__":
    main()
