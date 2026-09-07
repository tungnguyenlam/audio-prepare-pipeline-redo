#!/usr/bin/env python3
"""Package and push the Vietnamese Acoustic Boundary Dataset to Hugging Face with native Audio features.

Encodes audio bytes directly into Parquet so the Hugging Face Dataset Viewer
renders interactive waveform audio players for every sample.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from datasets import Audio, Dataset, DatasetDict
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("push_hf_audio_dataset")

HF_DATASET_REPO = "tungnguyenlam/vietnamese-acoustic-boundary-verifier-data"
TRAIN_JSONL = Path(".data/distillation/train_extended.jsonl")
VAL_JSONL = Path(".data/distillation/val_extended.jsonl")


def load_split_with_audio(jsonl_path: Path) -> Dataset:
    """Load JSONL and bind actual audio files as native datasets.Audio feature."""
    records = {
        "audio": [],
        "decision": [],
        "speaker_purity": [],
        "word_completeness": [],
        "boundary_issue": [],
        "failure_codes": [],
        "reason": [],
        "source": [],
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = Path(item["audio_path"])
            if not audio_path.is_file():
                # Check relative
                audio_path = REPO_ROOT / item["audio_path"]
            if audio_path.is_file():
                records["audio"].append(str(audio_path.resolve()))
                records["decision"].append(item.get("decision", ""))
                tgt = item.get("target_json", {})
                records["speaker_purity"].append(tgt.get("speaker_purity", ""))
                records["word_completeness"].append(tgt.get("word_completeness", ""))
                records["boundary_issue"].append(tgt.get("boundary_issue", ""))
                records["failure_codes"].append(tgt.get("failure_codes", []))
                records["reason"].append(tgt.get("reason", ""))
                records["source"].append(item.get("source", ""))
            else:
                logger.warning("Audio file missing: %s", item.get("audio_path"))

    logger.info("Loaded %d records with audio from %s", len(records["audio"]), jsonl_path.name)
    ds = Dataset.from_dict(records)
    # Cast audio column to native datasets.Audio (preserves bytes in parquet for HF Viewer)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not found in environment!")
        return

    logger.info("Packaging train dataset...")
    train_ds = load_split_with_audio(TRAIN_JSONL)

    logger.info("Packaging validation dataset...")
    val_ds = load_split_with_audio(VAL_JSONL)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
    })

    logger.info("Pushing DatasetDict with embedded audio to https://huggingface.co/datasets/%s...", HF_DATASET_REPO)
    dataset_dict.push_to_hub(
        repo_id=HF_DATASET_REPO,
        token=hf_token,
    )

    logger.info("Successfully pushed dataset with playable audio to https://huggingface.co/datasets/%s", HF_DATASET_REPO)


if __name__ == "__main__":
    main()
