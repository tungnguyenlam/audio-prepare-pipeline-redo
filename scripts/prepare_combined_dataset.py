#!/usr/bin/env python3
"""Merge Have A Sip samples with existing dataset and produce stratified train/val splits."""

from __future__ import annotations
import json
import random
from pathlib import Path
from collections import Counter

DISTILL_DIR = Path(".data/distillation")
HAVEASIP_JSONL = DISTILL_DIR / "haveasip" / "haveasip_samples.jsonl"
TRAIN_EXT = DISTILL_DIR / "train_extended.jsonl"
VAL_EXT = DISTILL_DIR / "val_extended.jsonl"

OUT_TRAIN = DISTILL_DIR / "train_combined.jsonl"
OUT_VAL = DISTILL_DIR / "val_combined.jsonl"

def main():
    items = []
    seen = set()
    
    for p in [TRAIN_EXT, VAL_EXT, HAVEASIP_JSONL]:
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                wav_path = Path(item["audio_path"])
                if not wav_path.is_file():
                    continue
                resolved = str(wav_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                items.append(item)

    print(f"Total valid unique items: {len(items)}")
    decisions = Counter(x["decision"] for x in items)
    print("Class distribution:", dict(decisions))

    # Stratified 85/15 split
    random.seed(42)
    pass_items = [x for x in items if x["decision"] == "pass"]
    reject_items = [x for x in items if x["decision"] == "reject"]

    random.shuffle(pass_items)
    random.shuffle(reject_items)

    n_val_pass = max(1, int(len(pass_items) * 0.15))
    n_val_reject = max(1, int(len(reject_items) * 0.15))

    val_items = pass_items[:n_val_pass] + reject_items[:n_val_reject]
    train_items = pass_items[n_val_pass:] + reject_items[n_val_reject:]

    random.shuffle(train_items)
    random.shuffle(val_items)

    print(f"Train items: {len(train_items)} (pass: {sum(1 for x in train_items if x['decision'] == 'pass')}, reject: {sum(1 for x in train_items if x['decision'] == 'reject')})")
    print(f"Val items: {len(val_items)} (pass: {sum(1 for x in val_items if x['decision'] == 'pass')}, reject: {sum(1 for x in val_items if x['decision'] == 'reject')})")

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for it in val_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_TRAIN} and {OUT_VAL}")

if __name__ == "__main__":
    main()
