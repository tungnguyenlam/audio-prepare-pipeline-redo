#!/usr/bin/env python3
"""Export experiment results from JSON to CSV for easy inspection in JupyterLab."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_csv")

EXP_DIR = Path(".data/experiment_khanhvy")
JSON_PATH = EXP_DIR / "results.json"
CSV_PATH = EXP_DIR / "results.csv"


def export_csv() -> None:
    if not JSON_PATH.exists():
        logger.error("Source JSON not found: %s", JSON_PATH)
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    fieldnames = [
        "index",
        "turn_id",
        "speaker_id",
        "duration_s",
        "start_s",
        "end_s",
        "gemini_decision",
        "gemma_12b_decision",
        "gemma_e4b_decision",
        "consensus",
        "gemini_reason",
        "gemma_12b_reason",
        "gemma_e4b_reason",
        "gemini_failure_codes",
        "gemma_e4b_failure_codes",
        "audio_path",
    ]

    rows = []
    for r in records:
        gem = r.get("gemini", {})
        e4b = r.get("gemma4_e4b", {})
        g12b = r.get("gemma4_12b", {})

        gem_dec = gem.get("decision", "")
        e4b_dec = e4b.get("decision", "")
        g12b_dec = g12b.get("decision", "")

        if gem_dec and e4b_dec and g12b_dec and gem_dec == e4b_dec == g12b_dec:
            consensus = "All 3 Agree"
        elif gem_dec == e4b_dec:
            consensus = "Gemini + E4B"
        elif gem_dec == g12b_dec:
            consensus = "Gemini + 12B"
        elif e4b_dec == g12b_dec:
            consensus = "E4B + 12B"
        else:
            consensus = "Split / Divergent"

        tid = r.get("turn_id", "")
        rows.append({
            "index": r.get("index"),
            "turn_id": tid,
            "speaker_id": r.get("speaker_id"),
            "duration_s": r.get("duration_s"),
            "start_s": r.get("start_s"),
            "end_s": r.get("end_s"),
            "gemini_decision": gem_dec,
            "gemma_12b_decision": g12b_dec,
            "gemma_e4b_decision": e4b_dec,
            "consensus": consensus,
            "gemini_reason": gem.get("reason", ""),
            "gemma_12b_reason": g12b.get("reason", ""),
            "gemma_e4b_reason": e4b.get("reason", ""),
            "gemini_failure_codes": ";".join(gem.get("failure_codes", [])),
            "gemma_e4b_failure_codes": ";".join(e4b.get("failure_codes", [])),
            "audio_path": f"data/experiment_khanhvy/cuts/{tid}.wav",
        })

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Successfully exported %d records to %s", len(rows), CSV_PATH)


if __name__ == "__main__":
    export_csv()
