#!/usr/bin/env python3
"""Compare Gemini 3.8 Flash with LOW vs MEDIUM thinking levels on 31 benchmark turns."""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.diarization.OverlapVerifier import GeminiOverlapVerifier
from src.utils.AudioClass import Audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_reasoning")

RESULTS_JSON = Path(".data/experiment_khanhvy/results.json")
OUTPUT_CSV = Path(".data/experiment_khanhvy/comparison_gemini_low_vs_medium.csv")
OUTPUT_JSON = Path(".data/experiment_khanhvy/comparison_gemini_low_vs_medium.json")


def main() -> None:
    logger.info("Initializing Gemini 3.8 Flash with thinkingLevel='MEDIUM'...")
    verifier_medium = GeminiOverlapVerifier(
        model="gemini-3.8-flash",
        thinking_level="MEDIUM",
        max_output_tokens=4096,
    )
    if not verifier_medium.check_ready():
        logger.error("Gemini 3.8 Flash is not ready! Check GEMINI_API_KEY.")
        return

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Comparing LOW vs MEDIUM reasoning across %d benchmark turns...", len(records))
    comparison_results = []
    flips = 0

    for idx, r in enumerate(records):
        turn_id = r["turn_id"]
        wav_path = Path(r["wav_path"])
        if not wav_path.is_file():
            wav_path = Path(".data/experiment_khanhvy/cuts") / f"{turn_id}.wav"

        low_res = r.get("gemini", {})
        low_dec = low_res.get("decision", "unknown")
        low_codes = low_res.get("failure_codes", [])
        low_reason = low_res.get("reason", "")

        logger.info("[%d/%d] Evaluating %s with MEDIUM reasoning...", idx + 1, len(records), turn_id)
        start_t = time.time()
        try:
            aud = Audio.from_file(wav_path)
            med_res = verifier_medium.verify(aud)
            elapsed = time.time() - start_t
            med_dec = med_res.get("decision", "unknown")
            med_codes = med_res.get("failure_codes", [])
            med_reason = med_res.get("reason", "")
            
            is_flipped = (low_dec != med_dec) and (low_dec in {"pass", "reject"}) and (med_dec in {"pass", "reject"})
            if is_flipped:
                flips += 1
                logger.info("  *** DECISION FLIPPED: LOW=%s -> MEDIUM=%s (%.2fs) ***", low_dec, med_dec, elapsed)
            else:
                logger.info("  -> LOW=%s | MEDIUM=%s | Time: %.2fs", low_dec, med_dec, elapsed)

            row = {
                "turn_id": turn_id,
                "start_s": r.get("start_s", 0.0),
                "end_s": r.get("end_s", 0.0),
                "duration_s": r.get("duration_s", 0.0),
                "flipped": is_flipped,
                "gemini_low_decision": low_dec,
                "gemini_medium_decision": med_dec,
                "gemini_low_codes": ";".join(low_codes),
                "gemini_medium_codes": ";".join(med_codes),
                "gemini_low_reason": low_reason,
                "gemini_medium_reason": med_reason,
                "latency_s": round(elapsed, 2),
            }
            comparison_results.append(row)
        except Exception as exc:
            logger.error("  -> Failed on %s: %s", turn_id, exc)

        time.sleep(0.5)

    logger.info("==========================================")
    logger.info("Comparison Complete! Flipped Decisions: %d/%d", flips, len(comparison_results))
    logger.info("==========================================")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "turn_id",
        "start_s",
        "end_s",
        "duration_s",
        "flipped",
        "gemini_low_decision",
        "gemini_medium_decision",
        "gemini_low_codes",
        "gemini_medium_codes",
        "latency_s",
        "gemini_low_reason",
        "gemini_medium_reason",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_results)

    logger.info("Exported comparison results to %s and %s", OUTPUT_CSV, OUTPUT_JSON)


if __name__ == "__main__":
    main()
