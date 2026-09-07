#!/usr/bin/env python3
"""Evaluate Gemma 4 12B UD-Q6_K_XL across all 31 Khanh Vy turns and update results."""

from __future__ import annotations

import base64
import csv
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diarization.OverlapVerifier import OVERLAP_PROMPT, _normalize_result, _read_audio
from src.utils.AudioClass import Audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(".data/experiment_khanhvy/gemma_12b_q6.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("12b_q6")

EXP_DIR = Path(".data/experiment_khanhvy")
RESULTS_JSON = EXP_DIR / "results.json"
RESULTS_CSV = EXP_DIR / "results.csv"
REPORT_MD = EXP_DIR / "COMPARISON_REPORT.md"


def query_12b_q6(audio_path: Path, timeout_s: float = 120.0) -> dict:
    audio = Audio.from_file(audio_path)
    audio_bytes, _, _ = _read_audio(audio)
    payload = {
        "model": "unsloth/gemma-4-12b-it-GGUF",
        "messages": [{"role": "user", "content": OVERLAP_PROMPT}],
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        "http://localhost:8889/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer sk-unsloth-d5e3632a095b7e04619fd36a650a9162",
            "Content-Type": "application/json",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        res = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    msg = res["choices"][0]["message"]
    content = msg.get("content", "").strip()
    reasoning = msg.get("reasoning_content", "")

    parsed = _normalize_result(content, backend="Unsloth-12B-Q6")
    return {
        "raw_content": content,
        "reasoning": reasoning,
        "decision": parsed["decision"],
        "speaker_purity": parsed["speaker_purity"],
        "word_completeness": parsed["word_completeness"],
        "boundary_issue": parsed["boundary_issue"],
        "failure_codes": parsed["failure_codes"],
        "reason": parsed["reason"],
        "latency_s": round(elapsed, 2),
    }


def update_csv_and_report(records: list[dict]) -> None:
    # Update CSV with comprehensive comparison columns
    fieldnames = [
        "index",
        "turn_id",
        "speaker_id",
        "duration_s",
        "gemini_decision",
        "gemma_12b_q6_decision",
        "gemma_12b_q4_decision",
        "gemma_e4b_q8_decision",
        "gemma_e4b_q4_decision",
        "gemini_reason",
        "gemma_12b_q6_reason",
        "gemma_12b_q4_reason",
        "gemma_e4b_q8_reason",
        "gemma_e4b_q4_reason",
        "audio_path",
    ]
    rows = []
    for r in records:
        tid = r.get("turn_id", "")
        rows.append({
            "index": r.get("index"),
            "turn_id": tid,
            "speaker_id": r.get("speaker_id"),
            "duration_s": r.get("duration_s"),
            "gemini_decision": r.get("gemini", {}).get("decision", ""),
            "gemma_12b_q6_decision": r.get("gemma4_12b_q6", {}).get("decision", ""),
            "gemma_12b_q4_decision": r.get("gemma4_12b", {}).get("decision", ""),
            "gemma_e4b_q8_decision": r.get("gemma4_e4b_q8", {}).get("decision", ""),
            "gemma_e4b_q4_decision": r.get("gemma4_e4b", {}).get("decision", ""),
            "gemini_reason": r.get("gemini", {}).get("reason", ""),
            "gemma_12b_q6_reason": r.get("gemma4_12b_q6", {}).get("reason", ""),
            "gemma_12b_q4_reason": r.get("gemma4_12b", {}).get("reason", ""),
            "gemma_e4b_q8_reason": r.get("gemma4_e4b_q8", {}).get("reason", ""),
            "gemma_e4b_q4_reason": r.get("gemma4_e4b", {}).get("reason", ""),
            "audio_path": f"data/experiment_khanhvy/cuts/{tid}.wav",
        })
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not RESULTS_JSON.exists():
        logger.error("No results.json found")
        return

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Starting Gemma 4 12B UD-Q6_K_XL evaluation across %d turns...", len(records))

    for idx, r in enumerate(records):
        turn_id = r["turn_id"]
        wav_path = Path(r["wav_path"])
        if not wav_path.is_file():
            wav_path = EXP_DIR / "cuts" / f"{turn_id}.wav"

        logger.info("[%d/%d] Evaluating %s on Gemma 4 12B UD-Q6_K_XL...", idx + 1, len(records), turn_id)

        try:
            res_q6 = query_12b_q6(wav_path)
            logger.info("  -> 12B Q6: %s in %.2fs | Reason: %s", res_q6["decision"], res_q6["latency_s"], res_q6["reason"])
            r["gemma4_12b_q6"] = res_q6
        except Exception as exc:
            logger.error("  -> 12B Q6 FAILED: %s", exc)
            r["gemma4_12b_q6"] = {"error": str(exc), "decision": "error"}

        with open(RESULTS_JSON, "w", encoding="utf-8") as jf:
            json.dump(records, jf, indent=2, ensure_ascii=False)

    update_csv_and_report(records)
    logger.info("=== EVALUATION OF GEMMA 4 12B UD-Q6_K_XL COMPLETE ===")


if __name__ == "__main__":
    main()
