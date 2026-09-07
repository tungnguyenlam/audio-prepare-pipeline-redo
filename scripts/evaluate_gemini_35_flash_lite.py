#!/usr/bin/env python3
"""Evaluate Gemini 3.5 Flash-Lite on the 31 Khanh Vy benchmark turns."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.diarization.OverlapVerifier import OVERLAP_PROMPT
from src.utils.AudioClass import Audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_gemini_35_lite")

EXP_DIR = Path(".data/experiment_khanhvy")
RESULTS_JSON = EXP_DIR / "results.json"
OUTPUT_JSON = EXP_DIR / "gemini_35_flash_lite_eval.json"
OUTPUT_MD = EXP_DIR / "GEMINI_35_FLASH_LITE_REPORT.md"
MODEL_NAME = "gemini-3.5-flash-lite"


def call_gemini_35_lite(cand_wav: Path, api_key: str) -> dict[str, Any]:
    with open(cand_wav, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
                    {"text": OVERLAP_PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                elapsed = round(time.time() - t0, 2)
                cand = body["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(cand.strip())
                
                # Derive decision
                is_pure = data.get("speaker_purity") == "pure"
                is_complete = data.get("word_completeness") == "complete"
                no_boundary = data.get("boundary_issue") == "none"
                decision = "pass" if (is_pure and is_complete and no_boundary) else "reject"
                data["decision"] = decision
                data["latency_s"] = elapsed
                return data
        except Exception as exc:
            if attempt == 2:
                logger.error("Gemini 3.5 Flash-Lite failed for %s: %s", cand_wav.name, exc)
                return {
                    "speaker_purity": "error",
                    "word_completeness": "error",
                    "boundary_issue": "uncertain",
                    "decision": "reject",
                    "failure_codes": ["api_error"],
                    "reason": str(exc),
                    "latency_s": round(time.time() - t0, 2),
                }
            time.sleep(2)


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is not set!")
        return

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Loaded %d benchmark turns from %s", len(records), RESULTS_JSON)
    logger.info("Evaluating with %s...", MODEL_NAME)

    eval_items = []
    for r in records:
        turn_id = r["turn_id"]
        wav_path = Path(r["wav_path"])
        if not wav_path.is_file():
            wav_path = EXP_DIR / "cuts" / f"{turn_id}.wav"
        eval_items.append({
            "index": r["index"],
            "turn_id": turn_id,
            "duration_s": r.get("duration_s", 0.0),
            "wav_path": wav_path,
            "gemini_38_decision": r.get("gemini", {}).get("decision", "unknown"),
            "gemma4_e4b_decision": r.get("gemma4_e4b", {}).get("decision", "unknown"),
        })

    results = []
    def worker(item):
        res = call_gemini_35_lite(item["wav_path"], api_key)
        item["eval"] = res
        logger.info("[%d/31] %s (%.2fs) -> %s (Purity: %s, Completeness: %s, Boundary: %s) [%.2fs]",
                    item["index"], item["turn_id"], item["duration_s"],
                    res["decision"], res.get("speaker_purity"), res.get("word_completeness"),
                    res.get("boundary_issue"), res["latency_s"])
        return item

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(worker, it) for it in eval_items]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x["index"])

    # Statistics
    total = len(results)
    n_pass = sum(1 for x in results if x["eval"]["decision"] == "pass")
    n_rej = total - n_pass
    avg_latency = sum(x["eval"]["latency_s"] for x in results) / max(1, total)

    g38_agreed = sum(1 for x in results if x["eval"]["decision"] == x["gemini_38_decision"])
    e4b_agreed = sum(1 for x in results if x["eval"]["decision"] == x["gemma4_e4b_decision"])

    logger.info("==========================================")
    logger.info("Gemini 3.5 Flash-Lite Benchmark Summary:")
    logger.info("  Total Evaluated: %d", total)
    logger.info("  Pass: %d (%.1f%%) | Reject: %d (%.1f%%)", n_pass, (n_pass/total)*100, n_rej, (n_rej/total)*100)
    logger.info("  Agreement with Gemini 3.8 Flash: %d/%d (%.1f%%)", g38_agreed, total, (g38_agreed/total)*100)
    logger.info("  Agreement with Gemma 4 E4B: %d/%d (%.1f%%)", e4b_agreed, total, (e4b_agreed/total)*100)
    logger.info("  Average Latency: %.2fs", avg_latency)
    logger.info("==========================================")

    # Save JSON
    clean_results = []
    for r in results:
        cr = dict(r)
        cr["wav_path"] = str(cr["wav_path"])
        clean_results.append(cr)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    # Generate Markdown Report
    lines = []
    lines.append(f"# Benchmark: Gemini 3.5 Flash-Lite on 31 Khanh Vy Benchmark Turns\n")
    lines.append(f"- **Evaluator:** `{MODEL_NAME}`")
    lines.append(f"- **Total Benchmark Turns:** {total}")
    lines.append(f"- **Pass Count:** {n_pass} ({(n_pass/total)*100:.1f}%)")
    lines.append(f"- **Reject Count:** {n_rej} ({(n_rej/total)*100:.1f}%)")
    lines.append(f"- **Agreement with Gemini 3.8 Flash:** {g38_agreed}/{total} ({(g38_agreed/total)*100:.1f}%)")
    lines.append(f"- **Agreement with Gemma 4 E4B:** {e4b_agreed}/{total} ({(e4b_agreed/total)*100:.1f}%)")
    lines.append(f"- **Average Latency:** {avg_latency:.2f}s per clip\n")

    lines.append("## Comparison Table\n")
    lines.append("| Index | Turn ID | Dur (s) | 3.5 Flash-Lite | 3.8 Flash | Gemma 4 E4B | 3.5 Latency | Purity | Completeness | Boundary Issue | Reason |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for r in results:
        ev = r["eval"]
        lines.append(
            f"| {r['index']} | `{r['turn_id']}` | {r['duration_s']}s | "
            f"**{ev['decision']}** | `{r['gemini_38_decision']}` | `{r['gemma4_e4b_decision']}` | "
            f"{ev['latency_s']}s | `{ev.get('speaker_purity')}` | `{ev.get('word_completeness')}` | "
            f"`{ev.get('boundary_issue')}` | {ev.get('reason', '')} |"
        )

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Report written to %s", OUTPUT_MD)


if __name__ == "__main__":
    main()
