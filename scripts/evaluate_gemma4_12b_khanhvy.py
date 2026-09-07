#!/usr/bin/env python3
"""Evaluate Gemma 4 12B against Khanh Vy candidate turns and generate a 3-way comparison."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

from src.diarization.OverlapVerifier import OVERLAP_PROMPT, _normalize_result, _read_audio
from src.utils.AudioClass import Audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(".data/experiment_khanhvy/gemma12b.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("gemma12b")

EXP_DIR = Path(".data/experiment_khanhvy")
RESULTS_JSON = EXP_DIR / "results.json"
REPORT_MD = EXP_DIR / "COMPARISON_REPORT.md"


def query_gemma4_12b(audio_path: Path, timeout_s: float = 120.0) -> dict[str, Any]:
    """Query local Gemma 4 12B on Unsloth Studio."""
    audio = Audio.from_file(audio_path)
    audio_bytes, _, _ = _read_audio(audio)
    payload = {
        "model": "unsloth/gemma-4-12B-it-qat-GGUF",
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

    parsed = _normalize_result(content, backend="Unsloth-12B")
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


def generate_3way_report(records: list[dict[str, Any]]) -> str:
    """Generate comprehensive 3-way Markdown comparison report."""
    total = len(records)
    gemini_passed = sum(1 for r in records if r.get("gemini", {}).get("decision") == "pass")
    e4b_passed = sum(1 for r in records if r.get("gemma4_e4b", {}).get("decision") == "pass")
    g12b_passed = sum(1 for r in records if r.get("gemma4_12b", {}).get("decision") == "pass")

    gemini_lats = [r["gemini"]["latency_s"] for r in records if "latency_s" in r.get("gemini", {})]
    e4b_lats = [r["gemma4_e4b"]["latency_s"] for r in records if "latency_s" in r.get("gemma4_e4b", {})]
    g12b_lats = [r["gemma4_12b"]["latency_s"] for r in records if "latency_s" in r.get("gemma4_12b", {})]

    avg_gemini_lat = sum(gemini_lats) / len(gemini_lats) if gemini_lats else 0.0
    avg_e4b_lat = sum(e4b_lats) / len(e4b_lats) if e4b_lats else 0.0
    avg_g12b_lat = sum(g12b_lats) / len(g12b_lats) if g12b_lats else 0.0

    agree_gemini_12b = sum(1 for r in records if r.get("gemini", {}).get("decision") == r.get("gemma4_12b", {}).get("decision"))
    agree_e4b_12b = sum(1 for r in records if r.get("gemma4_e4b", {}).get("decision") == r.get("gemma4_12b", {}).get("decision"))
    agree_gemini_e4b = sum(1 for r in records if r.get("gemini", {}).get("decision") == r.get("gemma4_e4b", {}).get("decision"))
    agree_all_3 = sum(
        1 for r in records
        if r.get("gemini", {}).get("decision") == r.get("gemma4_e4b", {}).get("decision") == r.get("gemma4_12b", {}).get("decision")
    )

    lines = [
        "# 3-Way Acoustic Verification Benchmark: Gemini 3.8 Flash vs Gemma 4 12B vs Gemma 4 E4B",
        "",
        "- **Source Audio:** `VLOG_Tháp_tùng_chị_gái_nhận_hàm_PGS_Đằng_sau_các_thành_tựu_viral_là_gì_(ft.YouTube_Works_Awards)__H0VpjeULCck.wav`",
        f"- **Total Candidate Turns:** {total}",
        f"- **Gemini 3.8 Flash vs Gemma 4 12B Agreement:** **{agree_gemini_12b}/{total} ({agree_gemini_12b/total*100:.1f}%)**",
        f"- **Gemma 4 E4B vs Gemma 4 12B Agreement:** **{agree_e4b_12b}/{total} ({agree_e4b_12b/total*100:.1f}%)**",
        f"- **Gemini 3.8 Flash vs Gemma 4 E4B Agreement:** **{agree_gemini_e4b}/{total} ({agree_gemini_e4b/total*100:.1f}%)**",
        f"- **Full 3-Way Consensus (All Models Agree):** **{agree_all_3}/{total} ({agree_all_3/total*100:.1f}%)**",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Google Gemini 3.8 Flash (`thinkingLevel=LOW`) | Gemma 4 12B (Local Unsloth / ROCm) | Gemma 4 E4B (Local Unsloth / ROCm) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Passed Turns | {gemini_passed} ({gemini_passed/total*100:.1f}%) | {g12b_passed} ({g12b_passed/total*100:.1f}%) | {e4b_passed} ({e4b_passed/total*100:.1f}%) |",
        f"| Rejected Turns | {total - gemini_passed} ({(total-gemini_passed)/total*100:.1f}%) | {total - g12b_passed} ({(total-g12b_passed)/total*100:.1f}%) | {total - e4b_passed} ({(total-e4b_passed)/total*100:.1f}%) |",
        f"| Avg Latency | {avg_gemini_lat:.2f}s | {avg_g12b_lat:.2f}s | {avg_e4b_lat:.2f}s |",
        f"| Hardware | Google Cloud API | AMD Radeon (12.87 GB VRAM) | AMD Radeon (9.47 GB VRAM) |",
        "",
        "## Turn-by-Turn Audit Table",
        "",
        "| Turn | Dur | Audio Cut | Gemini 3.8 Flash | Gemma 4 12B | Gemma 4 E4B | Consensus |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :---: |",
    ]

    for r in records:
        turn_id = r["turn_id"]
        dur = f"{r['duration_s']}s"
        wav_rel = f"cuts/{turn_id}.wav"
        gem = r.get("gemini", {})
        e4b = r.get("gemma4_e4b", {})
        g12b = r.get("gemma4_12b", {})

        gem_dec = gem.get("decision", "n/a")
        gem_rsn = gem.get("reason", "")
        e4b_dec = e4b.get("decision", "n/a")
        e4b_rsn = e4b.get("reason", "")
        g12b_dec = g12b.get("decision", "n/a")
        g12b_rsn = g12b.get("reason", "")

        consensus = "✅ All 3" if (gem_dec == e4b_dec == g12b_dec) else ("🤝 2-1" if (gem_dec == g12b_dec or e4b_dec == g12b_dec or gem_dec == e4b_dec) else "❌ Divergent")

        gem_cell = f"**{gem_dec}**" + (f" ({gem_rsn})" if gem_rsn else "")
        g12b_cell = f"**{g12b_dec}**" + (f" ({g12b_rsn})" if g12b_rsn else "")
        e4b_cell = f"**{e4b_dec}**" + (f" ({e4b_rsn})" if e4b_rsn else "")

        lines.append(f"| `{turn_id}` | {dur} | [{turn_id}.wav]({wav_rel}) | {gem_cell} | {g12b_cell} | {e4b_cell} | {consensus} |")

    lines.extend([
        "",
        "## Disagreement Deep Dive",
        "",
    ])

    for r in records:
        gem_dec = r.get("gemini", {}).get("decision")
        g12b_dec = r.get("gemma4_12b", {}).get("decision")
        e4b_dec = r.get("gemma4_e4b", {}).get("decision")
        if not (gem_dec == g12b_dec == e4b_dec):
            lines.extend([
                f"### Turn `{r['turn_id']}` ({r['duration_s']}s)",
                f"- **Audio Cut:** [Listen](data/experiment_khanhvy/cuts/{r['turn_id']}.wav)",
                f"- **Gemini 3.8 Flash:** `{gem_dec}` — {r.get('gemini', {}).get('reason')}",
                f"- **Gemma 4 12B:** `{g12b_dec}` — {r.get('gemma4_12b', {}).get('reason')}",
                f"- **Gemma 4 E4B:** `{e4b_dec}` — {r.get('gemma4_e4b', {}).get('reason')}",
                "",
            ])

    return "\n".join(lines)


def main() -> None:
    if not RESULTS_JSON.exists():
        logger.error("No results.json found at %s", RESULTS_JSON)
        return

    with open(RESULTS_JSON, "r", encoding="utf-8") as jf:
        records = json.load(jf)

    logger.info("Loaded %d turn records from results.json. Starting Gemma 4 12B evaluation...", len(records))

    for idx, r in enumerate(records):
        turn_id = r["turn_id"]
        wav_path = Path(r["wav_path"])
        if not wav_path.is_file():
            wav_path = EXP_DIR / "cuts" / f"{turn_id}.wav"

        logger.info("[%d/%d] Evaluating %s on Gemma 4 12B (dur=%.2fs)...", idx + 1, len(records), turn_id, r["duration_s"])

        if r.get("gemma4_12b") and r["gemma4_12b"].get("decision") in {"pass", "reject"}:
            logger.info("  -> Gemma 4 12B: %s (already evaluated)", r["gemma4_12b"]["decision"])
            continue

        try:
            res_12b = query_gemma4_12b(wav_path)
            logger.info("  -> Gemma 4 12B: %s in %.2fs | Reason: %s", res_12b["decision"], res_12b["latency_s"], res_12b["reason"])
            r["gemma4_12b"] = res_12b
        except Exception as exc:
            logger.error("  -> Gemma 4 12B FAILED: %s", exc)
            r["gemma4_12b"] = {"error": str(exc), "decision": "error"}

        # Incremental save
        with open(RESULTS_JSON, "w", encoding="utf-8") as jf:
            json.dump(records, jf, indent=2, ensure_ascii=False)

    logger.info("Generating updated 3-way Markdown report at %s...", REPORT_MD)
    report_content = generate_3way_report(records)
    with open(REPORT_MD, "w", encoding="utf-8") as mf:
        mf.write(report_content)

    logger.info("=== GEMMA 4 12B EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()
