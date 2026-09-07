"""Head-to-head comparison between Gemini 3.8 Flash and Gemma 4 E4B on Khanh Vy audio."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.diarization.OverlapVerifier import (
    DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS,
    OVERLAP_PROMPT,
    _read_audio,
    _normalize_result,
    GeminiOverlapVerifier,
)
from src.diarization.PyannoteDiarizer import PyannoteDiarizer
from src.diarization.zero_contamination import (
    erode_turn_boundaries,
    apply_context_aware_collar,
)
from src.utils.AudioClass import Audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("comparison")

EXP_DIR = Path(".data/experiment_khanhvy")
CUTS_DIR = EXP_DIR / "cuts"
LOG_FILE = EXP_DIR / "run.log"
RESULTS_JSON = EXP_DIR / "results.json"
REPORT_MD = EXP_DIR / "COMPARISON_REPORT.md"


def query_gemma4_e4b(audio: Audio, timeout_s: float = 120.0) -> dict[str, Any]:
    """Query local Gemma 4 E4B on Unsloth Studio."""
    import urllib.request

    audio_bytes, _, _ = _read_audio(audio)
    payload = {
        "model": "unsloth/gemma-4-E4B-it-qat-GGUF",
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

    # Parse and normalize result
    parsed = _normalize_result(content, backend="Unsloth-E4B")
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


def query_gemini_38_flash(verifier: GeminiOverlapVerifier, audio: Audio, max_retries: int = 4) -> dict[str, Any]:
    """Query Google Gemini 3.8 Flash API with retry backoff for 503 and 429 errors."""
    import re
    last_err = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            res = verifier.verify(audio)
            elapsed = time.time() - t0
            return {
                "decision": res["decision"],
                "speaker_purity": res["speaker_purity"],
                "word_completeness": res["word_completeness"],
                "boundary_issue": res["boundary_issue"],
                "failure_codes": res["failure_codes"],
                "reason": res["reason"],
                "latency_s": round(elapsed, 2),
                "usage": res.get("usage"),
            }
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                backoff_s = 2.0 ** (attempt + 1)
                logger.warning("  Gemini 503 spike, retrying in %.1fs (attempt %d/%d)...", backoff_s, attempt + 1, max_retries)
                time.sleep(backoff_s)
                continue
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                match = re.search(r"retry in ([0-9.]+)s", err_str)
                if not match:
                    match = re.search(r'"retryDelay":\s*"([0-9]+)s"', err_str)
                delay = float(match.group(1)) + 1.0 if match else (10.0 * (attempt + 1))
                logger.warning("  Gemini 429 rate limit hit, sleeping %.1fs (attempt %d/%d)...", delay, attempt + 1, max_retries)
                time.sleep(delay)
                continue
            raise
    raise last_err


def cut_turn_audio(src_path: Path, dst_path: Path, start_s: float, end_s: float) -> Audio:
    """Cut slice from audio and save to disk."""
    with wave.open(str(src_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        start_frame = int(start_s * rate)
        end_frame = int(end_s * rate)
        num_frames = max(1, end_frame - start_frame)
        wf.setpos(start_frame)
        frames = wf.readframes(num_frames)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst_path), "wb") as out_wf:
        out_wf.setnchannels(channels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(rate)
        out_wf.writeframes(frames)

    return Audio.from_file(dst_path)


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    CUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Attach file logger
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=== STARTING KHANH VY VERIFIER COMPARISON ===")
    source_audio_path = Path(
        ".data/yt_crawler/downloads/UCwe0_8ud1vpKbWUHnkagorw/"
        "VLOG_Tháp_tùng_chị_gái_nhận_hàm_PGS_Đằng_sau_các_thành_tựu_viral_là_gì_(ft.YouTube_Works_Awards)__H0VpjeULCck.wav"
    )

    if not source_audio_path.exists():
        logger.error("Source audio does not exist at %s", source_audio_path)
        return

    # Slice 180s window for testing
    analysis_duration_s = 180.0
    slice_path = EXP_DIR / "khanhvy_180s_slice.wav"
    logger.info("Extracting %.1fs slice to %s...", analysis_duration_s, slice_path)
    audio_slice = cut_turn_audio(source_audio_path, slice_path, 0.0, analysis_duration_s)
    logger.info("Slice ready: duration=%.2fs, rate=%d", audio_slice.duration_s, audio_slice.sample_rate)

    # 1. Diarization
    hf_token = os.getenv("HF_TOKEN")
    logger.info("Running Pyannote diarization on CPU...")
    t_diar0 = time.time()
    with PyannoteDiarizer(token=hf_token, device="cpu") as diarizer:
        diar_res = diarizer.diarize(audio_slice)
    logger.info("Diarization complete in %.2fs. Found %d raw turns.", time.time() - t_diar0, len(diar_res.turns))

    # 2. Boundary Collar & Handoff Erosion (Base Config)
    logger.info("Applying base boundary collar erosion and handoff guard...")
    collar_guarded, _ = apply_context_aware_collar(
        diar_res.turns,
        collar_s=0.20,
        handoff_risk_s=0.85,
        silence_tail_s=0.03,
        min_duration_s=0.60,
        transition_exclusion_s=0.40,
        audio_duration_s=audio_slice.duration_s,
    )
    # Filter valid turns
    candidate_turns = [t for t in collar_guarded if t.duration_s >= 0.80]
    logger.info("Base config processing produced %d candidate turns.", len(candidate_turns))

    # 3. Initialize Verifiers
    logger.info("Initializing Gemini 3.8 Flash verifier with thinking_level=LOW, max_output_tokens=2048...")
    gemini_verifier = GeminiOverlapVerifier(
        model="gemini-3.8-flash",
        thinking_level="LOW",
        max_output_tokens=2048,
    )
    logger.info("Gemini check: %s", gemini_verifier.check_ready())

    # Load existing Gemma results if available to avoid redundant GPU passes
    existing_gemma_cache = {}
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as jf:
                for rec in json.load(jf):
                    gemma_res = rec.get("gemma4_e4b", {})
                    if gemma_res.get("decision") in {"pass", "reject"}:
                        existing_gemma_cache[rec["turn_id"]] = gemma_res
        except Exception as exc:
            logger.warning("Could not read previous results.json cache: %s", exc)

    # 4. Process Each Turn Head-to-Head
    comparison_records = []
    logger.info("Starting head-to-head evaluation across %d turns...", len(candidate_turns))

    for idx, turn in enumerate(candidate_turns):
        turn_id = f"turn_{idx+1:03d}_{turn.speaker_id}_{turn.start_s:.2f}-{turn.end_s:.2f}"
        turn_wav_path = CUTS_DIR / f"{turn_id}.wav"
        turn_audio = cut_turn_audio(slice_path, turn_wav_path, turn.start_s, turn.end_s)

        logger.info(
            "[%d/%d] Evaluating %s (duration=%.2fs)...",
            idx + 1,
            len(candidate_turns),
            turn_id,
            turn.duration_s,
        )

        # Query Gemini 3.8 Flash
        gemini_result = {}
        try:
            gemini_result = query_gemini_38_flash(gemini_verifier, turn_audio)
            logger.info("  -> Gemini 3.8 Flash: %s in %.2fs | Reason: %s", gemini_result["decision"], gemini_result["latency_s"], gemini_result["reason"])
        except Exception as exc:
            logger.error("  -> Gemini 3.8 Flash FAILED: %s", exc)
            gemini_result = {"error": str(exc), "decision": "error"}

        # Query Gemma 4 E4B (reuse cached pass/reject if available)
        gemma_result = existing_gemma_cache.get(turn_id)
        if gemma_result:
            logger.info("  -> Gemma 4 E4B:      %s (cached) | Reason: %s", gemma_result["decision"], gemma_result["reason"])
        else:
            try:
                gemma_result = query_gemma4_e4b(turn_audio)
                logger.info("  -> Gemma 4 E4B:      %s in %.2fs | Reason: %s", gemma_result["decision"], gemma_result["latency_s"], gemma_result["reason"])
            except Exception as exc:
                logger.error("  -> Gemma 4 E4B FAILED: %s", exc)
                gemma_result = {"error": str(exc), "decision": "error"}

        agree = gemini_result.get("decision") == gemma_result.get("decision")
        record = {
            "index": idx + 1,
            "turn_id": turn_id,
            "speaker_id": turn.speaker_id,
            "start_s": round(turn.start_s, 2),
            "end_s": round(turn.end_s, 2),
            "duration_s": round(turn.duration_s, 2),
            "wav_path": str(turn_wav_path.resolve()),
            "agreement": agree,
            "gemini": gemini_result,
            "gemma4_e4b": gemma_result,
        }
        comparison_records.append(record)

        # Flush incremental results
        with open(RESULTS_JSON, "w", encoding="utf-8") as jf:
            json.dump(comparison_records, jf, indent=2, ensure_ascii=False)

    # 5. Generate Markdown Report
    logger.info("Generating comparison report at %s...", REPORT_MD)
    total = len(comparison_records)
    agreed = sum(1 for r in comparison_records if r["agreement"])
    agreement_rate = (agreed / total * 100) if total else 0.0

    gemini_passed = sum(1 for r in comparison_records if r["gemini"].get("decision") == "pass")
    gemma_passed = sum(1 for r in comparison_records if r["gemma4_e4b"].get("decision") == "pass")

    report_lines = [
        "# Head-to-Head Comparison: Gemini 3.8 Flash vs Gemma 4 E4B",
        "",
        f"- **Source Audio:** `{source_audio_path.name}`",
        f"- **Analyzed Duration:** {analysis_duration_s:.1f}s",
        f"- **Total Candidate Turns:** {total}",
        f"- **Overall Decision Agreement Rate:** **{agreed}/{total} ({agreement_rate:.1f}%)**",
        f"- **Gemini 3.8 Flash Pass Rate:** {gemini_passed}/{total} ({gemini_passed/total*100:.1f}%)",
        f"- **Gemma 4 E4B Pass Rate:** {gemma_passed}/{total} ({gemma_passed/total*100:.1f}%)",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Gemini 3.8 Flash (Cloud API) | Gemma 4 E4B (Local Unsloth / ROCm) |",
        "| :--- | :--- | :--- |",
        f"| Passed Turns | {gemini_passed} | {gemma_passed} |",
        f"| Rejected Turns | {total - gemini_passed} | {total - gemma_passed} |",
        f"| Avg Latency | {sum(r['gemini'].get('latency_s', 0) for r in comparison_records)/max(1, total):.2f}s | {sum(r['gemma4_e4b'].get('latency_s', 0) for r in comparison_records)/max(1, total):.2f}s |",
        "",
        "## Detailed Turn Audit Log",
        "",
        "| Turn | Duration | Audio Cut | Gemini 3.8 Flash | Gemma 4 E4B | Agree? |",
        "| :--- | :--- | :--- | :--- | :--- | :---: |",
    ]

    for r in comparison_records:
        t_id = r["turn_id"]
        dur = f"{r['duration_s']}s"
        wav_link = f"[{t_id}.wav](file://{r['wav_path']})"
        gem_dec = f"**{r['gemini'].get('decision')}** ({r['gemini'].get('reason', '')})"
        gem_dec = gem_dec.replace("\n", " ").replace("|", "\\|")
        gemm_dec = f"**{r['gemma4_e4b'].get('decision')}** ({r['gemma4_e4b'].get('reason', '')})"
        gemm_dec = gemm_dec.replace("\n", " ").replace("|", "\\|")
        agr_icon = "✅" if r["agreement"] else "❌"
        report_lines.append(f"| `{t_id}` | {dur} | {wav_link} | {gem_dec} | {gemm_dec} | {agr_icon} |")

    report_lines.append("")
    report_lines.append("## Disagreement Deep Dive")
    report_lines.append("")
    disagreements = [r for r in comparison_records if not r["agreement"]]
    if not disagreements:
        report_lines.append("None! Both models agreed on 100% of the candidate turns.")
    else:
        for d in disagreements:
            report_lines.append(f"### Turn `{d['turn_id']}` ({d['duration_s']}s)")
            report_lines.append(f"- **Audio File:** [Listen]({d['wav_path']})")
            report_lines.append(f"- **Gemini 3.8 Flash:** `{d['gemini'].get('decision')}` | Purity: `{d['gemini'].get('speaker_purity')}` | Boundary: `{d['gemini'].get('boundary_issue')}` | Codes: `{d['gemini'].get('failure_codes')}`")
            report_lines.append(f"  - *Reason:* {d['gemini'].get('reason')}")
            report_lines.append(f"- **Gemma 4 E4B:** `{d['gemma4_e4b'].get('decision')}` | Purity: `{d['gemma4_e4b'].get('speaker_purity')}` | Boundary: `{d['gemma4_e4b'].get('boundary_issue')}` | Codes: `{d['gemma4_e4b'].get('failure_codes')}`")
            report_lines.append(f"  - *Reason:* {d['gemma4_e4b'].get('reason')}")
            report_lines.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as rf:
        rf.write("\n".join(report_lines) + "\n")

    logger.info("=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
    logger.info("Results written to %s and %s", RESULTS_JSON, REPORT_MD)


if __name__ == "__main__":
    main()
