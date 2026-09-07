#!/usr/bin/env python3
"""Unified CLI for evaluating direct-audio speech verifiers (Gemini, Gemma Unsloth, HF LoRA).

Supports running evaluations on JSONL datasets, directory of WAVs, or experiment turn files,
with metrics computation, agreement comparisons, markdown report generation, and CSV export.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from src.diarization.verifier_training import resolve_audio_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_verifier")

DEFAULT_ACOUSTIC_PROMPT = """Listen to the supplied audio directly. Do not transcribe it.
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


def query_gemini(
    audio_path: Path,
    model: str,
    api_key: str,
    prompt: str,
    reasoning_effort: str = "none",
) -> dict[str, Any]:
    """Query Gemini REST endpoint directly with audio payload."""
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        },
    }
    if reasoning_effort and reasoning_effort.lower() != "none":
        payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": reasoning_effort.upper()}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    latency = round(time.time() - t0, 3)

    candidates = res.get("candidates", [])
    if not candidates:
        raise RuntimeError("No candidates returned from Gemini API")

    raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    parsed = json.loads(raw_text)
    parsed["_latency_s"] = latency
    return parsed


def query_hf_local(
    audio_path: Path,
    model: Any,
    processor: Any,
    device: str,
    prompt: str,
) -> dict[str, Any]:
    """Evaluate audio with local Hugging Face model + processor."""
    import librosa
    import torch

    t0 = time.time()
    audio_data, _ = librosa.load(str(audio_path), sr=16000)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_data},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, audio=audio_data, return_tensors="pt", sampling_rate=16000)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    use_cuda = device.startswith("cuda")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    )

    with torch.no_grad(), autocast_ctx:
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    new_tokens = generated_ids[0][inputs["input_ids"].shape[1] :]
    output_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    latency = round(time.time() - t0, 3)

    if output_text.startswith("```"):
        lines = output_text.splitlines()
        output_text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    parsed = json.loads(output_text)
    parsed["_latency_s"] = latency
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified direct-audio speech verifier evaluation CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Backend & Model
    parser.add_argument("--backend", type=str, default="gemini", choices=["gemini", "hf_local"], help="Evaluation backend")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash", help="Model name or HF model ID")
    parser.add_argument("--adapter-path", type=str, default=None, help="LoRA adapter directory (for hf_local)")
    parser.add_argument("--reasoning-effort", type=str, default="medium", choices=["none", "low", "medium", "high"], help="Reasoning level for Gemini")
    parser.add_argument("--device", type=str, default="auto", help="Device for hf_local ('auto', 'cuda:0', 'cpu')")

    # Input & Output
    parser.add_argument("--input", type=str, required=True, help="Input JSONL, JSON records, or directory of WAV files")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save raw evaluation output JSON")
    parser.add_argument("--output-report", type=str, default=None, help="Path to save Markdown evaluation report")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export evaluation results to CSV")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent workers for Gemini queries")

    return parser.parse_args()


def load_input_items(input_path: str) -> list[dict[str, Any]]:
    """Load items from JSONL, JSON records, or WAV folder."""
    p = Path(input_path)
    if not p.is_file():
        p = REPO_ROOT / input_path

    items = []
    if p.is_file() and p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    rec = json.loads(line)
                    rec["id"] = rec.get("id", f"sample_{idx+1:03d}")
                    items.append(rec)
    elif p.is_file() and p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [{"id": k, **v} for k, v in data.items()]
    elif p.is_dir():
        wavs = sorted(p.glob("*.wav"))
        for idx, wav in enumerate(wavs):
            items.append({"id": wav.stem, "audio_path": str(wav)})
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return items


def main() -> None:
    args = parse_args()
    items = load_input_items(args.input)
    logger.info("Loaded %d items from %s", len(items), args.input)

    # Backend initialization
    hf_model = None
    hf_processor = None
    actual_device = "cpu"

    if args.backend == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is required for gemini backend.")
            sys.exit(1)
    elif args.backend == "hf_local":
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration

        actual_device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) or args.device.startswith("cuda") else "cpu"
        logger.info("Loading HF base model %s on %s...", args.model, actual_device)
        hf_processor = AutoProcessor.from_pretrained(args.model)
        hf_model = Gemma4ForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=actual_device,
        )
        if args.adapter_path:
            logger.info("Loading LoRA adapter from %s...", args.adapter_path)
            hf_model = PeftModel.from_pretrained(hf_model, args.adapter_path)
        hf_model.eval()

    # Evaluation loop
    results: list[dict[str, Any]] = []

    if args.backend == "gemini":
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_item = {}
            for item in items:
                audio_p = resolve_audio_path(item.get("audio_path", item.get("audio", "")), REPO_ROOT)
                prompt = item.get("prompt", DEFAULT_ACOUSTIC_PROMPT)
                fut = executor.submit(query_gemini, audio_p, args.model, api_key, prompt, args.reasoning_effort)
                future_to_item[fut] = item

            for fut in as_completed(future_to_item):
                item = future_to_item[fut]
                try:
                    res = fut.result()
                    results.append({"item": item, "prediction": res, "success": True})
                    logger.info("Sample %s -> %s (latency: %.2fs)", item.get("id", ""), res.get("decision", ""), res.get("_latency_s", 0))
                except Exception as exc:
                    logger.error("Sample %s failed: %s", item.get("id", ""), exc)
                    results.append({"item": item, "error": str(exc), "success": False})
    else:
        for item in items:
            audio_p = resolve_audio_path(item.get("audio_path", item.get("audio", "")), REPO_ROOT)
            prompt = item.get("prompt", DEFAULT_ACOUSTIC_PROMPT)
            try:
                res = query_hf_local(audio_p, hf_model, hf_processor, actual_device, prompt)
                results.append({"item": item, "prediction": res, "success": True})
                logger.info("Sample %s -> %s (latency: %.2fs)", item.get("id", ""), res.get("decision", ""), res.get("_latency_s", 0))
            except Exception as exc:
                logger.error("Sample %s failed: %s", item.get("id", ""), exc)
                results.append({"item": item, "error": str(exc), "success": False})

    # Summary Statistics
    successful = [r for r in results if r.get("success")]
    pass_count = sum(1 for r in successful if r["prediction"].get("decision") == "pass")
    reject_count = sum(1 for r in successful if r["prediction"].get("decision") == "reject")
    latencies = [r["prediction"]["_latency_s"] for r in successful if "_latency_s" in r["prediction"]]
    avg_latency = round(sum(latencies) / max(1, len(latencies)), 2)

    logger.info("================ EVALUATION SUMMARY ================")
    logger.info("Total Evaluated: %d | Success: %d | Failed: %d", len(results), len(successful), len(results) - len(successful))
    logger.info("Pass: %d (%.1f%%) | Reject: %d (%.1f%%)", pass_count, (pass_count / max(1, len(successful))) * 100, reject_count, (reject_count / max(1, len(successful))) * 100)
    logger.info("Average Latency: %.2fs", avg_latency)

    # Save outputs
    if args.output_json:
        out_p = REPO_ROOT / args.output_json if not Path(args.output_json).is_file() else Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Saved raw results to %s", out_p)

    if args.export_csv:
        csv_p = REPO_ROOT / args.export_csv if not Path(args.export_csv).is_file() else Path(args.export_csv)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id", "audio_path", "decision", "speaker_purity", "word_completeness", "audio_quality", "failure_codes", "reason", "latency_s"]
        rows = []
        for r in successful:
            it = r["item"]
            pred = r["prediction"]
            rows.append({
                "id": it.get("id", ""),
                "audio_path": it.get("audio_path", ""),
                "decision": pred.get("decision", ""),
                "speaker_purity": pred.get("speaker_purity", ""),
                "word_completeness": pred.get("word_completeness", ""),
                "audio_quality": pred.get("audio_quality", ""),
                "failure_codes": ";".join(pred.get("failure_codes", [])),
                "reason": pred.get("reason", ""),
                "latency_s": pred.get("_latency_s", ""),
            })
        with open(csv_p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Exported CSV to %s", csv_p)

    if args.output_report:
        rpt_p = REPO_ROOT / args.output_report if not Path(args.output_report).is_file() else Path(args.output_report)
        rpt_p.parent.mkdir(parents=True, exist_ok=True)
        report_content = f"""# Speech Verifier Evaluation Report

- **Model / Backend:** `{args.model}` (`{args.backend}`)
- **Evaluation Target:** `{args.input}`
- **Total Evaluated:** {len(results)}
- **Successful:** {len(successful)} ({len(successful)/max(1, len(results))*100:.1f}%)
- **Pass Decisions:** {pass_count} ({pass_count/max(1, len(successful))*100:.1f}%)
- **Reject Decisions:** {reject_count} ({reject_count/max(1, len(successful))*100:.1f}%)
- **Average Latency:** {avg_latency}s

## Distribution Breakdown

| Metric | Pass | Reject | Total |
|---|---|---|---|
| Count | {pass_count} | {reject_count} | {len(successful)} |
| Percentage | {pass_count/max(1, len(successful))*100:.1f}% | {reject_count/max(1, len(successful))*100:.1f}% | 100.0% |
"""
        with open(rpt_p, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Saved Markdown report to %s", rpt_p)


if __name__ == "__main__":
    main()
