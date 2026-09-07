#!/usr/bin/env python3
"""Unified CLI for evaluating direct-audio speech verifiers and benchmarking against Gemini 3.8 Flash.

Supports evaluating any Hugging Face multimodal audio model (base or fine-tuned with LoRA)
or Gemini API models on:
  - The 31 Khanh Vy benchmark cuts (`.data/experiment_khanhvy/results.json` or directory of cuts)
  - Distillation datasets (`.data/distillation/val_e2b.jsonl`, `train_e2b.jsonl`)
  - Any directory of WAV files

Computes:
  - Model distribution (Pass / Reject counts, percentage, latency)
  - Agreement rate with Gemini 3.8 Flash reference
  - Confusion matrix (True Pass, True Reject, Contamination Leaks, False Rejects)
  - Precision, Recall, F1
  - Side-by-side disagreement table with model vs Gemini reasoning
  - Markdown summary reports and spreadsheet-ready CSV export
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


def extract_json_payload(text: str) -> dict[str, Any]:
    """Robustly extract and parse JSON object from model output text."""
    import re

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback: search for first { and last }
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


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

    # Check if model provides custom .chat() interface (e.g., MiniCPM-o)
    if hasattr(model, "chat") and not hasattr(processor, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        res = model.chat(image=None, audio=audio_data, msgs=msgs, tokenizer=processor)
        output_text = res if isinstance(res, str) else str(res)
    else:
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
    parsed = extract_json_payload(output_text)
    parsed["_latency_s"] = latency
    return parsed


def query_endpoint(
    audio_path: Path,
    endpoint: str,
    model_name: str,
    prompt: str,
) -> dict[str, Any]:
    """Query OpenAI / vLLM compatible multimodal audio endpoint."""
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    t0 = time.time()
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    latency = round(time.time() - t0, 3)

    raw_text = res["choices"][0]["message"]["content"]
    parsed = extract_json_payload(raw_text)
    parsed["_latency_s"] = latency
    return parsed


def load_input_items(input_path: str, ground_truth_file: str | None = None) -> list[dict[str, Any]]:
    """Load items from JSONL, JSON records, or WAV folder, attaching reference truth if present."""
    p = Path(input_path)
    if not p.is_file() and not p.is_dir():
        p = REPO_ROOT / input_path

    items: list[dict[str, Any]] = []
    if p.is_file() and p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    rec = json.loads(line)
                    rec["id"] = rec.get("id", rec.get("video_id", f"sample_{idx+1:03d}"))
                    items.append(rec)
    elif p.is_file() and p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    item["id"] = item.get("turn_id", item.get("id", f"sample_{idx+1:03d}"))
                    items.append(item)
            elif isinstance(data, dict):
                items = [{"id": k, **v} for k, v in data.items()]
    elif p.is_dir():
        wavs = sorted(p.glob("*.wav"))
        for idx, wav in enumerate(wavs):
            items.append({"id": wav.stem, "audio_path": str(wav)})
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Merge ground truth if explicitly provided as a separate file
    if ground_truth_file:
        gt_p = Path(ground_truth_file) if Path(ground_truth_file).is_file() else REPO_ROOT / ground_truth_file
        with open(gt_p, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_map = {}
        if isinstance(gt_data, list):
            for entry in gt_data:
                key = entry.get("turn_id") or entry.get("id") or Path(entry.get("wav_path", "")).stem
                if key:
                    gt_map[key] = entry.get("gemini") or entry.get("target_json") or entry
        for item in items:
            key = item.get("turn_id") or item.get("id") or Path(item.get("audio_path", item.get("wav_path", ""))).stem
            if key in gt_map and "gemini" not in item:
                item["gemini"] = gt_map[key]

    return items


def extract_reference_verdict(item: dict[str, Any]) -> dict[str, Any] | None:
    """Extract reference Gemini verdict from item if available."""
    if "gemini" in item and isinstance(item["gemini"], dict):
        return item["gemini"]
    if "target_json" in item and isinstance(item["target_json"], dict):
        return item["target_json"]
    if "decision" in item and isinstance(item["decision"], str):
        return {"decision": item["decision"]}
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified speech verifier evaluation & Gemini 3.8 Flash benchmark CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Backend & Model
    parser.add_argument("--backend", type=str, default="hf_local", choices=["hf_local", "gemini", "endpoint"], help="Evaluation backend")
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it", help="Model name or HF model ID (e.g. google/gemma-4-E2B-it, openbmb/MiniCPM-o-4_5, moonshotai/Kimi-Audio-7B-Instruct, gemini-3.5-flash-lite)")
    parser.add_argument("--adapter-path", type=str, default=None, help="LoRA adapter path or HF repo ID (e.g. .data/distillation/checkpoints_e2b/best_adapter)")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/chat/completions", help="OpenAI / vLLM compatible multimodal audio endpoint (for backend=endpoint)")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code when loading custom HF models (e.g. MiniCPM-o, Kimi-Audio)")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="PyTorch tensor dtype")
    parser.add_argument("--reasoning-effort", type=str, default="medium", choices=["none", "low", "medium", "high"], help="Reasoning level (for Gemini)")
    parser.add_argument("--device", type=str, default="auto", help="Device for hf_local ('auto', 'cuda:0', 'cpu')")

    # Input & Ground Truth
    parser.add_argument("--input", type=str, default=".data/experiment_khanhvy/results.json", help="Input JSON/JSONL or folder of WAV files")
    parser.add_argument("--ground-truth", type=str, default=None, help="Optional external ground-truth JSON/JSONL with Gemini verdicts")

    # Outputs
    parser.add_argument("--output-json", type=str, default=None, help="Path to save raw evaluation output JSON")
    parser.add_argument("--output-report", type=str, default=None, help="Path to save Markdown evaluation report")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export evaluation results to CSV")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent workers for Gemini/endpoint queries")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_input_items(args.input, args.ground_truth)
    logger.info("Loaded %d evaluation items from %s", len(items), args.input)

    # Initialize model backend
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
        from transformers import AutoProcessor, AutoModelForConditionalGeneration

        actual_device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) or args.device.startswith("cuda") else "cpu"
        logger.info("Loading HF model '%s' on %s (trust_remote_code=%s)...", args.model, actual_device, args.trust_remote_code)
        hf_token = os.getenv("HF_TOKEN")
        try:
            hf_processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, token=hf_token)
        except Exception:
            from transformers import AutoTokenizer
            hf_processor = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, token=hf_token)

        dtype = getattr(torch, args.torch_dtype, torch.bfloat16)
        try:
            hf_model = AutoModelForConditionalGeneration.from_pretrained(
                args.model,
                torch_dtype=dtype,
                device_map=actual_device,
                trust_remote_code=args.trust_remote_code,
                token=hf_token,
            )
        except Exception:
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained(
                args.model,
                torch_dtype=dtype,
                device_map=actual_device,
                trust_remote_code=args.trust_remote_code,
                token=hf_token,
            )

        if args.adapter_path:
            logger.info("Attaching LoRA adapter from '%s'...", args.adapter_path)
            hf_model = PeftModel.from_pretrained(hf_model, args.adapter_path)
        hf_model.eval()

    # Evaluation loop
    results: list[dict[str, Any]] = []

    if args.backend == "gemini":
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_item = {}
            for item in items:
                raw_path = item.get("audio_path") or item.get("wav_path") or item.get("audio") or ""
                audio_p = resolve_audio_path(raw_path, REPO_ROOT)
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
    elif args.backend == "endpoint":
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_item = {}
            for item in items:
                raw_path = item.get("audio_path") or item.get("wav_path") or item.get("audio") or ""
                audio_p = resolve_audio_path(raw_path, REPO_ROOT)
                prompt = item.get("prompt", DEFAULT_ACOUSTIC_PROMPT)
                fut = executor.submit(query_endpoint, audio_p, args.endpoint, args.model, prompt)
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
            raw_path = item.get("audio_path") or item.get("wav_path") or item.get("audio") or ""
            audio_p = resolve_audio_path(raw_path, REPO_ROOT)
            prompt = item.get("prompt", DEFAULT_ACOUSTIC_PROMPT)
            try:
                res = query_hf_local(audio_p, hf_model, hf_processor, actual_device, prompt)
                results.append({"item": item, "prediction": res, "success": True})
                logger.info("Sample %s -> %s (latency: %.2fs)", item.get("id", ""), res.get("decision", ""), res.get("_latency_s", 0))
            except Exception as exc:
                logger.error("Sample %s failed: %s", item.get("id", ""), exc)
                results.append({"item": item, "error": str(exc), "success": False})

    # Summary Statistics & Comparisons
    successful = [r for r in results if r.get("success")]
    pass_count = sum(1 for r in successful if r["prediction"].get("decision") == "pass")
    reject_count = sum(1 for r in successful if r["prediction"].get("decision") == "reject")
    latencies = [r["prediction"]["_latency_s"] for r in successful if "_latency_s" in r["prediction"]]
    avg_latency = round(sum(latencies) / max(1, len(latencies)), 2)

    # Compute comparison against Gemini reference
    comparisons = []
    tp, tn, fp, fn = 0, 0, 0, 0
    disagreements = []

    for r in successful:
        item = r["item"]
        pred = r["prediction"]
        ref = extract_reference_verdict(item)
        if ref and "decision" in ref:
            m_dec = pred.get("decision")
            g_dec = ref.get("decision")
            agree = (m_dec == g_dec)
            comparisons.append(agree)

            if g_dec == "pass" and m_dec == "pass":
                tp += 1
            elif g_dec == "reject" and m_dec == "reject":
                tn += 1
            elif g_dec == "reject" and m_dec == "pass":
                fp += 1  # False positive: Contamination leak!
                disagreements.append({
                    "id": item.get("id", ""),
                    "gemini_decision": g_dec,
                    "model_decision": m_dec,
                    "gemini_reason": ref.get("reason", ""),
                    "model_reason": pred.get("reason", ""),
                    "issue_type": "False Pass (Contamination Leak)",
                })
            elif g_dec == "pass" and m_dec == "reject":
                fn += 1  # False negative: Overly conservative
                disagreements.append({
                    "id": item.get("id", ""),
                    "gemini_decision": g_dec,
                    "model_decision": m_dec,
                    "gemini_reason": ref.get("reason", ""),
                    "model_reason": pred.get("reason", ""),
                    "issue_type": "False Reject (Overly Strict)",
                })

    has_ref = len(comparisons) > 0
    agreement_rate = (sum(comparisons) / max(1, len(comparisons))) * 100 if has_ref else 0.0
    precision = (tp / max(1, tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / max(1, tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / max(1e-6, precision + recall)) if (precision + recall) > 0 else 0.0

    logger.info("================ EVALUATION SUMMARY ================")
    logger.info("Model: %s | Evaluated: %d | Success: %d", args.model, len(results), len(successful))
    logger.info("Distribution: Pass=%d (%.1f%%) | Reject=%d (%.1f%%)", pass_count, (pass_count / max(1, len(successful))) * 100, reject_count, (reject_count / max(1, len(successful))) * 100)
    logger.info("Average Latency: %.2fs", avg_latency)

    if has_ref:
        logger.info("---------------- GEMINI 3.8 FLASH BENCHMARK ----------------")
        logger.info("Evaluated against Gemini ground-truth (%d samples):", len(comparisons))
        logger.info("  Agreement Rate: %.1f%% (%d/%d matches)", agreement_rate, sum(comparisons), len(comparisons))
        logger.info("  True Positives (Pass/Pass): %d | True Negatives (Reject/Reject): %d", tp, tn)
        logger.info("  False Passes (Contamination Leaks): %d | False Rejects (Overly Strict): %d", fp, fn)
        logger.info("  Precision: %.1f%% | Recall: %.1f%% | F1: %.1f%%", precision, recall, f1)

    # Save outputs
    if args.output_json:
        out_p = REPO_ROOT / args.output_json if not Path(args.output_json).is_file() else Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "backend": args.backend,
                "adapter_path": args.adapter_path,
                "summary": {
                    "total": len(results),
                    "success": len(successful),
                    "pass_count": pass_count,
                    "reject_count": reject_count,
                    "avg_latency_s": avg_latency,
                    "agreement_rate": agreement_rate if has_ref else None,
                    "f1_score": f1 if has_ref else None,
                },
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        logger.info("Saved raw JSON results to %s", out_p)

    if args.export_csv:
        csv_p = REPO_ROOT / args.export_csv if not Path(args.export_csv).is_file() else Path(args.export_csv)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id",
            "audio_path",
            "model_decision",
            "gemini_decision",
            "agreement",
            "model_reason",
            "gemini_reason",
            "latency_s",
        ]
        rows = []
        for r in successful:
            it = r["item"]
            pred = r["prediction"]
            ref = extract_reference_verdict(it) or {}
            m_dec = pred.get("decision", "")
            g_dec = ref.get("decision", "")
            rows.append({
                "id": it.get("id", ""),
                "audio_path": it.get("audio_path") or it.get("wav_path", ""),
                "model_decision": m_dec,
                "gemini_decision": g_dec,
                "agreement": (m_dec == g_dec) if g_dec else "",
                "model_reason": pred.get("reason", ""),
                "gemini_reason": ref.get("reason", ""),
                "latency_s": pred.get("_latency_s", ""),
            })
        with open(csv_p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Exported side-by-side comparison CSV to %s", csv_p)

    if args.output_report:
        rpt_p = REPO_ROOT / args.output_report if not Path(args.output_report).is_file() else Path(args.output_report)
        rpt_p.parent.mkdir(parents=True, exist_ok=True)

        disagree_table = ""
        if disagreements:
            rows_md = "\n".join([
                f"| `{d['id']}` | **{d['gemini_decision'].upper()}** | **{d['model_decision'].upper()}** | {d['issue_type']} | {d['gemini_reason']} | {d['model_reason']} |"
                for d in disagreements
            ])
            disagree_table = f"""
## Disagreements with Gemini 3.8 Flash ({len(disagreements)} samples)

| Sample ID | Gemini 3.8 Flash | Candidate Model | Diagnostic | Gemini Reason | Candidate Model Reason |
|---|---|---|---|---|---|
{rows_md}
"""

        benchmark_section = ""
        if has_ref:
            benchmark_section = f"""
## Benchmark vs Gemini 3.8 Flash Teacher

- **Total Ground-Truth Reference Items:** {len(comparisons)}
- **Overall Agreement Rate:** **{agreement_rate:.1f}%** ({sum(comparisons)}/{len(comparisons)})
- **Precision (TTS Purity Confidence):** {precision:.1f}%
- **Recall (Passing Yield Retention):** {recall:.1f}%
- **F1 Score:** {f1:.1f}%

### Confusion Matrix

| Reference \\ Model | Model Pass | Model Reject | Total Reference |
|---|---|---|---|
| **Gemini Pass** | {tp} (True Pass) | {fn} (False Reject) | {tp + fn} |
| **Gemini Reject** | {fp} (Contamination Leak) | {tn} (True Reject) | {fp + tn} |
| **Total Model** | {tp + fp} | {fn + tn} | {len(comparisons)} |
"""

        report_content = f"""# Speech Verifier Benchmark Report

- **Evaluated Model:** `{args.model}` ({args.backend})
- **LoRA Adapter:** `{args.adapter_path or "None (Base Model)"}`
- **Evaluation Dataset:** `{args.input}`
- **Total Evaluated:** {len(results)}
- **Successful:** {len(successful)} ({len(successful)/max(1, len(results))*100:.1f}%)
- **Average Latency:** {avg_latency}s

## Verdict Distribution

| Verdict | Count | Percentage |
|---|---|---|
| **Pass** | {pass_count} | {pass_count/max(1, len(successful))*100:.1f}% |
| **Reject** | {reject_count} | {reject_count/max(1, len(successful))*100:.1f}% |
| **Total** | {len(successful)} | 100.0% |

{benchmark_section}
{disagree_table}
"""
        with open(rpt_p, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Saved full Markdown benchmark report to %s", rpt_p)


if __name__ == "__main__":
    main()
