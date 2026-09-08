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
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from src.diarization.audio_utils import resolve_audio_path
from src.diarization.verifiers import (
    DEFAULT_ACOUSTIC_PROMPT,
    extract_json_payload,
    get_verifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_verifier")


# Compatibility wrappers for external imports
def query_gemini(
    audio_path: Path,
    model: str,
    api_key: str,
    prompt: str,
    reasoning_effort: str = "none",
) -> dict[str, Any]:
    """Compatibility wrapper around GeminiVerifier."""
    from src.diarization.verifiers.GeminiVerifier import GeminiVerifier
    verifier = GeminiVerifier(model=model, api_key=api_key, reasoning_effort=reasoning_effort)
    return verifier.verify(audio_path, prompt)


def query_hf_local(
    audio_path: Path,
    model: Any,
    processor: Any,
    device: str,
    prompt: str,
) -> dict[str, Any]:
    """Compatibility fallback for raw model+processor tuples."""
    from src.diarization.verifiers.BaseVerifier import load_audio_waveform
    import time
    import torch

    t0 = time.time()
    audio_data = load_audio_waveform(audio_path, target_sr=16000)

    if hasattr(model, "generate") and type(model).__name__ == "KimiAudio":
        chats = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": str(audio_path)},
        ]
        _, text_output = model.generate(chats, output_type="text")
        output_text = text_output or ""
    elif hasattr(model, "chat"):
        msgs = [{"role": "user", "content": [prompt, audio_data]}]
        res = model.chat(msgs=msgs, do_sample=False, max_new_tokens=512, generate_audio=False)
        output_text = res[0] if isinstance(res, tuple) else str(res)
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
        with torch.no_grad():
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
    """Compatibility wrapper around EndpointVerifier."""
    from src.diarization.verifiers.EndpointVerifier import EndpointVerifier
    verifier = EndpointVerifier(endpoint=endpoint, model=model_name)
    return verifier.verify(audio_path, prompt)


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
    parser.add_argument(
        "--backend",
        type=str,
        default="hf_local",
        choices=["hf_local", "gemini", "endpoint"],
        help="Evaluation backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-E2B-it",
        help=(
            "Model name or HF model ID (e.g. google/gemma-4-E2B-it, "
            "OpenMOSS-Team/MOSS-Audio-8B-Thinking, openbmb/MiniCPM-o-4_5, "
            "moonshotai/Kimi-Audio-7B-Instruct, gemini-3.8-flash)"
        ),
    )
    parser.add_argument("--adapter-path", type=str, default=None, help="LoRA adapter path or HF repo ID")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="OpenAI / vLLM compatible multimodal audio endpoint",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading custom HF models",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="PyTorch tensor dtype",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high"],
        help="Reasoning level (for Gemini)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for hf_local ('auto', 'cuda:0', 'cpu', 'mps')",
    )

    # Input & Ground Truth
    parser.add_argument("--input", type=str, default=".data/experiment_khanhvy/results.json", help="Input JSON/JSONL or folder of WAV files")
    parser.add_argument("--ground-truth", type=str, default=None, help="Optional external ground-truth JSON/JSONL with Gemini verdicts")

    # Prompt Options
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt string override for evaluation")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to file containing custom prompt")

    # Outputs
    parser.add_argument("--output-json", type=str, default=None, help="Path to save raw evaluation output JSON")
    parser.add_argument("--output-report", type=str, default=None, help="Path to save Markdown evaluation report")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export evaluation results to CSV")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent workers for Gemini/endpoint queries")
    parser.add_argument("--hf-dataset-repo", type=str, default="tungnguyenlam/gemma-4-e2b-acoustic-verifier-data", help="HF dataset repository to fetch missing audio from")

    return parser.parse_args()


def ensure_evaluation_audio(
    items: list[dict[str, Any]],
    repo_root: Path,
    hf_dataset_repo: str = "tungnguyenlam/gemma-4-e2b-acoustic-verifier-data",
) -> None:
    """Check if audio files in evaluation set exist locally, downloading from HF Hub if needed."""
    missing_count = 0
    sample_missing: str | None = None
    for item in items:
        raw_path = item.get("audio_path") or item.get("wav_path") or item.get("audio") or ""
        try:
            p = resolve_audio_path(raw_path, repo_root)
            if not p.is_file():
                missing_count += 1
                if sample_missing is None:
                    sample_missing = str(raw_path)
        except FileNotFoundError:
            missing_count += 1
            if sample_missing is None:
                sample_missing = str(raw_path)

    if missing_count == 0:
        return

    logger.warning(
        "Found %d/%d evaluation audio files missing locally (e.g. %s). Attempting automatic download from Hugging Face Hub...",
        missing_count,
        len(items),
        sample_missing,
    )

    hf_token = os.getenv("HF_TOKEN")
    from huggingface_hub import hf_hub_download
    import tarfile

    archive_name = "khanhvy_cuts.tar.gz" if (sample_missing and "experiment_khanhvy" in sample_missing) else "e2b_audio_dataset.tar.gz"

    try:
        logger.info("Downloading dataset archive '%s' from HF Hub '%s'...", archive_name, hf_dataset_repo)
        tar_path = hf_hub_download(
            repo_id=hf_dataset_repo,
            filename=archive_name,
            repo_type="dataset",
            token=hf_token,
        )
        logger.info("Extracting %s into %s...", tar_path, repo_root)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=str(repo_root))
        logger.info("Successfully extracted audio files into %s.", repo_root)
    except Exception as exc:
        logger.error("Failed to automatically fetch evaluation audio from HF Hub: %s", exc)


def main() -> None:
    args = parse_args()
    items = load_input_items(args.input, args.ground_truth)
    logger.info("Loaded %d evaluation items from %s", len(items), args.input)
    ensure_evaluation_audio(items, REPO_ROOT, args.hf_dataset_repo)

    cli_prompt = None
    if args.prompt_file:
        p_file = Path(args.prompt_file)
        if not p_file.is_file():
            p_file = REPO_ROOT / args.prompt_file
        if not p_file.is_file():
            logger.error("Prompt file not found: %s", args.prompt_file)
            sys.exit(1)
        cli_prompt = p_file.read_text(encoding="utf-8").strip()
        logger.info("Loaded custom evaluation prompt from %s (%d chars)", p_file, len(cli_prompt))
    elif args.prompt:
        cli_prompt = args.prompt.strip()
        logger.info("Using custom evaluation prompt from CLI (%d chars)", len(cli_prompt))

    # Initialize model verifier using modular factory
    logger.info("Initializing verifier (backend=%s, model=%s)...", args.backend, args.model)
    verifier = get_verifier(
        backend=args.backend,
        model=args.model,
        device=args.device,
        adapter_path=args.adapter_path,
        endpoint=args.endpoint,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        reasoning_effort=args.reasoning_effort,
        hf_token=os.getenv("HF_TOKEN"),
    )

    # Evaluation loop
    results: list[dict[str, Any]] = []

    if verifier.supports_concurrency:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_item = {}
            for item in items:
                raw_path = item.get("audio_path") or item.get("wav_path") or item.get("audio") or ""
                audio_p = resolve_audio_path(raw_path, REPO_ROOT)
                prompt = cli_prompt if cli_prompt is not None else item.get("prompt", DEFAULT_ACOUSTIC_PROMPT)
                fut = executor.submit(verifier.verify, audio_p, prompt)
                future_to_item[fut] = (item, prompt)

            for fut in as_completed(future_to_item):
                item, prompt = future_to_item[fut]
                try:
                    res = fut.result()
                    results.append({"item": item, "prompt": prompt, "prediction": res, "success": True})
                    logger.info("Sample %s -> %s (latency: %.2fs)", item.get("id", ""), res.get("decision", ""), res.get("_latency_s", 0))
                except Exception as exc:
                    logger.error("Sample %s failed: %s", item.get("id", ""), exc)
                    results.append({"item": item, "prompt": prompt, "error": str(exc), "success": False})
    else:
        for item in items:
            raw_path = item.get("audio_path") or item.get("wav_path") or item.get("audio") or ""
            audio_p = resolve_audio_path(raw_path, REPO_ROOT)
            prompt = cli_prompt if cli_prompt is not None else item.get("prompt", DEFAULT_ACOUSTIC_PROMPT)
            try:
                res = verifier.verify(audio_p, prompt)
                results.append({"item": item, "prompt": prompt, "prediction": res, "success": True})
                logger.info("Sample %s -> %s (latency: %.2fs)", item.get("id", ""), res.get("decision", ""), res.get("_latency_s", 0))
            except Exception as exc:
                logger.error("Sample %s failed: %s", item.get("id", ""), exc)
                results.append({"item": item, "prompt": prompt, "error": str(exc), "success": False})

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
    sample_prompt = cli_prompt or (results[0].get("prompt") if results else DEFAULT_ACOUSTIC_PROMPT)

    if args.output_json:
        out_p = REPO_ROOT / args.output_json if not Path(args.output_json).is_file() else Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "backend": args.backend,
                "adapter_path": args.adapter_path,
                "prompt": sample_prompt,
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
            "prompt",
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
                "prompt": r.get("prompt", ""),
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

## Evaluation Prompt

```text
{sample_prompt}
```
"""
        with open(rpt_p, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Saved full Markdown benchmark report to %s", rpt_p)


if __name__ == "__main__":
    main()
