#!/usr/bin/env python3
"""Evaluate fine-tuned Gemma 4 E4B LoRA adapter on Khanh Vy benchmark turns."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

import librosa
import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
import safetensors.torch
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma4ForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_finetuned")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.diarization.OverlapVerifier import OVERLAP_PROMPT

BASE_MODEL_ID = "google/gemma-4-E4B-it"
ADAPTER_PATH = Path(".data/distillation/checkpoints/best_adapter")
RESULTS_JSON = Path(".data/experiment_khanhvy/results.json")
OUTPUT_CSV = Path(".data/experiment_khanhvy/results_finetuned.csv")
OUTPUT_JSON = Path(".data/experiment_khanhvy/results_finetuned.json")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def clean_json_string(text: str) -> str:
    """Extract raw json from markdown or thought blocks."""
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(\{.*\})", text, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return text.strip()


def parse_response(raw_text: str) -> dict:
    cleaned = clean_json_string(raw_text)
    try:
        data = json.loads(cleaned)
        pure = str(data.get("speaker_purity", "")).lower() == "pure"
        complete = str(data.get("word_completeness", "")).lower() == "complete"
        codes = data.get("failure_codes", [])
        decision = "pass" if (pure and complete and len(codes) == 0) else "reject"
        return {
            "decision": decision,
            "speaker_purity": data.get("speaker_purity", "unknown"),
            "word_completeness": data.get("word_completeness", "unknown"),
            "boundary_issue": data.get("boundary_issue", "unknown"),
            "failure_codes": codes,
            "reason": data.get("reason", ""),
            "raw": raw_text,
        }
    except Exception as exc:
        return {
            "decision": "error",
            "error": str(exc),
            "raw": raw_text,
        }


def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    logger.info("Loading base processor and model...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, token=hf_token)

    device_map = {
        "model.vision_tower": "cpu",
        "model.embed_vision": "cpu",
        "model.audio_tower": "cuda:0",
        "model.embed_audio": "cuda:0",
        "model.language_model": "cuda:0",
        "lm_head": "cuda:0",
    }
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_skip_modules=["audio_tower", "vision_tower", "embed_audio", "embed_vision", "lm_head"],
    )
    base_model = Gemma4ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    logger.info("Attaching LoRA adapter from %s...", ADAPTER_PATH)
    peft_config = LoraConfig.from_pretrained(str(ADAPTER_PATH))
    model = get_peft_model(base_model, peft_config)
    weights = safetensors.torch.load_file(str(ADAPTER_PATH / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, weights)
    model.eval()

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Evaluating %d turns from Khanh Vy benchmark...", len(records))
    finetuned_results = []
    agreements = 0
    total_valid = 0

    for idx, r in enumerate(records):
        turn_id = r["turn_id"]
        wav_path = Path(r["wav_path"])
        if not wav_path.is_file():
            wav_path = Path(".data/experiment_khanhvy/cuts") / f"{turn_id}.wav"

        logger.info("[%d/%d] Evaluating %s...", idx + 1, len(records), turn_id)
        audio_data, _ = librosa.load(str(wav_path), sr=16000)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_data},
                    {"type": "text", "text": OVERLAP_PROMPT},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text_prompt, audio=audio_data, return_tensors="pt", sampling_rate=16000).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        resp_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        del outputs, inputs
        torch.cuda.empty_cache()
        parsed = parse_response(resp_text)
        logger.info("  -> Decision: %s | Codes: %s", parsed["decision"], parsed.get("failure_codes", []))

        gemini_dec = r.get("gemini", {}).get("decision")
        if gemini_dec in {"pass", "reject"} and parsed["decision"] in {"pass", "reject"}:
            total_valid += 1
            if gemini_dec == parsed["decision"]:
                agreements += 1

        r_copy = dict(r)
        r_copy["gemma4_e4b_lora_distilled"] = parsed
        finetuned_results.append(r_copy)

    acc = (agreements / total_valid * 100) if total_valid > 0 else 0.0
    logger.info("=========================================")
    logger.info("Evaluation Complete!")
    logger.info("Gemini 3.8 Flash Agreement: %d/%d (%.1f%%)", agreements, total_valid, acc)
    logger.info("=========================================")

    # Write output JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(finetuned_results, f, indent=2, ensure_ascii=False)

    # Write output CSV
    fieldnames = [
        "turn_id",
        "start_s",
        "end_s",
        "duration_s",
        "gemini_decision",
        "gemma4_e4b_lora_decision",
        "gemma4_e4b_lora_codes",
        "gemma4_e4b_lora_reason",
        "gemma4_e4b_q4_decision",
        "gemma4_12b_decision",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in finetuned_results:
            lora_eval = r.get("gemma4_e4b_lora_distilled", {})
            writer.writerow({
                "turn_id": r.get("turn_id", ""),
                "start_s": r.get("start_s", 0.0),
                "end_s": r.get("end_s", 0.0),
                "duration_s": r.get("duration_s", 0.0),
                "gemini_decision": r.get("gemini", {}).get("decision", ""),
                "gemma4_e4b_lora_decision": lora_eval.get("decision", ""),
                "gemma4_e4b_lora_codes": ";".join(lora_eval.get("failure_codes", [])),
                "gemma4_e4b_lora_reason": lora_eval.get("reason", ""),
                "gemma4_e4b_q4_decision": r.get("gemma4_e4b_q4_xl", {}).get("decision", ""),
                "gemma4_12b_decision": r.get("gemma4_12b_q4_xl", {}).get("decision", ""),
            })

    logger.info("Exported evaluation results to %s and %s", OUTPUT_CSV, OUTPUT_JSON)


if __name__ == "__main__":
    main()
