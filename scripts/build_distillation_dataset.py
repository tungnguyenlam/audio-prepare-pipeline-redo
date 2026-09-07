#!/usr/bin/env python3
"""Unified CLI for creating, balancing, and publishing speech distillation datasets.

Subcommands:
  - annotate : Annotate raw audio turns using teacher model (Gemini 3.8 Flash).
  - balance  : Stratify and balance pass/reject ratios into train/val JSONL splits.
  - package  : Compress dataset audio into a single archive and upload to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import sys
import tarfile
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_distillation_dataset")

DEFAULT_PROMPT = """Listen to the supplied audio directly. Do not transcribe it.
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


def query_gemini_teacher(wav_path: Path, api_key: str, model: str = "gemini-3.8-flash", reasoning: str = "medium") -> dict[str, Any]:
    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
                    {"text": DEFAULT_PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        },
    }
    if reasoning and reasoning.lower() != "none":
        payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": reasoning.upper()}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    raw_text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return json.loads(raw_text)


def cmd_annotate(args: argparse.Namespace) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is required for teacher annotation.")
        sys.exit(1)

    input_dirs = [Path(d) if Path(d).is_dir() else REPO_ROOT / d for d in args.audio_dirs]
    wav_files: list[Path] = []
    for d in input_dirs:
        wav_files.extend(sorted(d.glob("*.wav")))

    if args.max_samples:
        wav_files = wav_files[: args.max_samples]

    logger.info("Found %d WAV files to annotate with %s across %d directories.", len(wav_files), args.teacher_model, len(input_dirs))

    output_path = REPO_ROOT / args.output_file if not Path(args.output_file).is_file() else Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_wav = {
            executor.submit(query_gemini_teacher, wav, api_key, args.teacher_model, args.reasoning_effort): wav
            for wav in wav_files
        }
        for fut in as_completed(future_to_wav):
            wav = future_to_wav[fut]
            try:
                target_json = fut.result()
                rec = {
                    "audio_path": str(wav),
                    "prompt": DEFAULT_PROMPT,
                    "target_json": target_json,
                    "decision": target_json.get("decision", "reject"),
                }
                annotated.append(rec)
                logger.info("Annotated %s -> %s", wav.name, target_json.get("decision"))
            except Exception as exc:
                logger.error("Annotation failed for %s: %s", wav.name, exc)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in annotated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Saved %d annotated samples to %s", len(annotated), output_path)


def cmd_balance(args: argparse.Namespace) -> None:
    input_paths = [Path(p) if Path(p).is_file() else REPO_ROOT / p for p in args.input_files]
    records = []
    for p in input_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    passes = [r for r in records if r.get("decision") == "pass"]
    rejects = [r for r in records if r.get("decision") == "reject"]
    logger.info("Total input records: %d (Pass: %d, Reject: %d)", len(records), len(passes), len(rejects))

    target_pass_ratio = args.pass_ratio
    max_total = args.target_samples or len(records)

    # Compute balanced counts
    target_passes = min(len(passes), int(max_total * target_pass_ratio))
    target_rejects = min(len(rejects), int(target_passes * ((1.0 - target_pass_ratio) / target_pass_ratio)))

    random.seed(42)
    selected_passes = random.sample(passes, target_passes)
    selected_rejects = random.sample(rejects, target_rejects)
    dataset = selected_passes + selected_rejects
    random.shuffle(dataset)

    # Split train and val
    val_ratio = args.val_ratio
    val_count = max(1, int(len(dataset) * val_ratio))
    val_data = dataset[:val_count]
    train_data = dataset[val_count:]

    out_train = REPO_ROOT / args.train_out if not Path(args.train_out).is_file() else Path(args.train_out)
    out_val = REPO_ROOT / args.val_out if not Path(args.val_out).is_file() else Path(args.val_out)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    with open(out_train, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_val, "w", encoding="utf-8") as f:
        for r in val_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        "Balanced Dataset Created: Train=%d (Pass: %d, Reject: %d) | Val=%d (Pass: %d, Reject: %d)",
        len(train_data),
        sum(1 for r in train_data if r.get("decision") == "pass"),
        sum(1 for r in train_data if r.get("decision") == "reject"),
        len(val_data),
        sum(1 for r in val_data if r.get("decision") == "pass"),
        sum(1 for r in val_data if r.get("decision") == "reject"),
    )
    logger.info("Saved: %s and %s", out_train, out_val)


def cmd_package(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    source_dir = REPO_ROOT / args.audio_dir if not Path(args.audio_dir).is_dir() else Path(args.audio_dir)
    archive_out = REPO_ROOT / args.output_archive if not Path(args.output_archive).is_file() else Path(args.output_archive)
    archive_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Archiving %s to %s...", source_dir, archive_out)
    with tarfile.open(archive_out, "w:gz") as tar:
        rel_path = source_dir.relative_to(REPO_ROOT)
        tar.add(str(source_dir), arcname=str(rel_path))

    size_mb = archive_out.stat().st_size / (1024 * 1024)
    logger.info("Created archive %s (%.1f MB)", archive_out, size_mb)

    if args.hf_repo:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN is required to upload to Hugging Face Hub.")
            sys.exit(1)

        api = HfApi(token=hf_token)
        api.create_repo(repo_id=args.hf_repo, repo_type="dataset", private=False, exist_ok=True)
        logger.info("Uploading %s to HF Hub dataset %s...", archive_out.name, args.hf_repo)
        api.upload_file(
            path_or_fileobj=str(archive_out),
            path_in_repo=archive_out.name,
            repo_id=args.hf_repo,
            repo_type="dataset",
        )
        logger.info("Successfully uploaded dataset to https://huggingface.co/datasets/%s", args.hf_repo)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified speech distillation dataset builder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: annotate
    p_ann = subparsers.add_parser("annotate", help="Annotate audio files with Gemini teacher model")
    p_ann.add_argument("--audio-dirs", nargs="+", required=True, help="Directories containing WAV files")
    p_ann.add_argument("--output-file", type=str, default=".data/distillation/annotated_raw.jsonl", help="Output JSONL path")
    p_ann.add_argument("--teacher-model", type=str, default="gemini-3.8-flash", help="Gemini teacher model")
    p_ann.add_argument("--reasoning-effort", type=str, default="medium", choices=["none", "low", "medium", "high"], help="Reasoning level")
    p_ann.add_argument("--max-samples", type=int, default=None, help="Max samples to annotate")
    p_ann.add_argument("--concurrency", type=int, default=5, help="Concurrent request threads")
    p_ann.set_defaults(func=cmd_annotate)

    # Subcommand: balance
    p_bal = subparsers.add_parser("balance", help="Balance dataset pass/reject distribution and split")
    p_bal.add_argument("--input-files", nargs="+", required=True, help="Input raw JSONL annotation files")
    p_bal.add_argument("--train-out", type=str, default=".data/distillation/train_e2b.jsonl", help="Output train JSONL")
    p_bal.add_argument("--val-out", type=str, default=".data/distillation/val_e2b.jsonl", help="Output val JSONL")
    p_bal.add_argument("--pass-ratio", type=float, default=0.60, help="Target ratio of passing samples (e.g. 0.60 = 60% pass, 40% reject)")
    p_bal.add_argument("--val-ratio", type=float, default=0.20, help="Validation set split ratio (e.g. 0.20 = 20% val)")
    p_bal.add_argument("--target-samples", type=int, default=None, help="Optional max total samples")
    p_bal.set_defaults(func=cmd_balance)

    # Subcommand: package
    p_pkg = subparsers.add_parser("package", help="Compress audio files into tar.gz and optionally upload to HF Hub")
    p_pkg.add_argument("--audio-dir", type=str, default=".data/distillation_e2b/audio", help="Directory of audio WAVs")
    p_pkg.add_argument("--output-archive", type=str, default=".data/distillation/e2b_audio_dataset.tar.gz", help="Output tar.gz path")
    p_pkg.add_argument("--hf-repo", type=str, default=None, help="Hugging Face dataset repository to push (e.g. tungnguyenlam/gemma-4-e2b-acoustic-verifier-data)")
    p_pkg.set_defaults(func=cmd_package)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
