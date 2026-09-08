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
import hashlib
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
    """Split whole recordings, then optionally balance only the training split."""
    if not 0 < args.pass_ratio < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("pass-ratio and val-ratio must be between zero and one.")
    if args.target_samples is not None and args.target_samples < 1:
        raise ValueError("target-samples must be positive.")
    input_paths = [Path(p) if Path(p).is_file() else REPO_ROOT / p for p in args.input_files]
    records = []
    seen_paths = set()
    for p in input_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if not item.get("recording_id"):
                        raise ValueError("Every row needs a verified recording_id; recover or quarantine legacy lineage first.")
                    if item.get("decision") not in {"pass", "reject"}:
                        raise ValueError("Unresolved/unlabeled rows must be quarantined before splitting.")
                    apath = item.get("audio_path")
                    if apath and apath in seen_paths:
                        continue
                    if apath:
                        seen_paths.add(apath)
                    records.append(item)

    recording_ids = sorted({r["recording_id"] for r in records})
    if len(recording_ids) < 2:
        raise ValueError("At least two recordings are required for disjoint train/validation splits.")
    # Shared intros or copied files can leak even when recording IDs differ.
    hashes: dict[str, str] = {}
    for row in records:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = REPO_ROOT / audio_path
        with audio_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate audio bytes found; deduplicate/quarantine before balancing.")
        hashes[digest] = row["recording_id"]
    rng = random.Random(42)
    rng.shuffle(recording_ids)
    if args.validation_recordings:
        validation_ids = set(args.validation_recordings)
        if not validation_ids < set(recording_ids):
            raise ValueError("Validation recordings must be a nonempty proper subset of input recordings.")
    else:
        count = max(1, min(len(recording_ids) - 1, round(len(recording_ids) * args.val_ratio)))
        validation_ids = set(recording_ids[:count])
    val_data = [r for r in records if r["recording_id"] in validation_ids]
    training = [r for r in records if r["recording_id"] not in validation_ids]
    passes = [r for r in training if r["decision"] == "pass"]
    rejects = [r for r in training if r["decision"] == "reject"]

    target_pass_ratio = args.pass_ratio
    max_total = args.target_samples or len(training)

    # Compute balanced counts
    target_passes = min(len(passes), int(max_total * target_pass_ratio))
    target_rejects = min(len(rejects), int(target_passes * ((1.0 - target_pass_ratio) / target_pass_ratio)))

    if not target_passes or not target_rejects:
        raise ValueError("Training split cannot supply both classes at this balance; collect more sources.")
    train_data = rng.sample(passes, target_passes) + rng.sample(rejects, target_rejects)
    rng.shuffle(train_data)

    out_train = REPO_ROOT / args.train_out if not Path(args.train_out).is_file() else Path(args.train_out)
    out_val = REPO_ROOT / args.val_out if not Path(args.val_out).is_file() else Path(args.val_out)
    if out_train.resolve() == out_val.resolve() or out_train.exists() or out_val.exists():
        raise ValueError("Choose distinct new output paths; existing splits are never overwritten.")
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
    logger.info("Validation recordings (not class-balanced): %s", sorted(validation_ids))


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


def cmd_augment_boundaries(args: argparse.Namespace) -> None:
    """Generate synthetic hard-negative boundary clipping examples from clean passing samples."""
    import numpy as np
    import soundfile as sf
    from src.diarization.verifier_training import resolve_audio_path

    out_audio_dir = REPO_ROOT / args.output_audio_dir if not Path(args.output_audio_dir).is_dir() else Path(args.output_audio_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = REPO_ROOT / args.output_jsonl if not Path(args.output_jsonl).is_file() else Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    input_file = REPO_ROOT / args.input_file if not Path(args.input_file).is_file() else Path(args.input_file)
    records: list[dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter passing samples
    pass_samples = [
        r for r in records
        if r.get("decision") == "pass" or (
            r.get("target_json", {}).get("speaker_purity") == "pure"
            and r.get("target_json", {}).get("word_completeness") == "complete"
        )
    ]
    logger.info("Found %d clean passing candidate samples for boundary augmentation.", len(pass_samples))

    if args.max_augmentations and len(pass_samples) > args.max_augmentations // 2:
        pass_samples = random.sample(pass_samples, args.max_augmentations // 2)

    augmented_records: list[dict[str, Any]] = []
    count_end = 0
    count_start = 0

    for sample in pass_samples:
        raw_path = sample.get("audio_path") or sample.get("wav_path") or sample.get("audio_filepath") or ""
        audio_p = resolve_audio_path(raw_path, REPO_ROOT)
        if not audio_p.exists():
            continue

        try:
            data, sr = sf.read(str(audio_p))
        except Exception as e:
            logger.warning("Could not read %s: %s", audio_p, e)
            continue

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        dur_s = len(data) / sr
        if dur_s < 1.0:
            continue

        base_stem = audio_p.stem

        # 1. Generate clipped_word_end (chop off 80ms - 200ms from vocal end)
        cut_ms_end = random.uniform(80.0, 200.0)
        samples_to_cut_end = int((cut_ms_end / 1000.0) * sr)
        if len(data) > samples_to_cut_end + int(0.5 * sr):
            clipped_end_data = data[:-samples_to_cut_end]
            end_wav_path = out_audio_dir / f"{base_stem}_synth_clipped_end.wav"
            sf.write(str(end_wav_path), clipped_end_data, sr)

            target_end = {
                "speaker_purity": "pure",
                "word_completeness": "clipped_word_end",
                "audio_quality": "studio_clean",
                "decision": "reject",
                "failure_codes": ["clipped_word_end"],
                "reason": (
                    f"The final syllable is cut off abruptly mid-vocalization ({cut_ms_end:.0f}ms premature truncation), "
                    "cutting off its natural acoustic decay and coda consonant closure."
                ),
            }
            augmented_records.append({
                "audio_path": str(end_wav_path),
                "prompt": sample.get("prompt", DEFAULT_PROMPT),
                "target_json": target_end,
                "decision": "reject",
                "source": "synthetic_clipped_end",
                "duration_s": round(len(clipped_end_data) / sr, 3),
            })
            count_end += 1

        # 2. Generate clipped_word_start (chop off 60ms - 160ms from vocal onset)
        cut_ms_start = random.uniform(60.0, 160.0)
        samples_to_cut_start = int((cut_ms_start / 1000.0) * sr)
        if len(data) > samples_to_cut_start + int(0.5 * sr):
            clipped_start_data = data[samples_to_cut_start:]
            start_wav_path = out_audio_dir / f"{base_stem}_synth_clipped_start.wav"
            sf.write(str(start_wav_path), clipped_start_data, sr)

            target_start = {
                "speaker_purity": "pure",
                "word_completeness": "clipped_word_start",
                "audio_quality": "studio_clean",
                "decision": "reject",
                "failure_codes": ["clipped_word_start"],
                "reason": (
                    f"The initial syllable is abruptly truncated at the onset ({cut_ms_start:.0f}ms onset attack missing), "
                    "cutting off its initial consonant closure."
                ),
            }
            augmented_records.append({
                "audio_path": str(start_wav_path),
                "prompt": sample.get("prompt", DEFAULT_PROMPT),
                "target_json": target_start,
                "decision": "reject",
                "source": "synthetic_clipped_start",
                "duration_s": round(len(clipped_start_data) / sr, 3),
            })
            count_start += 1

    # Combine original records + augmented records
    all_combined = records + augmented_records
    random.shuffle(all_combined)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in all_combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        "Augmentation Complete: Generated %d clipped_end + %d clipped_start = %d synthetic negatives.",
        count_end, count_start, len(augmented_records),
    )
    logger.info(
        "Total Output Dataset: %d samples (Pass: %d, Reject: %d) saved to %s",
        len(all_combined),
        sum(1 for r in all_combined if r.get("decision") == "pass"),
        sum(1 for r in all_combined if r.get("decision") == "reject"),
        out_jsonl,
    )


def cmd_slice_candidates(args: argparse.Namespace) -> None:
    """Slice raw audio tracks into speech turn candidates using acoustic energy valleys."""
    import numpy as np
    import soundfile as sf

    out_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_dir() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks: list[Path] = []
    for inp in args.input_tracks:
        p = REPO_ROOT / inp if not Path(inp).exists() else Path(inp)
        if p.is_file() and p.suffix.lower() in [".wav", ".mp3", ".flac"]:
            tracks.append(p)
        elif p.is_dir():
            tracks.extend(sorted(p.rglob("*.wav")))

    logger.info("Found %d audio tracks to slice candidates from.", len(tracks))
    min_dur = args.min_duration
    max_dur = args.max_duration
    total_saved = 0

    for track_path in tracks:
        try:
            data, sr = sf.read(str(track_path))
        except Exception as e:
            logger.warning("Failed reading %s: %s", track_path, e)
            continue

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        dur_s = len(data) / sr
        logger.info("Slicing candidates from %s (%.1fs)...", track_path.name, dur_s)

        frame_len = int(0.020 * sr)
        hop_len = int(0.010 * sr)
        n_frames = (len(data) - frame_len) // hop_len + 1
        if n_frames <= 0:
            continue

        frames = np.lib.stride_tricks.sliding_window_view(data[: n_frames * hop_len + frame_len], frame_len)[::hop_len]
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
        max_rms = np.max(rms) + 1e-9
        db = 20 * np.log10(rms / max_rms)

        is_speech = db > args.threshold_db
        regions: list[tuple[float, float]] = []
        in_speech = False
        start_idx = 0
        for i, s in enumerate(is_speech):
            if s and not in_speech:
                in_speech = True
                start_idx = i
            elif not s and in_speech:
                in_speech = False
                seg_dur = (i - start_idx) * hop_len / sr
                if seg_dur >= min_dur:
                    regions.append((start_idx * hop_len / sr, i * hop_len / sr))

        cuts: list[tuple[float, float]] = []
        for s_t, e_t in regions:
            seg_dur = e_t - s_t
            if seg_dur <= max_dur:
                cuts.append((s_t, e_t))
            else:
                n_chunks = int(np.ceil(seg_dur / 8.0))
                chunk_len = seg_dur / n_chunks
                for c in range(n_chunks):
                    cuts.append((s_t + c * chunk_len, s_t + (c + 1) * chunk_len))

        stem = track_path.stem[:25]
        track_saved = 0
        for c_idx, (st, et) in enumerate(cuts):
            if args.max_cuts_per_track and track_saved >= args.max_cuts_per_track:
                break
            s_samp = max(0, int((st - 0.05) * sr))
            e_samp = min(len(data), int((et + 0.06) * sr))
            clip = data[s_samp:e_samp]
            if len(clip) / sr < min_dur:
                continue

            out_wav = out_dir / f"{stem}_turn_{c_idx:03d}_{st:.1f}-{et:.1f}.wav"
            sf.write(str(out_wav), clip, sr)
            total_saved += 1
            track_saved += 1

    logger.info("Candidate Slicing Complete: Generated %d audio cuts in %s", total_saved, out_dir)


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
    p_bal.add_argument("--val-ratio", type=float, default=0.20, help="Fraction of recording groups held for validation")
    p_bal.add_argument("--validation-recordings", nargs="+", help="Explicit recording IDs to hold for validation")
    p_bal.add_argument("--target-samples", type=int, default=None, help="Optional maximum training samples after recording split")
    p_bal.set_defaults(func=cmd_balance)

    # Subcommand: augment-boundaries
    p_aug = subparsers.add_parser("augment-boundaries", help="Generate synthetic boundary-clipping negatives from clean turns")
    p_aug.add_argument("--input-file", type=str, required=True, help="Input JSONL file containing passing samples")
    p_aug.add_argument("--output-audio-dir", type=str, default=".data/distillation/augmented_audio", help="Directory to save augmented WAVs")
    p_aug.add_argument("--output-jsonl", type=str, default=".data/distillation/train_augmented.jsonl", help="Output augmented JSONL file")
    p_aug.add_argument("--max-augmentations", type=int, default=300, help="Max synthetic samples to generate")
    p_aug.set_defaults(func=cmd_augment_boundaries)

    # Subcommand: slice-candidates
    p_sli = subparsers.add_parser("slice-candidates", help="Slice audio tracks into candidate speech turns using energy valleys")
    p_sli.add_argument("--input-tracks", nargs="+", required=True, help="Input audio files or directories")
    p_sli.add_argument("--output-dir", type=str, default=".data/distillation/crawled_cuts", help="Directory to save sliced turns")
    p_sli.add_argument("--min-duration", type=float, default=2.0, help="Minimum duration of speech turn in seconds")
    p_sli.add_argument("--max-duration", type=float, default=12.0, help="Maximum duration of speech turn in seconds")
    p_sli.add_argument("--threshold-db", type=float, default=-34.0, help="Silence threshold in dBFS")
    p_sli.add_argument("--max-cuts-per-track", type=int, default=35, help="Max cuts per input track")
    p_sli.set_defaults(func=cmd_slice_candidates)

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
