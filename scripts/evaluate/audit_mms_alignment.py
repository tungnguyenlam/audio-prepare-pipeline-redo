#!/usr/bin/env python3
"""Align all 100 Vietnamese stop-coda utterances using MMS-FA and prepare exact boundary cuts."""
from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger("audit.mms_fa")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ).replace("đ", "d").replace("Đ", "d")


def main() -> None:
    base_anchors_dir = Path(".data/vicut_bench/anchors")
    manifest_path = base_anchors_dir / "vicut_bench_anchors_300.jsonl"
    output_dir = Path(".data/goal/gemini_checks/mms_fa_audit_100/clips")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = Path(".data/goal/gemini_checks/mms_fa_audit_100/alignment_report.json")

    anchors = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    stop_codas = [a for a in anchors if a.get("is_stop_coda")]
    logger.info("Found %d stop-coda anchors in %s", len(stop_codas), manifest_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading MMS_FA on %s...", device)
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(device)
    aligner = bundle.get_aligner()
    tok = bundle.get_tokenizer()

    results = []
    for idx, item in enumerate(stop_codas, 1):
        audio_path = base_anchors_dir / item["rel_path"]
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        waveform = torch.from_numpy(mono).unsqueeze(0)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        words_orig = item["text"].split()
        words_norm = [re.sub(r"[^a-z\']", "", strip_accents(w).lower()) for w in words_orig]
        tokens = tok(words_norm)

        with torch.no_grad():
            emission, _ = model(waveform.to(device))
            spans = aligner(emission[0], tokens)

        num_frames = emission.size(1)
        ratio = waveform.size(1) / num_frames / bundle.sample_rate

        start_time_s = spans[0][0].start * ratio
        end_time_s = spans[-1][-1].end * ratio
        duration_s = end_time_s - start_time_s

        # Slice exact audio at MMS-FA boundaries
        start_sample = int(round(start_time_s * sr))
        end_sample = int(round(end_time_s * sr))
        cut_data = mono[start_sample:end_sample]

        out_clip_path = output_dir / f"{item['id']}__mms_exact.wav"
        sf.write(out_clip_path, cut_data, sr)

        info = {
            "id": item["id"],
            "speaker": item.get("speaker"),
            "text": item["text"],
            "final_word": item.get("final_word"),
            "source_path": str(audio_path),
            "mms_clip_path": str(out_clip_path),
            "original_duration_s": round(len(mono) / sr, 4),
            "mms_start_s": round(start_time_s, 4),
            "mms_end_s": round(end_time_s, 4),
            "mms_cut_duration_s": round(duration_s, 4),
        }
        results.append(info)
        if idx % 20 == 0 or idx == len(stop_codas):
            logger.info("[%d/%d] Processed %s -> %.3fs (dur: %.3fs)", idx, len(stop_codas), item["id"], end_time_s, duration_s)

    report_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved 100-sample alignment report to %s", report_file)


if __name__ == "__main__":
    main()
