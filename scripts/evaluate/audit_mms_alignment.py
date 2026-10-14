#!/usr/bin/env python3
"""Align Vietnamese utterances using MMS-FA and prepare exact boundary cuts for Gemini audit."""
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

CLIPS = [
    {
        "id": "vivos_VIVOSDEV01_R027",
        "audio_path": ".data/vicut_bench/anchors/vivos/vivos_VIVOSDEV01_VIVOSDEV01_R027.wav",
        "text": "DI TÍCH CÓ BẢY PHỖNG ÔNG VÀ HAI PHỖNG BÀ ĐƯỢC LÀM BẰNG GỖ MÍT",
        "final_word": "mít",
    },
    {
        "id": "vivos_VIVOSDEV03_R136",
        "audio_path": ".data/vicut_bench/anchors/vivos/vivos_VIVOSDEV03_VIVOSDEV03_R136.wav",
        "text": "NÀY BÓN CHO CON THƯƠNG ĐỠ YẾU ĐAU CÒI CỌC",
        "final_word": "cọc",
    },
    {
        "id": "vivos_VIVOSDEV05_043",
        "audio_path": ".data/vicut_bench/anchors/vivos/vivos_VIVOSDEV05_VIVOSDEV05_043.wav",
        "text": "NGUYÊN THỨ TRƯỞNG MAI VĂN DÂU ĐÃ BỊ BẮT",
        "final_word": "bắt",
    },
    {
        "id": "vivos_VIVOSDEV05_120",
        "audio_path": ".data/vicut_bench/anchors/vivos/vivos_VIVOSDEV05_VIVOSDEV05_120.wav",
        "text": "NHƯNG MÙI HÔI VẪN XỘC VÀO TẬN ÓC",
        "final_word": "óc",
    },
    {
        "id": "vivos_VIVOSDEV06_070",
        "audio_path": ".data/vicut_bench/anchors/vivos/vivos_VIVOSDEV06_VIVOSDEV06_070.wav",
        "text": "RỒI KHỀ KHÀ ĐÔI CÂU VỌNG CỔ VỚI CHÂN CHUỘT",
        "final_word": "chuột",
    },
]


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ).replace("đ", "d").replace("Đ", "d")


def main() -> None:
    output_dir = Path(".data/goal/gemini_checks/mms_fa_audit/clips")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = Path(".data/goal/gemini_checks/mms_fa_audit/alignment_report.json")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading MMS_FA on %s...", device)
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(device)
    aligner = bundle.get_aligner()
    tok = bundle.get_tokenizer()

    results = []
    for item in CLIPS:
        audio_path = Path(item["audio_path"])
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
            "text": item["text"],
            "final_word": item["final_word"],
            "source_path": str(audio_path),
            "mms_clip_path": str(out_clip_path),
            "original_duration_s": len(mono) / sr,
            "mms_start_s": round(start_time_s, 4),
            "mms_end_s": round(end_time_s, 4),
            "mms_cut_duration_s": round(duration_s, 4),
        }
        results.append(info)
        logger.info(
            "%s: aligned %.3fs -> %.3fs (dur: %.3fs) written to %s",
            item["id"],
            start_time_s,
            end_time_s,
            duration_s,
            out_clip_path.name,
        )

    report_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved alignment report to %s", report_file)


if __name__ == "__main__":
    main()
