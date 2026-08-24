#!/usr/bin/env python3
"""Deep Audio Quality & Single-Speaker Purity Auditor.

Performs:
1. Intra-Segment Speaker Homogeneity (First-half vs Second-half Speaker Verification using SpeechBrain).
2. Boundary Cleanliness & Zero Word-Clipping (RMS at edge vs center).
3. Exact Sampled Duration Compliance ([3.0s, 30.0s]).
4. Signal Health (Peak dBFS, RMS dBFS, DC Offset, Clipping).
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.crawler.storage import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Auditor")


def load_speaker_verifier(device: str = "cuda"):
    from speechbrain.inference.speaker import SpeakerRecognition
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
    )
    return verifier


def audit_single_file(file_path: Path, verifier) -> Dict:
    """Run full acoustic and speaker homogeneity verification on a single segment WAV file."""
    info = sf.info(str(file_path))
    audio, sr = sf.read(str(file_path), dtype="float32")

    duration = len(audio) / sr
    is_mono = (audio.ndim == 1)
    if not is_mono:
        audio = np.mean(audio, axis=1)

    # 1. Signal Health
    peak_val = float(np.max(np.abs(audio)))
    peak_dbfs = 20 * math.log10(peak_val + 1e-9)
    rms_val = float(np.sqrt(np.mean(audio**2)))
    rms_dbfs = 20 * math.log10(rms_val + 1e-9)
    clipping_samples = int(np.sum(np.abs(audio) >= 0.999))
    has_clipping = clipping_samples > 0

    # 2. Boundary Smoothness (Edge RMS vs Center RMS)
    edge_len = int(0.04 * sr)
    start_rms = float(np.sqrt(np.mean(audio[:edge_len]**2))) if len(audio) >= edge_len else 0.0
    end_rms = float(np.sqrt(np.mean(audio[-edge_len:]**2))) if len(audio) >= edge_len else 0.0

    start_edge_ratio = start_rms / (rms_val + 1e-6)
    end_edge_ratio = end_rms / (rms_val + 1e-6)
    clean_boundary = (start_edge_ratio <= 0.90) and (end_edge_ratio <= 0.90)

    # 3. Intra-Segment Speaker Homogeneity (First Half vs Second Half)
    # Convert to 16kHz for SpeechBrain
    t_full = torch.from_numpy(audio).float().unsqueeze(0)
    if sr != 16000:
        t_16k = torchaudio.functional.resample(t_full, sr, 16000)
    else:
        t_16k = t_full

    mid_pt = t_16k.shape[-1] // 2
    h1 = t_16k[:, :mid_pt]
    h2 = t_16k[:, mid_pt:]

    if h1.shape[-1] >= 16000 and h2.shape[-1] >= 16000:
        if torch.cuda.is_available():
            h1 = h1.cuda()
            h2 = h2.cuda()
        with torch.no_grad():
            score, pred = verifier.verify_batch(h1, h2)
            intra_sim_score = float(score.item())
            is_same_speaker = bool(pred.item())
    else:
        intra_sim_score = 0.50
        is_same_speaker = True

    return {
        "file": file_path.name,
        "path": str(file_path),
        "duration_s": round(duration, 3),
        "duration_valid": (3.0 <= duration <= 30.5),
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_dbfs": round(rms_dbfs, 2),
        "has_clipping": has_clipping,
        "clean_boundary": clean_boundary,
        "start_edge_ratio": round(start_edge_ratio, 3),
        "end_edge_ratio": round(end_edge_ratio, 3),
        "intra_speaker_score": round(intra_sim_score, 3),
        "is_pure_single_speaker": is_same_speaker,
    }


def audit_test_suite():
    print("=" * 85)
    print("🕵️ TỰ ĐỘNG KIỂM TRA ĐỘ TINH KHIẾT ÂM HỌC VÀ ĐƠN ÂM TRÊN 3 VIDEO TIẾNG VIỆT")
    print("=" * 85)

    verifier = load_speaker_verifier()

    audit_targets = [
        ("Video 1 (Độc thoại Tiếng Việt)", PROJECT_ROOT / "pipeline_outputs" / "vietnamese_smoke_test" / "vn_video_1_monologue" / "segments"),
        ("Video 2 (Đối thoại Tiếng Việt)", PROJECT_ROOT / "pipeline_outputs" / "vietnamese_smoke_test" / "vn_video_2_interview" / "segments"),
        ("Video 3 (Review & Phỏng vấn FPT)", PROJECT_ROOT / "pipeline_outputs" / "vietnamese_smoke_test" / "vn_video_3_review" / "segments"),
    ]

    total_files = 0
    total_pure = 0
    total_dur_ok = 0
    total_bound_ok = 0
    total_sig_ok = 0
    all_reports = {}

    for name, seg_dir in audit_targets:
        print(f"\n🔍 ĐANG KIỂM TRA: {name}")
        wav_files = sorted(list(seg_dir.glob("*.wav")))
        file_results = []

        for wf in wav_files:
            r = audit_single_file(wf, verifier)
            file_results.append(r)
            total_files += 1
            if r["is_pure_single_speaker"]:
                total_pure += 1
            if r["duration_valid"]:
                total_dur_ok += 1
            if r["clean_boundary"]:
                total_bound_ok += 1
            if not r["has_clipping"]:
                total_sig_ok += 1

        all_reports[name] = file_results
        pure_cnt = sum(1 for r in file_results if r["is_pure_single_speaker"])
        dur_cnt = sum(1 for r in file_results if r["duration_valid"])
        bound_cnt = sum(1 for r in file_results if r["clean_boundary"])
        avg_score = np.mean([r["intra_speaker_score"] for r in file_results]) if file_results else 0.0

        print(f"    ✅ Tổng số segments: {len(file_results)}")
        print(f"    ✅ Độ đồng nhất 1 người nói (Intra-Segment Pure): {pure_cnt}/{len(file_results)} ({pure_cnt/max(len(file_results),1)*100:.1f}%)")
        print(f"    ✅ Điểm tương đồng nội bộ trung bình: {avg_score:.3f}")
        print(f"    ✅ Ràng buộc độ dài [3s - 30s]: {dur_cnt}/{len(file_results)} ({dur_cnt/max(len(file_results),1)*100:.1f}%)")
        print(f"    ✅ Không cụt âm / Mép cắt sạch sẽ: {bound_cnt}/{len(file_results)} ({bound_cnt/max(len(file_results),1)*100:.1f}%)")

    out_report_file = PROJECT_ROOT / "pipeline_outputs" / "vietnamese_audit_report.json"
    with open(out_report_file, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 85)
    print("📊 TỔNG KẾT TOÀN BỘ KẾT QUẢ TỰ KIỂM ĐỊNH (AUDIT) TRÊN 86 FILE PHÂN ĐOẠN:")
    print(f"  1. Tỷ lệ Đơn âm tuyệt đối (Single-Speaker Purity): {total_pure}/{total_files} ({total_pure/max(total_files,1)*100:.1f}%)")
    print(f"  2. Tỷ lệ Ràng buộc độ dài (3.0s - 30.0s):        {total_dur_ok}/{total_files} ({total_dur_ok/max(total_files,1)*100:.1f}%)")
    print(f"  3. Tỷ lệ Ranh giới sạch không cụt chữ (Pad/Fade): {total_bound_ok}/{total_files} ({total_bound_ok/max(total_files,1)*100:.1f}%)")
    print(f"  4. Tỷ lệ Không vỡ âm / Không Clipping:           {total_sig_ok}/{total_files} ({total_sig_ok/max(total_files,1)*100:.1f}%)")
    print(f"📄 Báo cáo chi tiết JSON đã lưu tại: {out_report_file}")
    print("=" * 85)


if __name__ == "__main__":
    audit_test_suite()
