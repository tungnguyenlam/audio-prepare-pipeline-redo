#!/usr/bin/env python3
"""Smoke test script for Audio Processing Pipeline on a real Khanh Vy YouTube video (< 10 min)."""

import asyncio
import json
import time
from pathlib import Path

from src.crawler.downloader import crawl_youtube_audio
from src.separation import SeparationManager
from src.diarization import DiarizationManager
from src.crawler.storage import STORAGE_DIR, PROCESSED_DIR, PROJECT_ROOT

VIDEO_URL = "https://www.youtube.com/watch?v=zK3qFnKZFRo"  # 8m12s Khanh Vy VyUni Ep

async def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU SMOKE TEST PIPELINE VỚI VIDEO KHÁNH VY (< 10 PHÚT)")
    print(f"🔗 Video URL: {VIDEO_URL}")
    print("=" * 70)

    # -------------------------------------------------------------
    # GIAI ĐOẠN 1: CRAWL & TRÍCH XUẤT AUDIO (WAV 44.1kHz Stereo)
    # -------------------------------------------------------------
    print("\n📥 [1/3] GIAI ĐOẠN 1: Tải và chuẩn hoá audio từ YouTube...")
    t0 = time.time()
    crawl_record = await crawl_youtube_audio(
        url=VIDEO_URL,
        sample_rate=44100,
        mono=False
    )
    crawl_time = time.time() - t0
    crawled_file = STORAGE_DIR / crawl_record["filename"]
    print(f"  ✅ Đã tải xong trong {crawl_time:.1f}s!")
    print(f"  📂 File: {crawled_file.name}")
    print(f"  ⏱️ Thời lượng: {crawl_record['duration_formatted']} ({crawl_record['duration']}s)")
    print(f"  📦 Dung lượng: {crawl_record['filesize_formatted']}")

    # -------------------------------------------------------------
    # GIAI ĐOẠN 2: SOURCE SEPARATION VỚI DEEPFILTERNET3
    # -------------------------------------------------------------
    print("\n🛡️ [2/3] GIAI ĐOẠN 2: Tách Vocal & Khử nhiễu với DeepFilterNet3...")
    t1 = time.time()
    sep_mgr = SeparationManager(device="cuda")
    run_id = crawl_record["id"]
    output_dir = PROCESSED_DIR / "deepfilternet" / run_id
    
    sep_res = sep_mgr.separators["deepfilternet"].separate(
        input_path=crawled_file,
        output_dir=output_dir,
        atten_lim_db=100.0,
        post_filter=False
    )
    sep_time = time.time() - t1
    vocal_file = sep_res.stems["vocals"]
    print(f"  ✅ Tách vocal hoàn tất trong {sep_time:.1f}s (Real-time factor: {sep_time / crawl_record['duration']:.2f}x)!")
    print(f"  📂 Vocal Stem: {vocal_file}")
    print(f"  📦 Kích thước Vocal: {vocal_file.stat().st_size / (1024 * 1024):.1f} MB")

    # -------------------------------------------------------------
    # GIAI ĐOẠN 4: SPEAKER DIARIZATION & LỌC ĐÈ TIẾNG
    # -------------------------------------------------------------
    print("\n👥 [3/3] GIAI ĐOẠN 4: Phân đoạn người nói & Lọc đè tiếng (Offline ECAPA-TDNN)...")
    t2 = time.time()
    diar_dir = PROJECT_ROOT / "diarized_audio"
    diar_mgr = DiarizationManager(base_diarized_dir=diar_dir, device="cuda")
    
    diar_res = diar_mgr.run_diarization(
        input_audio_path=vocal_file,
        engine="offline_clustering",
        filter_overlap=True,
        min_duration_s=0.5
    )
    diar_time = time.time() - t2
    print(f"  ✅ Diarization hoàn tất trong {diar_time:.1f}s!")
    print(f"  👥 Số người nói phát hiện được: {diar_res.num_speakers}")
    for spk in diar_res.speakers:
        print(f"     - {spk.speaker_id}: Tổng thời lượng {spk.total_time_s:.1f}s ({spk.percentage:.1f}%), {spk.turn_count} lượt nói")
    print(f"  ✂️ Tổng số lượt nói (Turns sau khi lọc overlap & collar trimming): {len(diar_res.turns)}")
    print(f"  📂 Thư mục xuất kết quả: {diar_res.output_dir}")

    total_pipeline_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"🎉 TOÀN BỘ SMOKE TEST THÀNH CÔNG RỰC RỠ TRONG {total_pipeline_time:.1f}s!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
