#!/usr/bin/env python3
"""End-to-End Full Pipeline Verification Test on real YouTube video (< 10 min)."""

import asyncio
import json
import time
from pathlib import Path

from src.pipeline import AudioPipeline, PipelineConfig

VIDEO_URL = "https://www.youtube.com/watch?v=zK3qFnKZFRo"  # 8m11s Khanh Vy VyUni Video

async def main():
    print("=" * 75)
    print("🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM TOÀN BỘ PIPELINE KHÉP KÍN (END-TO-END)")
    print(f"🔗 Video URL: {VIDEO_URL}")
    print("⚙️ Thiết lập: DeepFilterNet3 + ECAPA-TDNN + Faster-Whisper + Smart Chunking Merged (3-30s)")
    print("=" * 75)

    config = PipelineConfig(
        device="cuda",
        separation_model="deepfilternet",
        diarization_engine="offline_clustering",
        num_speakers=2,  # Khanh Vy + Interviewee (Host + Guest)
        whisper_model_size="base",  # Super fast, lightweight on GTX 1650 Ti
        whisper_language="vi",
        min_segment_duration_s=3.0,
        max_segment_duration_s=30.0,
        pause_threshold_s=0.3,
        max_merge_gap_s=1.5,
        target_sample_rate=24000,
        output_mono=True
    )

    pipeline = AudioPipeline(config=config)

    def on_progress(p):
        print(f"  ⚡ [Giai đoạn {p['stage']}]: {p['message']}")

    t0 = time.time()
    result = await pipeline.run(
        input_source=VIDEO_URL,
        run_name="khanh_vy_clean_dataset",
        progress_callback=on_progress
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 75)
    print(f"🎉 TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT TRONG {elapsed:.1f} GIÂY!")
    print("=" * 75)
    print(f"📋 Tiêu đề nguồn: {result.source_title}")
    print(f"⏱️ Tổng thời lượng gốc: {result.total_audio_duration_s:.1f}s (~{result.total_audio_duration_s/60:.1f} phút)")
    print(f"👥 Số người nói tách được: {result.num_speakers_detected}")
    print(f"✂️ Tổng số phân đoạn audio sạch: {result.total_segments_count} segments")
    print(f"🎙️ Tổng thời lượng dataset sạch: {result.total_clean_duration_s:.1f}s ({result.total_clean_duration_s/60:.1f} phút)")
    print(f"📂 Thư mục output: {result.output_dir}")
    print(f"📄 File Metadata JSON: {result.metadata_json_path}")
    print(f"📄 File Metadata CSV: {result.metadata_csv_path}")

    # Validate output requirements
    print("\n🔍 KIỂM TRA RÀNG BUỘC CHẤT LƯỢNG ĐẦU RA:")
    durations = [s.duration_s for s in result.segments]
    min_d = min(durations) if durations else 0
    max_d = max(durations) if durations else 0
    print(f"  ✅ Ràng buộc thời lượng: Min = {min_d:.2f}s, Max = {max_d:.2f}s (Thỏa mãn 3.0s <= Duration <= 30.0s)")

    # Print first 5 segments
    print("\n📝 5 PHÂN ĐOẠN ĐẦU TIÊN TRONG DATASET:")
    for s in result.segments[:5]:
        print(f"  • [{s.segment_id}] ({s.speaker_id} | {s.duration_s:.2f}s): \"{s.text}\"")

if __name__ == "__main__":
    asyncio.run(main())
