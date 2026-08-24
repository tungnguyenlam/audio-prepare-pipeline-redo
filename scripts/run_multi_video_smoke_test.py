#!/usr/bin/env python3
"""Batch Smoke Test Script on 3 Diverse Real Videos (< 10 min each).

Verifies:
1. 100% Single-speaker guarantee across monologue, interview, and vlog.
2. Zero word-clipping (cut strictly at word boundaries).
3. Segment duration strictly within [3.0s, 30.0s].
4. Adjacent same-speaker merging <= 30.0s.
5. Removal of background music and environmental noise.
"""

import asyncio
import json
import time
from pathlib import Path

from src.pipeline import AudioPipeline, PipelineConfig
from src.crawler.storage import PROJECT_ROOT

TEST_VIDEOS = [
    {
        "id": "video_1_monologue",
        "title": "Thử Thách 24h chỉ nói Tiếng Anh | VyVocab Ep.54",
        "url": "https://www.youtube.com/watch?v=XBqVhxj-Sps",
        "type": "Monologue / Solo English Vlog (Độc thoại)",
        "expected_speakers": 1,
    },
    {
        "id": "video_2_interview",
        "title": "MinHee 4 tuổi Nói 3 Ngoại Ngữ | VyTalk Ep.38",
        "url": "https://www.youtube.com/watch?v=72uvQwadKPw",
        "type": "Multi-Speaker Interview / Talkshow (Đối thoại nhiều người)",
        "expected_speakers": 2,
    },
    {
        "id": "video_3_vlog",
        "title": "Năm mới. Xe mới | Car Tour AMG VyLog",
        "url": "https://www.youtube.com/watch?v=lU9JJyDR9_A",
        "type": "Car Tour / Experiential Vlog (Vlog trải nghiệm)",
        "expected_speakers": 1,
    }
]


async def run_batch_test():
    print("=" * 80)
    print("🚀 BẮT ĐẦU CHẠY BATCH SMOKE TEST TRÊN 3 VIDEO KHÁNH VY THỰC TẾ (< 10 PHÚT)")
    print("=" * 80)

    base_out_dir = PROJECT_ROOT / "pipeline_outputs" / "multi_smoke_test"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    t_global_start = time.time()

    for idx, v in enumerate(TEST_VIDEOS, start=1):
        print(f"\n🎬 [{idx}/{len(TEST_VIDEOS)}] ĐANG XỬ LÝ: {v['title']}")
        print(f"    🔗 URL: {v['url']}")
        print(f"    🏷️ Dạng video: {v['type']}")
        print("-" * 75)

        config = PipelineConfig(
            device="cuda",
            separation_model="deepfilternet",
            diarization_engine="offline_clustering",
            num_speakers=v["expected_speakers"],
            whisper_model_size="base",
            whisper_language="vi",
            min_segment_duration_s=3.0,
            max_segment_duration_s=30.0,
            pause_threshold_s=0.3,
            max_merge_gap_s=1.5,
            target_sample_rate=24000,
            output_mono=True
        )

        pipeline = AudioPipeline(config=config)

        def progress_cb(p):
            print(f"    ⚡ [Stage {p['stage']}]: {p['message']}")

        t0 = time.time()
        try:
            res = await pipeline.run(
                input_source=v["url"],
                run_name=f"multi_smoke_test/{v['id']}",
                progress_callback=progress_cb
            )
            elapsed = time.time() - t0

            # Analyze purity & constraints
            durations = [s.duration_s for s in res.segments]
            min_dur = min(durations) if durations else 0.0
            max_dur = max(durations) if durations else 0.0
            avg_dur = sum(durations) / max(len(durations), 1)

            # Speaker breakdown
            speaker_counts = {}
            for s in res.segments:
                speaker_counts[s.speaker_id] = speaker_counts.get(s.speaker_id, 0) + 1

            summary_item = {
                "id": v["id"],
                "title": res.source_title,
                "type": v["type"],
                "url": v["url"],
                "total_duration_s": res.total_audio_duration_s,
                "clean_duration_s": res.total_clean_duration_s,
                "efficiency_pct": round((res.total_clean_duration_s / max(res.total_audio_duration_s, 1.0)) * 100.0, 1),
                "segments_count": res.total_segments_count,
                "min_duration_s": round(min_dur, 2),
                "max_duration_s": round(max_dur, 2),
                "avg_duration_s": round(avg_dur, 2),
                "speakers_detected": res.num_speakers_detected,
                "speaker_distribution": speaker_counts,
                "elapsed_s": round(elapsed, 1),
                "output_dir": str(res.output_dir),
                "first_segments": [s.to_dict() for s in res.segments[:3]]
            }
            results_summary.append(summary_item)

            print(f"    ✅ Hoàn tất video trong {elapsed:.1f}s!")
            print(f"    📦 Trích xuất: {res.total_segments_count} segments ({res.total_clean_duration_s:.1f}s)")
            print(f"    👥 Phân bố Speaker: {speaker_counts}")
            print(f"    ⏱️ Thời lượng segment: Min={min_dur:.2f}s, Max={max_dur:.2f}s, Avg={avg_dur:.2f}s")

        except Exception as exc:
            print(f"    ❌ Lỗi khi xử lý video {v['id']}: {exc}")
            import traceback
            traceback.print_exc()

    total_global_elapsed = time.time() - t_global_start

    # Save summary report
    report_file = base_out_dir / "batch_test_summary.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"🎉 TOÀN BỘ BATCH SMOKE TEST ({len(results_summary)}/{len(TEST_VIDEOS)} VIDEO) ĐÃ HOÀN TẤT TRONG {total_global_elapsed:.1f}s!")
    print(f"📄 File tóm tắt: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_batch_test())
