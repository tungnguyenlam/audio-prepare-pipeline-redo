"""End-to-End Audio Ingestion & Clean Segmentation Pipeline Orchestrator.

Fulfills all requirements from audio-processing-pipeline-specification.md:
1. Ingestion: YouTube link or local audio file -> 44.1kHz Stereo PCM 16-bit WAV
2. Source Separation: DeepFilterNet3 / HTDemucs -> clean vocal stem without BGM / noise
3. Diarization: SpeechBrain ECAPA-TDNN / PyAnnote -> isolate single speaker turns & filter overlaps
4. Word Alignment: Faster-Whisper -> exact millisecond timestamps per word
5. Smart Chunking & Merging:
   - Preserves word boundaries (strictly cuts at word.end_s, never inside words)
   - Groups segments: 3.0s <= Duration <= 30.0s
   - Automatically merges adjacent segments of the same speaker if combined duration <= 30.0s
6. Export: Clean mono WAV segments (24kHz / 16kHz PCM 16-bit) + metadata.json & metadata.csv
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import soundfile as sf

from src.alignment.aligner import WordAlignmentEngine
from src.alignment.base import AlignedWord
from src.chunking.smart_chunker import AudioSegment, SmartChunker
from src.crawler.downloader import crawl_youtube_audio
from src.crawler.storage import PROCESSED_DIR, PROJECT_ROOT, STORAGE_DIR
from src.diarization.manager import DiarizationManager
from src.separation.manager import SeparationManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline execution configuration parameters."""
    device: str = "cuda"
    separation_model: str = "deepfilternet"
    diarization_engine: str = "offline_clustering"
    num_speakers: Optional[int] = None
    whisper_model_size: str = "medium"  # 'base', 'small', 'medium', 'large-v3'
    whisper_language: str = "vi"  # 'vi', 'en', or 'auto'
    min_segment_duration_s: float = 3.0
    max_segment_duration_s: float = 30.0
    pause_threshold_s: float = 0.5  # Không cắt nếu im lặng < 0.5s
    max_merge_gap_s: float = 1.5
    target_sample_rate: int = 24000  # 24kHz SOTA for TTS / Voice Cloning
    output_mono: bool = True


@dataclass
class PipelineResult:
    """End-to-End Pipeline Execution Output Result."""
    run_id: str
    source_title: str
    source_url_or_path: str
    total_audio_duration_s: float
    elapsed_time_s: float
    num_speakers_detected: int
    total_segments_count: int
    total_clean_duration_s: float
    output_dir: Path
    segments: List[AudioSegment]
    metadata_json_path: Path
    metadata_csv_path: Path


class AudioPipeline:
    """Master Pipeline class executing all 7 stages seamlessly on local machine."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.output_base_dir = PROJECT_ROOT / "pipeline_outputs"
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.separation_mgr = SeparationManager(device=self.config.device)
        self.diarization_mgr = DiarizationManager(
            base_diarized_dir=PROJECT_ROOT / "diarized_audio",
            device=self.config.device
        )
        self.alignment_engine = WordAlignmentEngine(device=self.config.device)
        self.smart_chunker = SmartChunker(
            min_duration_s=self.config.min_segment_duration_s,
            max_duration_s=self.config.max_segment_duration_s,
            pause_threshold_s=self.config.pause_threshold_s,
            max_merge_gap_s=self.config.max_merge_gap_s,
        )

    async def run(
        self,
        input_source: Union[str, Path],
        run_name: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> PipelineResult:
        """Run the complete pipeline from Raw Input -> Final Clean Dataset."""
        t_start = time.time()
        run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        run_dir = self.output_base_dir / (run_name or f"run_{run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)

        input_str = str(input_source).strip()
        is_youtube = any(d in input_str.lower() for d in ["youtube.com", "youtu.be"])

        # =====================================================================
        # STAGE 1: INGESTION (Crawl / Local File Normalization)
        # =====================================================================
        if progress_callback:
            progress_callback({"stage": 1, "message": "Nạp và chuẩn hoá âm thanh đầu vào..."})

        if is_youtube:
            logger.info("Stage 1: Crawling audio from YouTube: %s", input_str)
            crawl_record = await crawl_youtube_audio(
                url=input_str,
                sample_rate=44100,
                mono=False
            )
            raw_audio_path = STORAGE_DIR / crawl_record["filename"]
            title = crawl_record.get("title", "YouTube Audio")
            total_duration = float(crawl_record.get("duration", 0.0))
        else:
            raw_audio_path = Path(input_source).resolve()
            if not raw_audio_path.is_file():
                raise FileNotFoundError(f"Input file not found: {raw_audio_path}")
            title = raw_audio_path.stem
            info = sf.info(str(raw_audio_path))
            total_duration = info.duration

        # =====================================================================
        # STAGE 2: SOURCE SEPARATION & BGM/NOISE REMOVAL
        # =====================================================================
        if progress_callback:
            progress_callback({"stage": 2, "message": f"Tách Vocal & Khử nhiễu nền ({self.config.separation_model})..."})

        logger.info("Stage 2: Running Source Separation with %s on %s...", self.config.separation_model, raw_audio_path.name)
        sep_output_dir = PROCESSED_DIR / self.config.separation_model / run_id
        sep_res = self.separation_mgr.separators[self.config.separation_model].separate(
            input_path=raw_audio_path,
            output_dir=sep_output_dir,
            atten_lim_db=100.0,
        )
        raw_vocal_path = sep_res.stems["vocals"]

        # Deep speech enhancement pass (Post-Filter) to eliminate residual hiss, room reverb & sound effects
        try:
            from src.denoise.deepfilter import DeepFilterEnhancer
            enhancer = DeepFilterEnhancer(post_filter=True)
            enhanced_vocal_path = sep_output_dir / "vocals_enhanced.wav"
            enhancer.enhance(input_audio=raw_vocal_path, output_path=enhanced_vocal_path)
            vocal_path = enhanced_vocal_path
            logger.info("Stage 2: Applied DeepFilterNet3 speech enhancement on %s", vocal_path.name)
        except Exception as enh_exc:
            logger.warning("DeepFilterEnhancer pass failed (%s), using raw separated vocal", enh_exc)
            vocal_path = raw_vocal_path

        # =====================================================================
        # STAGE 4: SPEAKER DIARIZATION & OVERLAP FILTERING
        # =====================================================================
        if progress_callback:
            progress_callback({"stage": 4, "message": "Phân đoạn người nói & lọc đè tiếng..."})

        logger.info("Stage 4: Running Diarization (%s) on %s...", self.config.diarization_engine, vocal_path.name)
        diar_res = self.diarization_mgr.run_diarization(
            input_audio_path=vocal_path,
            engine=self.config.diarization_engine,
            num_speakers=self.config.num_speakers,
            filter_overlap=True,
            min_duration_s=0.5,
        )
        speaker_turns = diar_res.turns

        # =====================================================================
        # STAGE 5: WORD-LEVEL ALIGNMENT & ASR (Faster-Whisper)
        # =====================================================================
        if progress_callback:
            progress_callback({"stage": 5, "message": f"Gióng hàng từ vựng ASR (Faster-Whisper {self.config.whisper_model_size})..."})

        logger.info("Stage 5: Aligning words on %s using Faster-Whisper %s...", vocal_path.name, self.config.whisper_model_size)
        _, all_words, full_transcript, _, _, detected_lang, _ = self.alignment_engine.align(
            audio_path=vocal_path,
            language=self.config.whisper_language,
            model_size=self.config.whisper_model_size,
            vad_filter=True,
            word_timestamps=True,
        )

        # =====================================================================
        # STAGE 6: STRICT SINGLE-SPEAKER SMART CHUNKING & ADJACENT MERGING
        # =====================================================================
        if progress_callback:
            progress_callback({"stage": 6, "message": "Gom phân đoạn đơn âm (3s-30s) & Ghép câu cùng speaker..."})

        logger.info("Stage 6: Building strictly single-speaker segments without word clipping...")
        all_candidate_segments = self.smart_chunker.build_single_speaker_segments(
            turns=speaker_turns,
            all_words=all_words,
            total_audio_duration=total_duration,
            prefix="seg"
        )

        # =====================================================================
        # STAGE 7: SLICE & EXPORT FINAL CLEAN DATASET
        # =====================================================================
        if progress_callback:
            progress_callback({"stage": 7, "message": "Cắt xuất các file audio phân đoạn & tạo metadata..."})

        logger.info("Stage 7: Slicing %d clean segments into %s...", len(all_candidate_segments), run_dir)
        exported_segments = self.smart_chunker.export_audio_segments(
            source_audio_path=vocal_path,
            segments=all_candidate_segments,
            output_dir=run_dir,
            target_sr=self.config.target_sample_rate,
            mono=self.config.output_mono,
        )

        total_clean_duration = sum(s.duration_s for s in exported_segments)
        elapsed_total = time.time() - t_start

        # Summary manifest
        summary_manifest = {
            "run_id": run_id,
            "source_title": title,
            "source_url_or_path": input_str,
            "total_audio_duration_s": round(total_duration, 2),
            "total_clean_duration_s": round(total_clean_duration, 2),
            "efficiency_ratio": round((total_clean_duration / max(total_duration, 1.0)) * 100.0, 1),
            "elapsed_time_s": round(elapsed_total, 2),
            "num_speakers_detected": diar_res.num_speakers,
            "total_segments_count": len(exported_segments),
            "config": {
                "separation_model": self.config.separation_model,
                "diarization_engine": self.config.diarization_engine,
                "whisper_model": self.config.whisper_model_size,
                "target_sample_rate": self.config.target_sample_rate,
                "min_duration_s": self.config.min_segment_duration_s,
                "max_duration_s": self.config.max_segment_duration_s,
            }
        }

        with open(run_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_manifest, f, ensure_ascii=False, indent=2)

        return PipelineResult(
            run_id=run_id,
            source_title=title,
            source_url_or_path=input_str,
            total_audio_duration_s=total_duration,
            elapsed_time_s=elapsed_total,
            num_speakers_detected=diar_res.num_speakers,
            total_segments_count=len(exported_segments),
            total_clean_duration_s=total_clean_duration,
            output_dir=run_dir,
            segments=exported_segments,
            metadata_json_path=run_dir / "metadata.json",
            metadata_csv_path=run_dir / "metadata.csv",
        )
