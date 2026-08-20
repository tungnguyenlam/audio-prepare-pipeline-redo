"""Alignment Manager orchestrating pipeline execution, file artifact exports, and history."""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import soundfile as sf

from src.alignment.aligner import WordAlignmentEngine
from src.alignment.base import AlignedSegment, AlignedWord, AlignmentResult

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_filesize(size_bytes: int) -> str:
    """Format bytes into readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


class AlignmentManager:
    """Orchestrates Word Alignment workflows, exports, and history."""

    def __init__(self, base_aligned_dir: Path, device: str = "cuda"):
        self.base_aligned_dir = Path(base_aligned_dir).resolve()
        self.base_aligned_dir.mkdir(parents=True, exist_ok=True)
        self.engine = WordAlignmentEngine(device=device)
        self.current_progress: dict = {
            "status": "idle",
            "progress_percent": 0.0,
            "current_time_s": 0.0,
            "total_time_s": 0.0,
            "words_count": 0,
            "segments_count": 0,
            "message": "Sẵn sàng",
        }

    def get_progress(self) -> dict:
        """Get the current realtime progress snapshot."""
        return self.current_progress

    def update_progress(self, data: dict):
        """Update progress metrics from engine callback."""
        self.current_progress.update(data)

    def get_models_status(self) -> dict:
        """Report engine and GPU status."""
        return self.engine.get_models_status()

    def get_available_sources(
        self,
        crawl_dir: Path,
        processed_dir: Path,
        diarized_dir: Path,
    ) -> dict:
        """Fetch all eligible audio sources for Alignment from Stages 4, 2, and 1."""
        diarized_sources = []
        if diarized_dir.is_dir():
            for run_dir in diarized_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name.startswith("_"):
                    continue
                meta_file = run_dir / "metadata.json"
                meta = {}
                if meta_file.is_file():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass

                input_filename = meta.get("input_filename", run_dir.name)
                # Find all speaker full master files
                for spk_file in run_dir.glob("SPEAKER_*_full.wav"):
                    spk_id = spk_file.stem.replace("_full", "")
                    # Match speaker percentage from meta if available
                    spk_pct = 0.0
                    for s in meta.get("speakers", []):
                        if s.get("speaker_id") == spk_id:
                            spk_pct = s.get("percentage", 0.0)
                            break

                    info = sf.info(str(spk_file))
                    diarized_sources.append({
                        "type": "diarized",
                        "run_id": run_dir.name,
                        "speaker_id": spk_id,
                        "speaker_label": f"{spk_id} ({spk_pct}%)",
                        "input_filename": input_filename,
                        "audio_path": str(spk_file),
                        "audio_url": f"/api/diarized/{run_dir.name}/{spk_file.name}",
                        "duration_s": round(info.duration, 2),
                        "duration_formatted": format_duration(info.duration),
                        "filesize": spk_file.stat().st_size,
                        "filesize_formatted": format_filesize(spk_file.stat().st_size),
                        "mtime": spk_file.stat().st_mtime,
                    })

        diarized_sources.sort(key=lambda x: x["mtime"], reverse=True)

        processed_sources = []
        if processed_dir.is_dir():
            for model_dir in processed_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith("_"):
                    continue
                model_name = model_dir.name
                for run_dir in model_dir.iterdir():
                    if not run_dir.is_dir() or run_dir.name.startswith("_"):
                        continue
                    vocal_file = run_dir / "vocals.wav"
                    meta_file = run_dir / "metadata.json"
                    if vocal_file.is_file():
                        input_filename = "Unknown"
                        if meta_file.is_file():
                            try:
                                with open(meta_file, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                                input_filename = meta.get("input_filename", "Unknown")
                            except Exception:
                                pass

                        info = sf.info(str(vocal_file))
                        processed_sources.append({
                            "type": "processed",
                            "model": model_name,
                            "run_id": run_dir.name,
                            "input_filename": input_filename,
                            "audio_path": str(vocal_file),
                            "audio_url": f"/api/processed/{model_name}/{run_dir.name}/vocals.wav",
                            "duration_s": round(info.duration, 2),
                            "duration_formatted": format_duration(info.duration),
                            "filesize": vocal_file.stat().st_size,
                            "filesize_formatted": format_filesize(vocal_file.stat().st_size),
                            "mtime": vocal_file.stat().st_mtime,
                        })

        processed_sources.sort(key=lambda x: x["mtime"], reverse=True)

        crawl_sources = []
        if crawl_dir.is_dir():
            for wav_file in crawl_dir.glob("*.wav"):
                if wav_file.name.startswith("_"):
                    continue
                info = sf.info(str(wav_file))
                crawl_sources.append({
                    "type": "crawl",
                    "filename": wav_file.name,
                    "title": wav_file.stem,
                    "audio_path": str(wav_file),
                    "audio_url": f"/audio/{wav_file.name}",
                    "duration_s": round(info.duration, 2),
                    "duration_formatted": format_duration(info.duration),
                    "filesize": wav_file.stat().st_size,
                    "filesize_formatted": format_filesize(wav_file.stat().st_size),
                    "mtime": wav_file.stat().st_mtime,
                })

        crawl_sources.sort(key=lambda x: x["mtime"], reverse=True)

        return {
            "diarized_speakers": diarized_sources,
            "processed_vocals": processed_sources,
            "crawl_audios": crawl_sources,
        }

    def run_alignment(
        self,
        input_audio_path: Path,
        source_type: str = "diarized",
        speaker_id: Optional[str] = None,
        language: Optional[str] = None,
        model_size: str = "large-v3",
        vad_filter: bool = True,
        beam_size: int = 5,
        word_timestamps: bool = True,
        initial_prompt: Optional[str] = None,
    ) -> AlignmentResult:
        """Run speech recognition and word-level alignment, export artifacts and sidecars."""
        input_audio_path = Path(input_audio_path).resolve()
        if not input_audio_path.is_file():
            raise FileNotFoundError(f"File audio đầu vào không tồn tại: {input_audio_path}")

        run_id = uuid.uuid4().hex[:12]
        output_dir = self.base_aligned_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        info = sf.info(str(input_audio_path))
        total_duration = info.duration

        t0 = time.time()
        logger.info(
            "Starting Alignment run %s on %s (duration: %.1fs)...",
            run_id,
            input_audio_path.name,
            total_duration,
        )

        self.update_progress({
            "status": "starting",
            "progress_percent": 1.0,
            "current_time_s": 0.0,
            "total_time_s": round(total_duration, 1),
            "words_count": 0,
            "segments_count": 0,
            "message": f"Bắt đầu gióng hàng âm thanh ({total_duration:.1f}s)...",
        })

        (
            segments,
            words,
            full_transcript,
            srt_content,
            vtt_content,
            detected_lang,
            lang_prob,
        ) = self.engine.align(
            audio_path=input_audio_path,
            language=language,
            model_size=model_size,
            vad_filter=vad_filter,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            progress_callback=self.update_progress,
        )

        elapsed = time.time() - t0

        self.update_progress({
            "status": "completed",
            "progress_percent": 100.0,
            "current_time_s": round(total_duration, 1),
            "total_time_s": round(total_duration, 1),
            "words_count": len(words),
            "segments_count": len(segments),
            "message": f"Hoàn tất gióng hàng {len(words)} từ trong {elapsed:.1f}s!",
        })

        # Calculate Words Per Minute
        duration_minutes = max(total_duration / 60.0, 0.01)
        wpm = round(len(words) / duration_minutes, 1)

        # Copy or link source audio to output_dir
        target_audio_name = f"audio.wav"
        target_audio_path = output_dir / target_audio_name
        try:
            shutil.copyfile(str(input_audio_path), str(target_audio_path))
        except Exception:
            pass

        # Save artifacts: words.json, transcript.txt, subtitles.srt, subtitles.vtt
        words_path = output_dir / "words.json"
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump([w.to_dict() for w in words], f, ensure_ascii=False, indent=2)

        transcript_path = output_dir / "transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)

        srt_path = output_dir / "subtitles.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        vtt_path = output_dir / "subtitles.vtt"
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write(vtt_content)

        # Save metadata.json
        meta_record = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "input_filename": input_audio_path.name,
            "input_path": str(input_audio_path),
            "source_type": source_type,
            "speaker_id": speaker_id,
            "language": detected_lang,
            "language_probability": round(float(lang_prob), 3),
            "model_size": model_size,
            "vad_filter": vad_filter,
            "beam_size": beam_size,
            "total_duration_s": round(float(total_duration), 2),
            "total_duration_formatted": format_duration(total_duration),
            "total_words": len(words),
            "total_segments": len(segments),
            "words_per_minute": wpm,
            "elapsed_seconds": round(elapsed, 2),
            "full_transcript": full_transcript,
            "audio_url": f"/api/aligned/{run_id}/audio.wav",
            "words_url": f"/api/aligned/{run_id}/words.json",
            "transcript_url": f"/api/aligned/{run_id}/transcript.txt",
            "srt_url": f"/api/aligned/{run_id}/subtitles.srt",
            "vtt_url": f"/api/aligned/{run_id}/subtitles.vtt",
            "segments": [s.to_dict() for s in segments],
            "words": [w.to_dict() for w in words],
        }

        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_record, f, ensure_ascii=False, indent=2)

        return AlignmentResult(
            run_id=run_id,
            input_file=input_audio_path,
            source_type=source_type,
            speaker_id=speaker_id,
            language=detected_lang,
            language_probability=lang_prob,
            model_size=model_size,
            total_duration_s=total_duration,
            total_words=len(words),
            total_segments=len(segments),
            words_per_minute=wpm,
            segments=segments,
            words=words,
            full_transcript=full_transcript,
            srt_content=srt_content,
            vtt_content=vtt_content,
            output_dir=output_dir,
            created_at=meta_record["created_at"],
            audio_url=meta_record["audio_url"],
            metadata_file=meta_path,
        )

    def get_history(self) -> List[dict]:
        """Fetch all previous alignment runs sorted by newest first."""
        history = []
        if not self.base_aligned_dir.is_dir():
            return []

        for run_dir in self.base_aligned_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            meta_file = run_dir / "metadata.json"
            if meta_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["mtime"] = run_dir.stat().st_mtime
                    history.append(meta)
                except Exception:
                    pass

        history.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        return history

    def delete_run(self, run_id: str) -> bool:
        """Delete an alignment run directory from disk."""
        run_dir = self.base_aligned_dir / run_id
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
            return True
        return False
