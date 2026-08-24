"""Faster-Whisper SOTA Word Alignment & ASR Engine.

Stage 5 in SOTA Audio Processing Pipeline:
Extracts exact millisecond timestamps (word.start_s, word.end_s, word.probability)
for every spoken word using Faster-Whisper + Silero VAD, preventing word clipping at segment boundaries.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch

from src.alignment.base import (
    AlignedSegment,
    AlignedWord,
    format_srt_timestamp,
    format_vtt_timestamp,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache" / "faster_whisper"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["HF_XET_CACHE"] = str(HF_CACHE_DIR / "xet")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class WordAlignmentEngine:
    """SOTA Word-Level Forced Alignment and ASR Engine via Faster-Whisper."""

    SUPPORTED_MODELS = {
        "large-v3": "Mô hình Large-v3 SOTA (Chính xác cao nhất thế giới cho Tiếng Việt & Đa ngôn ngữ)",
        "large-v2": "Mô hình Large-v2 SOTA (Cân bằng xuất sắc chất lượng và độ ổn định)",
        "medium": "Mô hình Medium (Tốc độ cao, phù hợp GPU tầm trung)",
        "small": "Mô hình Small (Rất nhanh, nhẹ)",
        "base": "Mô hình Base (Siêu nhanh, kiểm tra nhanh)",
    }

    def __init__(self, device: str = "cuda", compute_type: Optional[str] = None):
        self.device = "cuda" if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"
        
        # Determine best compute type for GPU/CPU
        if compute_type:
            self.compute_type = compute_type
        else:
            if self.device == "cuda":
                # Check for float16 support
                self.compute_type = "float16" if torch.cuda.is_bf16_supported() or torch.cuda.is_available() else "int8_float16"
            else:
                self.compute_type = "int8"

        self._models: Dict[str, Any] = {}

    def get_models_status(self) -> dict:
        """Report available models and GPU configuration."""
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        return {
            "device": self.device,
            "gpu_name": gpu_name,
            "compute_type": self.compute_type,
            "models": self.SUPPORTED_MODELS,
            "default_model": "large-v3",
        }

    def _get_model(self, model_size: str = "large-v3", progress_callback: Optional[callable] = None):
        """Lazy load or reuse Faster-Whisper model with live GPU loading progress."""
        import threading

        model_size = model_size.lower()
        if model_size not in self.SUPPORTED_MODELS:
            model_size = "large-v3"

        if model_size in self._models:
            return self._models[model_size]

        from faster_whisper import WhisperModel

        project_root = Path(__file__).resolve().parents[2]
        local_cache_dir = project_root / ".cache" / "faster_whisper"
        local_cache_dir.mkdir(parents=True, exist_ok=True)

        EXPECTED_SIZES = {
            "large-v3": 3087 * 1024 * 1024,
            "large-v2": 3087 * 1024 * 1024,
            "medium": 1530 * 1024 * 1024,
            "small": 484 * 1024 * 1024,
            "base": 145 * 1024 * 1024,
        }
        total_expected_bytes = EXPECTED_SIZES.get(model_size, 3087 * 1024 * 1024)

        if progress_callback:
            progress_callback({
                "status": "loading_gpu",
                "progress_percent": 1.0,
                "model_size": model_size,
                "message": f"Đang khởi động nạp mô hình Faster-Whisper '{model_size}'...",
            })

        logger.info(
            "Loading Faster-Whisper model '%s' on %s (%s)...",
            model_size,
            self.device,
            self.compute_type,
        )

        stop_monitor = threading.Event()

        def _monitor_loop():
            t_mon_start = time.time()
            scan_dirs = [local_cache_dir, HF_CACHE_DIR, Path.home() / ".cache" / "huggingface"]
            while not stop_monitor.is_set():
                current_bytes = 0
                for s_dir in scan_dirs:
                    if not s_dir.exists():
                        continue
                    # Scan for .incomplete files
                    for p in s_dir.rglob("*.incomplete"):
                        try:
                            sz = p.stat().st_size
                            if sz > current_bytes:
                                current_bytes = sz
                        except Exception:
                            pass

                    # Scan for active blobs
                    for p in s_dir.rglob("blobs/*"):
                        try:
                            sz = p.stat().st_size
                            if sz > current_bytes and sz < total_expected_bytes:
                                current_bytes = sz
                        except Exception:
                            pass

                if current_bytes > 0 and progress_callback:
                    pct = min(99.0, max(1.0, (current_bytes / total_expected_bytes) * 100.0))
                    elapsed = max(time.time() - t_mon_start, 0.1)
                    speed_mb = (current_bytes / elapsed) / (1024 * 1024)
                    rem_bytes = max(0, total_expected_bytes - current_bytes)
                    eta_s = round(rem_bytes / max(current_bytes / elapsed, 1024), 1)
                    dl_mb = current_bytes / (1024 * 1024)
                    tot_mb = total_expected_bytes / (1024 * 1024)

                    progress_callback({
                        "status": "downloading_model",
                        "progress_percent": round(pct, 1),
                        "downloaded_mb": round(dl_mb, 1),
                        "total_mb": round(tot_mb, 1),
                        "speed_mb_s": round(speed_mb, 2),
                        "eta_s": eta_s,
                        "file_name": "model.bin",
                        "model_size": model_size,
                        "message": f"Đang tải trọng số {model_size}: {dl_mb:.1f} MB / {tot_mb:.1f} MB ({pct:.1f}%) • {speed_mb:.2f} MB/s • Còn ~{eta_s:.0f}s",
                    })

                stop_monitor.wait(0.5)

        monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        monitor_thread.start()

        # Setup LD_LIBRARY_PATH for NVIDIA CUDA runtime libs if bundled
        import os
        import sys
        for p in sys.path:
            nvidia_dir = Path(p) / "nvidia"
            if nvidia_dir.is_dir():
                for sub in nvidia_dir.iterdir():
                    lib_dir = sub / "lib"
                    if lib_dir.is_dir():
                        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
                        if str(lib_dir) not in cur_ld:
                            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{cur_ld}" if cur_ld else str(lib_dir)

        model_local_dir = local_cache_dir / model_size
        model_target = str(model_local_dir) if (model_local_dir / "model.bin").is_file() else model_size

        device_index = 0 if self.device == "cuda" else 0
        try:
            try:
                model = WhisperModel(
                    model_size_or_path=model_target,
                    device=self.device,
                    device_index=device_index,
                    compute_type=self.compute_type,
                    download_root=str(local_cache_dir),
                )
            except Exception as cuda_exc:
                if self.device == "cuda":
                    logger.warning(
                        "Không thể nạp WhisperModel trên GPU CUDA (%s). Tự động chuyển sang CPU int8 (12 threads)...",
                        cuda_exc
                    )
                    model = WhisperModel(
                        model_size_or_path=model_target,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=min(12, os.cpu_count() or 4),
                        download_root=str(local_cache_dir),
                    )
                else:
                    raise
        finally:
            stop_monitor.set()

        if progress_callback:
            progress_callback({
                "status": "model_ready",
                "progress_percent": 5.0,
                "model_size": model_size,
                "message": f"Mô hình '{model_size}' đã sẵn sàng!",
            })

        self._models[model_size] = model
        return model

    def align(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        model_size: str = "large-v3",
        vad_filter: bool = True,
        beam_size: int = 5,
        word_timestamps: bool = True,
        initial_prompt: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[AlignedSegment], List[AlignedWord], str, str, str, str, float]:
        """Perform ASR and Word-Level Alignment.

        Returns:
            (segments, all_words, full_transcript, srt_content, vtt_content, detected_lang, lang_prob)
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file không tồn tại: {audio_path}")

        if progress_callback:
            progress_callback({
                "status": "loading_model",
                "progress_percent": 2.0,
                "message": f"Đang nạp mô hình Faster-Whisper '{model_size}'...",
            })

        model = self._get_model(model_size=model_size, progress_callback=progress_callback)

        # Configure VAD parameters
        vad_params = dict(
            min_silence_duration_ms=400,
            speech_pad_ms=200,
        ) if vad_filter else None

        # Language configuration
        lang_code = language.strip() if language and language.strip() not in ("auto", "") else None

        logger.info(
            "Starting Word Alignment on %s (model: %s, language: %s, vad: %s)...",
            audio_path.name,
            model_size,
            lang_code or "auto",
            vad_filter,
        )

        if progress_callback:
            progress_callback({
                "status": "transcribing",
                "progress_percent": 5.0,
                "message": "Bắt đầu nhận diện và gióng hàng timestamp từng từ...",
            })

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=lang_code,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
                initial_prompt=initial_prompt,
            )
            # Test generator
            segments_list = []
            for s in segments_iter:
                segments_list.append(s)
        except Exception as trans_exc:
            logger.warning("Lỗi chạy Faster-Whisper trên GPU (%s). Chuyển sang CPU int8...", trans_exc)
            from faster_whisper import WhisperModel
            model = WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=min(12, os.cpu_count() or 4),
                download_root=str(local_cache_dir),
            )
            self._models[model_size] = model
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=lang_code,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
                initial_prompt=initial_prompt,
            )
            segments_list = list(segments_iter)

        detected_lang = info.language
        lang_prob = info.language_probability
        total_audio_duration = max(float(getattr(info, "duration", 1.0)), 1.0)

        aligned_segments: List[AlignedSegment] = []
        all_aligned_words: List[AlignedWord] = []
        full_transcript_parts: List[str] = []
        srt_lines: List[str] = []
        vtt_lines: List[str] = ["WEBVTT", ""]

        t_infer_start = time.time()

        for seg_idx, segment in enumerate(segments_list, 1):
            seg_text = segment.text.strip()
            if not seg_text:
                continue

            full_transcript_parts.append(seg_text)
            seg_words: List[AlignedWord] = []

            if segment.words:
                for w in segment.words:
                    word_clean = w.word.strip()
                    if not word_clean:
                        continue
                    w_obj = AlignedWord(
                        word=word_clean,
                        start_s=float(w.start),
                        end_s=float(w.end),
                        probability=float(w.probability if hasattr(w, "probability") and w.probability is not None else 1.0),
                    )
                    seg_words.append(w_obj)
                    all_aligned_words.append(w_obj)
            else:
                # If word timestamps were not extracted, generate a fallback word from the segment
                w_obj = AlignedWord(
                    word=seg_text,
                    start_s=float(segment.start),
                    end_s=float(segment.end),
                    probability=1.0,
                )
                seg_words.append(w_obj)
                all_aligned_words.append(w_obj)

            aligned_seg = AlignedSegment(
                id=seg_idx,
                text=seg_text,
                start_s=float(segment.start),
                end_s=float(segment.end),
                words=seg_words,
                avg_logprob=float(getattr(segment, "avg_logprob", 0.0)),
                no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0)),
            )
            aligned_segments.append(aligned_seg)

            # Build SRT Block
            srt_start = format_srt_timestamp(segment.start)
            srt_end = format_srt_timestamp(segment.end)
            srt_lines.append(f"{seg_idx}\n{srt_start} --> {srt_end}\n{seg_text}\n")

            # Build VTT Block
            vtt_start = format_vtt_timestamp(segment.start)
            vtt_end = format_vtt_timestamp(segment.end)
            vtt_lines.append(f"{seg_idx}\n{vtt_start} --> {vtt_end}\n{seg_text}\n")

            if progress_callback:
                cur_s = float(segment.end)
                pct = min(98.0, max(5.0, (cur_s / total_audio_duration) * 100))
                infer_elapsed = max(time.time() - t_infer_start, 0.05)
                speed_x = round(cur_s / infer_elapsed, 1)
                remaining_s = max(0.0, total_audio_duration - cur_s)
                eta_s = round(remaining_s / speed_x, 1) if speed_x > 0 else 0.0

                progress_callback({
                    "status": "aligning",
                    "progress_percent": round(pct, 1),
                    "current_time_s": round(cur_s, 1),
                    "total_time_s": round(total_audio_duration, 1),
                    "words_count": len(all_aligned_words),
                    "segments_count": seg_idx,
                    "speed_x": speed_x,
                    "eta_s": eta_s,
                    "latest_snippet": seg_text,
                    "detected_lang": detected_lang,
                    "message": f"Đang gióng hàng: {cur_s:.1f}s / {total_audio_duration:.1f}s ({pct:.0f}%) • {len(all_aligned_words)} từ • {speed_x}x GPU",
                })

        full_transcript = " ".join(full_transcript_parts)
        srt_content = "\n".join(srt_lines)
        vtt_content = "\n".join(vtt_lines)

        logger.info(
            "Word Alignment completed for %s: %d segments, %d words in %.1fs audio (language: %s, prob: %.2f)",
            audio_path.name,
            len(aligned_segments),
            len(all_aligned_words),
            info.duration,
            detected_lang,
            lang_prob,
        )

        return (
            aligned_segments,
            all_aligned_words,
            full_transcript,
            srt_content,
            vtt_content,
            detected_lang,
            lang_prob,
        )
