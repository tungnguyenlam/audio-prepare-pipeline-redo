import os
import re
import asyncio
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form, status
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.crawler.storage import (
    init_storage,
    get_all_audio,
    get_audio_record,
    save_audio_record,
    delete_audio_record,
    get_audio_path,
    format_duration,
    format_filesize,
    STORAGE_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT
)
from src.crawler.downloader import crawl_youtube_audio
from src.separation import SeparationManager, SeparationError
from src.benchmark import SeparationBenchmarkRunner, BENCHMARK_RESULTS_DIR
from src.diarization import DiarizationManager
from src.alignment import AlignmentManager

HF_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FASTER_WHISPER_CACHE = PROJECT_ROOT / ".cache" / "faster_whisper"
FASTER_WHISPER_CACHE.mkdir(parents=True, exist_ok=True)

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

app = FastAPI(
    title="Audio Processing Pipeline API",
    description="Backend API for Audio Ingestion, Source Separation, Benchmark, Diarization, Alignment, Chunking, and Denoising",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DIARIZED_DIR = PROJECT_ROOT / "diarized_audio"
DIARIZED_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_DIR = PROJECT_ROOT / "aligned_audio"
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

separation_manager = SeparationManager(device="cuda")
benchmark_runner = SeparationBenchmarkRunner(device="cuda")
diarization_manager = DiarizationManager(base_diarized_dir=DIARIZED_DIR, device="cuda")
alignment_manager = AlignmentManager(base_aligned_dir=ALIGNED_DIR, device="cuda")


class CrawlRequest(BaseModel):
    url: str = Field(..., description="YouTube video, short or playlist item URL")
    sample_rate: int = Field(default=44100, description="Target sample rate (e.g. 16000, 24000, 44100, 48000)")
    mono: bool = Field(default=False, description="Whether to convert audio to mono channel")
    cookies_from_browser: Optional[str] = Field(default=None, description="Browser to extract cookies from (e.g. firefox, chrome)")


class SeparationRequest(BaseModel):
    filename: str = Field(..., min_length=1, description="Filename from audio_crawl")
    models: List[Literal["htdemucs", "mel_roformer", "deepfilternet", "ht_then_mel", "mel_then_ht"]] = Field(..., min_length=1)
    deepfilternet_atten_lim_db: Optional[float] = Field(default=None, description="Noise attenuation limit in dB for DeepFilterNet (e.g. 100 for max, 12-40 for custom)")
    deepfilternet_post_filter: Optional[bool] = Field(default=False, description="Enable DeepFilterNet post-filter")


class BenchmarkRequest(BaseModel):
    filename: str = Field(..., min_length=1, description="Filename from audio_crawl")
    duration_limit: Optional[int] = Field(default=30, description="Duration in seconds (e.g. 15, 30, 60 or 0 for full)")


class DiarizationRequest(BaseModel):
    source_type: Literal["processed", "crawl"] = Field(default="processed", description="Source audio location")
    filename: Optional[str] = Field(default=None, description="Filename from audio_crawl if source_type == 'crawl'")
    processed_model: Optional[str] = Field(default=None, description="Model in processed_audio")
    processed_run_id: Optional[str] = Field(default=None, description="Run ID in processed_audio")
    engine: Literal["pyannote", "offline_clustering"] = Field(default="offline_clustering", description="Diarization engine")
    hf_token: Optional[str] = Field(default=None, description="Hugging Face User Access Token")
    num_speakers: Optional[int] = Field(default=None, description="Exact speaker count if known")
    min_speakers: Optional[int] = Field(default=None, description="Min speaker count")
    max_speakers: Optional[int] = Field(default=None, description="Max speaker count")
    filter_overlap: bool = Field(default=True, description="Filter overlapping speech turns")
    min_duration_s: float = Field(default=0.5, description="Min segment duration in seconds")


class AlignmentRequest(BaseModel):
    source_type: Literal["diarized", "processed", "crawl"] = Field(default="diarized", description="Source audio location")
    diarized_run_id: Optional[str] = Field(default=None, description="Run ID in diarized_audio")
    diarized_speaker_id: Optional[str] = Field(default=None, description="Speaker ID in diarized_audio")
    processed_model: Optional[str] = Field(default=None, description="Model in processed_audio")
    processed_run_id: Optional[str] = Field(default=None, description="Run ID in processed_audio")
    filename: Optional[str] = Field(default=None, description="Filename from audio_crawl if source_type == 'crawl'")
    language: Optional[str] = Field(default="auto", description="Language code or auto")
    model_size: Literal["large-v3", "large-v2", "medium", "small", "base"] = Field(default="large-v3", description="Faster-Whisper model size")
    vad_filter: bool = Field(default=True, description="Enable Silero VAD filtering")
    beam_size: int = Field(default=5, ge=1, le=10, description="Beam search size")
    word_timestamps: bool = Field(default=True, description="Extract exact word timestamps")
    initial_prompt: Optional[str] = Field(default=None, description="Optional prompt/context for transcription")


@app.on_event("startup")
async def on_startup():
    init_storage()


@app.get("/api/health")
async def health_check():
    """Health check and system status."""
    return {
        "status": "healthy",
        "storage_dir": str(STORAGE_DIR),
        "total_files": len(list(STORAGE_DIR.glob("*.wav"))),
    }


@app.get("/api/audio")
async def list_audio():
    """List all crawled audio files and overall library stats."""
    items = await get_all_audio()
    total_duration = sum(item.get("duration", 0) for item in items)
    total_size = sum(item.get("filesize", 0) for item in items)
    
    return {
        "success": True,
        "items": items,
        "stats": {
            "total_files": len(items),
            "total_duration": total_duration,
            "total_duration_formatted": format_duration(total_duration),
            "total_size": total_size,
            "total_size_formatted": format_filesize(total_size)
        }
    }


@app.post("/api/crawl")
async def crawl_audio(payload: CrawlRequest):
    """Crawl a YouTube video, convert to .wav, and save to audio_crawl/."""
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Vui lòng nhập đường dẫn YouTube hợp lệ.")
    
    if not any(domain in url.lower() for domain in ["youtube.com", "youtu.be"]):
        raise HTTPException(status_code=400, detail="Đường dẫn phải là link YouTube (youtube.com hoặc youtu.be).")

    try:
        record = await crawl_youtube_audio(
            url=url,
            sample_rate=payload.sample_rate,
            mono=payload.mono,
            cookies_from_browser=payload.cookies_from_browser
        )
        return {
            "success": True,
            "message": f"Tải và trích xuất thành công: {record['title']}",
            "data": record
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý YouTube audio: {str(e)}")


@app.post("/api/upload")
async def upload_audio_file(file: UploadFile = File(...)):
    """Upload a local audio file (.wav, .mp3, .flac, .m4a) directly into audio_crawl/."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tên tệp không hợp lệ.")

    ext = Path(file.filename).suffix.lower()
    allowed_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma"}
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Định dạng tệp không được hỗ trợ ({ext}).")

    clean_base = re.sub(r"[^\w\-\.]", "_", Path(file.filename).stem)
    target_filename = f"{clean_base}__{uuid.uuid4().hex[:6]}.wav"
    target_path = STORAGE_DIR / target_filename
    temp_upload = STORAGE_DIR / f"temp_{uuid.uuid4().hex[:8]}{ext}"

    try:
        with open(temp_upload, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert to standardized 44.1kHz Stereo PCM 16-bit WAV
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(temp_upload),
            "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
            str(target_path)
        ]
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not target_path.exists():
            raise RuntimeError(f"FFmpeg conversion failed: {res.stderr}")

        # Probe duration
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(target_path)
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True, check=False)
        duration = float(probe_res.stdout.strip()) if probe_res.stdout.strip() else 0.0

        file_size = target_path.stat().st_size
        record = {
            "id": f"upload_{uuid.uuid4().hex[:8]}",
            "title": Path(file.filename).stem,
            "uploader": "Local Upload",
            "duration": duration,
            "duration_formatted": format_duration(duration),
            "filesize": file_size,
            "filesize_formatted": format_filesize(file_size),
            "sample_rate": 44100,
            "channels": "Stereo",
            "channel_count": 2,
            "filename": target_filename,
            "relative_path": target_filename,
            "created_at": datetime.now().isoformat(),
            "source": "upload"
        }

        # Save sidecar json
        sidecar_path = STORAGE_DIR / f"{target_path.stem}.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        await save_audio_record(record)
        return {
            "success": True,
            "message": f"Tải lên và chuyển đổi thành công: {record['title']}",
            "data": record
        }
    except Exception as exc:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Lỗi tải lên file: {str(exc)}")
    finally:
        if temp_upload.exists():
            temp_upload.unlink(missing_ok=True)


# ============================================================================
# STAGE 2: SOURCE SEPARATION API ENDPOINTS
# ============================================================================

@app.get("/api/separation/models")
async def separation_models():
    """Report model readiness without loading model weights into GPU memory."""
    return {
        "success": True,
        "models": separation_manager.get_models_status()
    }


@app.post("/api/separation")
async def run_separation(payload: SeparationRequest):
    """Run selected vocal-separation models for one audio file."""
    input_path = get_audio_path(payload.filename)
    if input_path is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy file audio đầu vào.")

    results, errors = await asyncio.to_thread(
        separation_manager.run_separation,
        input_path=input_path,
        models=payload.models,
        base_output_dir=PROCESSED_DIR,
        deepfilternet_atten_lim_db=payload.deepfilternet_atten_lim_db,
        deepfilternet_post_filter=payload.deepfilternet_post_filter,
    )

    if not results:
        message = errors[0]["message"] if errors else "Không thể chạy source separation."
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "input": input_path.name,
        "results": results,
        "errors": errors
    }


@app.get("/api/separation/history")
async def get_separation_history():
    """List previous source separation runs with stems and metadata."""
    history = await asyncio.to_thread(separation_manager.get_history, PROCESSED_DIR)
    return {
        "success": True,
        "history": history
    }


VALID_SEPARATION_MODELS = {"htdemucs", "mel_roformer", "deepfilternet", "ht_then_mel", "mel_then_ht"}


@app.delete("/api/separation/{model}/{run_id}")
async def delete_separation_run(model: str, run_id: str):
    """Delete a specific separation run from disk."""
    if model not in VALID_SEPARATION_MODELS or not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=400, detail="Tham số không hợp lệ.")

    success = await asyncio.to_thread(
        separation_manager.delete_run,
        model=model,
        run_id=run_id,
        base_output_dir=PROCESSED_DIR
    )
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi tách nguồn để xóa.")
    return {"success": True, "message": f"Đã xóa thành công run {run_id}"}


@app.get("/api/processed/{model}/{run_id}/{relative_path:path}")
@app.head("/api/processed/{model}/{run_id}/{relative_path:path}")
async def stream_processed_audio(model: str, run_id: str, relative_path: str, request: Request):
    """Stream a separated stem audio with HTTP 206 Partial Content (seeking support)."""
    if model not in VALID_SEPARATION_MODELS or not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy output audio.")

    base_dir = (PROCESSED_DIR / model / run_id).resolve()
    output_path = (base_dir / relative_path).resolve()

    if base_dir not in output_path.parents or not output_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy output audio.")

    file_size = output_path.stat().st_size
    range_header = request.headers.get("Range")

    if not range_header:
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    # Parse Range Header
    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

    if start >= file_size or end >= file_size:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    content_length = (end - start) + 1

    def file_iterator(file_path: Path, start_offset: int, length: int, chunk_size: int = 64 * 1024):
        with open(file_path, "rb") as f:
            f.seek(start_offset)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, chunk_size))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "audio/wav",
    }

    return StreamingResponse(
        file_iterator(output_path, start, content_length),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        headers=headers
    )


# ============================================================================
# STAGE 3: SEPARATION BENCHMARK API ENDPOINTS
# ============================================================================

@app.get("/api/benchmark/models")
async def benchmark_models():
    """Report readiness of all benchmark models."""
    return {
        "success": True,
        "models": benchmark_runner.check_status()
    }


class BenchmarkEvaluateRequest(BaseModel):
    input_filename: str = Field(..., min_length=1, description="Filename of original audio in audio_crawl/")
    htdemucs_run_id: Optional[str] = Field(default=None, description="Run ID of HT Demucs in processed_audio/")
    mel_roformer_run_id: Optional[str] = Field(default=None, description="Run ID of Mel-RoFormer in processed_audio/")
    deepfilternet_run_id: Optional[str] = Field(default=None, description="Run ID of DeepFilterNet in processed_audio/")
    ht_then_mel_run_id: Optional[str] = Field(default=None, description="Run ID of Cascade HT->Mel in processed_audio/")
    mel_then_ht_run_id: Optional[str] = Field(default=None, description="Run ID of Cascade Mel->HT in processed_audio/")


@app.get("/api/benchmark/sources")
async def benchmark_sources():
    """List all audio files that have separated vocal stems from Tab 2 ready for evaluation."""
    sources = await asyncio.to_thread(benchmark_runner.get_available_sources)
    return {
        "success": True,
        "sources": sources
    }


@app.post("/api/benchmark/evaluate")
async def evaluate_separation_outputs(payload: BenchmarkEvaluateRequest):
    """Evaluate pre-separated vocal stems from Tab 2 using Speaker Similarity and DNSMOS P.835."""
    try:
        result = await asyncio.to_thread(
            benchmark_runner.evaluate_existing,
            input_filename=payload.input_filename,
            htdemucs_run_id=payload.htdemucs_run_id,
            mel_roformer_run_id=payload.mel_roformer_run_id,
            deepfilternet_run_id=payload.deepfilternet_run_id,
            ht_then_mel_run_id=payload.ht_then_mel_run_id,
            mel_then_ht_run_id=payload.mel_then_ht_run_id,
        )
        return {
            "success": True,
            "data": result
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi chấm điểm benchmark: {str(exc)}")


@app.get("/api/benchmark/history")
async def get_benchmark_history():
    """List previous separation benchmark runs."""
    history = await asyncio.to_thread(benchmark_runner.get_history)
    return {
        "success": True,
        "history": history
    }


@app.delete("/api/benchmark/{benchmark_id}")
async def delete_benchmark(benchmark_id: str):
    """Delete a benchmark run from disk."""
    if not re.fullmatch(r"[0-9a-f]{12}", benchmark_id):
        raise HTTPException(status_code=400, detail="Mã benchmark không hợp lệ.")

    success = await asyncio.to_thread(benchmark_runner.delete_benchmark, benchmark_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi benchmark để xóa.")
    return {"success": True, "message": f"Đã xóa thành công benchmark {benchmark_id}"}


@app.get("/api/benchmark/{benchmark_id}/{relative_path:path}")
@app.head("/api/benchmark/{benchmark_id}/{relative_path:path}")
async def stream_benchmark_audio(benchmark_id: str, relative_path: str, request: Request):
    """Stream benchmark audio (reference or separated vocals) with seeking support."""
    if not re.fullmatch(r"[0-9a-f]{12}", benchmark_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy file audio benchmark.")

    base_dir = (BENCHMARK_RESULTS_DIR / benchmark_id).resolve()
    if relative_path == "reference.wav":
        output_path = (base_dir / "reference_input.wav").resolve()
    else:
        output_path = (base_dir / relative_path).resolve()

    if base_dir not in output_path.parents or not output_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file audio benchmark.")

    file_size = output_path.stat().st_size
    range_header = request.headers.get("Range")

    if not range_header:
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

    if start >= file_size or end >= file_size:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    content_length = (end - start) + 1

    def file_iterator(file_path: Path, start_offset: int, length: int, chunk_size: int = 64 * 1024):
        with open(file_path, "rb") as f:
            f.seek(start_offset)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, chunk_size))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "audio/wav",
    }

    return StreamingResponse(
        file_iterator(output_path, start, content_length),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        headers=headers
    )


# ============================================================================
# STAGE 4: SPEAKER DIARIZATION (src/diarization)
# ============================================================================

@app.get("/api/diarization/engines")
async def get_diarization_engines():
    """Get status of available diarization engines."""
    return diarization_manager.get_engines_status()


@app.get("/api/diarization/sources")
async def list_diarization_sources():
    """List available audio sources (Vocal stems from Stage 2 + Raw Crawled audios)."""
    processed_sources = []
    if PROCESSED_DIR.is_dir():
        for model_dir in PROCESSED_DIR.iterdir():
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
                            import json
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            input_filename = meta.get("input_filename", "Unknown")
                        except Exception:
                            pass
                    
                    processed_sources.append({
                        "type": "processed",
                        "model": model_name,
                        "run_id": run_dir.name,
                        "input_filename": input_filename,
                        "vocal_url": f"/api/processed/{model_name}/{run_dir.name}/vocals.wav",
                        "vocal_path": str(vocal_file),
                        "filesize": vocal_file.stat().st_size,
                        "filesize_formatted": format_filesize(vocal_file.stat().st_size),
                    })

    crawl_items = await get_all_audio()

    return {
        "success": True,
        "processed_vocals": processed_sources,
        "crawl_audios": crawl_items,
    }


@app.post("/api/diarization")
async def run_diarization(payload: DiarizationRequest):
    """Execute speaker diarization on chosen audio."""
    if payload.source_type == "processed":
        if not payload.processed_model or not payload.processed_run_id:
            raise HTTPException(status_code=400, detail="Vui lòng chọn model và run_id của vocal đã tách.")
        vocal_path = PROCESSED_DIR / payload.processed_model / payload.processed_run_id / "vocals.wav"
        if not vocal_path.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy file vocal đã tách tương ứng.")
        input_path = vocal_path
    else:
        if not payload.filename:
            raise HTTPException(status_code=400, detail="Vui lòng chọn file audio từ thư viện crawl.")
        raw_path = get_audio_path(payload.filename)
        if not raw_path or not raw_path.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy file audio crawl.")
        input_path = raw_path

    try:
        import logging
        logger = logging.getLogger(__name__)
        result = await asyncio.to_thread(
            diarization_manager.run_diarization,
            input_audio_path=input_path,
            engine=payload.engine,
            hf_token=payload.hf_token,
            num_speakers=payload.num_speakers,
            min_speakers=payload.min_speakers,
            max_speakers=payload.max_speakers,
            filter_overlap=payload.filter_overlap,
            min_duration_s=payload.min_duration_s,
        )
        return {
            "success": True,
            "data": result.to_dict()
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/diarization/history")
async def get_diarization_history():
    """List all previous diarization runs."""
    history = await asyncio.to_thread(diarization_manager.get_history)
    return {
        "success": True,
        "history": history
    }


@app.delete("/api/diarization/{run_id}")
async def delete_diarization_run(run_id: str):
    """Delete a diarization run directory."""
    if not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=400, detail="Run ID không hợp lệ.")
    success = await asyncio.to_thread(diarization_manager.delete_run, run_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi diarization để xóa.")
    return {"success": True, "message": f"Đã xóa thành công bản ghi diarization {run_id}"}


@app.get("/api/diarized/{run_id}/{relative_path:path}")
@app.head("/api/diarized/{run_id}/{relative_path:path}")
async def stream_diarized_audio(run_id: str, relative_path: str, request: Request):
    """Stream diarized speaker audio clips with HTTP 206 Partial Content support."""
    if not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy audio.")

    base_dir = (DIARIZED_DIR / run_id).resolve()
    target_path = (base_dir / relative_path).resolve()

    if base_dir not in target_path.parents or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file audio diarized.")

    file_size = target_path.stat().st_size
    range_header = request.headers.get("Range")

    if not range_header:
        return FileResponse(
            path=target_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        return FileResponse(
            path=target_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

    if start >= file_size or end >= file_size:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    content_length = (end - start) + 1

    def file_iterator(file_path: Path, start_offset: int, length: int, chunk_size: int = 64 * 1024):
        with open(file_path, "rb") as f:
            f.seek(start_offset)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, chunk_size))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "audio/wav",
    }

    return StreamingResponse(
        file_iterator(target_path, start, content_length),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        headers=headers
    )


# ============================================================================
# STAGE 5: WORD ALIGNMENT & VAD (src/alignment)
# ============================================================================

@app.get("/api/alignment/models")
async def alignment_models():
    """Get status of available Faster-Whisper ASR models and GPU configuration."""
    return alignment_manager.get_models_status()


@app.get("/api/alignment/sources")
async def list_alignment_sources():
    """List available audio sources (Stage 4 Diarized Speaker Voices, Stage 2 Vocal Stems, Stage 1 Crawl Audios)."""
    sources = await asyncio.to_thread(
        alignment_manager.get_available_sources,
        crawl_dir=STORAGE_DIR,
        processed_dir=PROCESSED_DIR,
        diarized_dir=DIARIZED_DIR,
    )
    return {
        "success": True,
        **sources
    }


@app.get("/api/alignment/progress")
async def get_alignment_progress():
    """Report realtime progress of the ongoing word alignment."""
    return {
        "success": True,
        "progress": alignment_manager.get_progress()
    }


@app.post("/api/alignment")
async def run_alignment(payload: AlignmentRequest):
    """Execute speech recognition and word-level alignment on selected audio."""
    if payload.source_type == "diarized":
        if not payload.diarized_run_id or not payload.diarized_speaker_id:
            raise HTTPException(status_code=400, detail="Vui lòng chọn Run ID và Speaker ID của bản Diarization.")
        spk_master = DIARIZED_DIR / payload.diarized_run_id / f"{payload.diarized_speaker_id}_full.wav"
        if not spk_master.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy file audio của Speaker đã chọn.")
        input_path = spk_master
        speaker_id = payload.diarized_speaker_id
    elif payload.source_type == "processed":
        if not payload.processed_model or not payload.processed_run_id:
            raise HTTPException(status_code=400, detail="Vui lòng chọn model và run_id của bản tách Vocal.")
        vocal_path = PROCESSED_DIR / payload.processed_model / payload.processed_run_id / "vocals.wav"
        if not vocal_path.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy file vocal đã tách.")
        input_path = vocal_path
        speaker_id = None
    else:
        if not payload.filename:
            raise HTTPException(status_code=400, detail="Vui lòng chọn file audio từ thư viện crawl.")
        raw_path = get_audio_path(payload.filename)
        if not raw_path or not raw_path.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy file audio crawl.")
        input_path = raw_path
        speaker_id = None

    try:
        result = await asyncio.to_thread(
            alignment_manager.run_alignment,
            input_audio_path=input_path,
            source_type=payload.source_type,
            speaker_id=speaker_id,
            language=payload.language,
            model_size=payload.model_size,
            vad_filter=payload.vad_filter,
            beam_size=payload.beam_size,
            word_timestamps=payload.word_timestamps,
            initial_prompt=payload.initial_prompt,
        )
        return {
            "success": True,
            "data": result.to_dict()
        }
    except Exception as exc:
        logger.error("Alignment run failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Lỗi khi gióng hàng từ (Alignment): {str(exc)}")


@app.get("/api/alignment/history")
async def get_alignment_history():
    """List all previous word alignment runs."""
    history = await asyncio.to_thread(alignment_manager.get_history)
    return {
        "success": True,
        "history": history
    }


@app.delete("/api/alignment/{run_id}")
async def delete_alignment_run(run_id: str):
    """Delete an alignment run from disk."""
    if not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=400, detail="Run ID không hợp lệ.")
    success = await asyncio.to_thread(alignment_manager.delete_run, run_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi alignment để xóa.")
    return {"success": True, "message": f"Đã xóa thành công bản ghi alignment {run_id}"}


@app.get("/api/aligned/{run_id}/{relative_path:path}")
@app.head("/api/aligned/{run_id}/{relative_path:path}")
async def stream_aligned_file(run_id: str, relative_path: str, request: Request):
    """Stream audio or download subtitle/JSON artifacts with HTTP 206 Partial Content support."""
    if not re.fullmatch(r"[0-9a-f]{12}", run_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")

    base_dir = (ALIGNED_DIR / run_id).resolve()
    target_path = (base_dir / relative_path).resolve()

    if base_dir not in target_path.parents or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file tương ứng.")

    # Determine media type based on extension
    ext = target_path.suffix.lower()
    media_type = "application/octet-stream"
    if ext == ".wav":
        media_type = "audio/wav"
    elif ext == ".json":
        media_type = "application/json"
    elif ext == ".srt":
        media_type = "text/plain; charset=utf-8"
    elif ext == ".vtt":
        media_type = "text/vtt; charset=utf-8"
    elif ext == ".txt":
        media_type = "text/plain; charset=utf-8"

    if ext != ".wav":
        return FileResponse(
            path=target_path,
            media_type=media_type,
            filename=target_path.name
        )

    file_size = target_path.stat().st_size
    range_header = request.headers.get("Range")

    if not range_header:
        return FileResponse(
            path=target_path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes"}
        )

    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        return FileResponse(
            path=target_path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes"}
        )

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

    if start >= file_size or end >= file_size:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    content_length = (end - start) + 1

    def file_iterator(file_path: Path, start_offset: int, length: int, chunk_size: int = 64 * 1024):
        with open(file_path, "rb") as f:
            f.seek(start_offset)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, chunk_size))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "audio/wav",
    }

    return StreamingResponse(
        file_iterator(target_path, start, content_length),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        headers=headers
    )


# ============================================================================
# GENERAL AUDIO STREAMING & MANAGEMENT
# ============================================================================

@app.get("/api/audio/{filename}")
@app.head("/api/audio/{filename}")
async def stream_audio(filename: str, request: Request):
    """Stream audio with support for HTTP 206 Partial Content (seeking in audio players)."""
    file_path = get_audio_path(filename)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp audio yêu cầu.")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("Range")

    if not range_header:
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            headers={"Accept-Ranges": "bytes"}
        )

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

    if start >= file_size or end >= file_size:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    content_length = (end - start) + 1

    def file_iterator(file_path: Path, start_offset: int, length: int, chunk_size: int = 64 * 1024):
        with open(file_path, "rb") as f:
            f.seek(start_offset)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, chunk_size))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "audio/wav",
    }

    return StreamingResponse(
        file_iterator(file_path, start, content_length),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        headers=headers
    )


@app.delete("/api/audio/{filename}")
async def delete_audio(filename: str):
    """Delete an audio file and its metadata record from local disk."""
    success = await delete_audio_record(filename)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp hoặc không thể xóa.")
    return {"success": True, "message": f"Đã xóa thành công {filename}"}


# Mount Frontend Static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
