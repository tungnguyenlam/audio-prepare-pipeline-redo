"""Web server application for the audio preparation pipeline.

Provides a full REST API, background task management, audio streaming,
waveform extraction, spectrogram generation, live-reload SSE, and static
file serving for the frontend studio.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import logging
import os
import shutil
import sys
import time
import uuid
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web
import soundfile as sf
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio
from src.utils.AudioCutter import AudioCutter, AudioCutterError
from src.utils.SpectrogramComparer import SpectrogramComparer
from src.utils.WaveformComparer import WaveformComparer
from src.separation import HTDemucs, BSRoFormer, MelRoFormer, MVSepMDX23
from src.diarization import PyannoteDiarizer, SortformerDiarizer
from src.benchmark.separation.mixer import AudioMixer
from src.yt_crawler.YtCrawlerClass import YtCrawler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web_server")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = ROOT_DIR / ".data"
TEMP_DIR = ROOT_DIR / "temp"
UPLOADS_DIR = DATA_DIR / "web_uploads"

# Ensure runtime directories exist
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def get_default_device() -> str:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_system_device_info() -> dict[str, Any]:
    """Return hardware accelerator and environment details."""
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    device_name = "CPU"
    device_type = "cpu"
    
    if cuda_available:
        device_type = "cuda"
        device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
    elif mps_available:
        device_type = "mps"
        device_name = "Apple Silicon (MPS)"
        
    return {
        "device_type": device_type,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }


class AudioRegistry:
    """In-memory store mapping audio IDs to Audio objects and metadata."""

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._waveform_cache: Dict[str, List[float]] = {}

    def register(
        self,
        audio: Audio,
        *,
        source_type: str = "local",
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register an Audio object and return a unique ID."""
        audio_id = f"aud_{uuid.uuid4().hex[:10]}"
        self._items[audio_id] = {
            "id": audio_id,
            "audio": audio,
            "source_type": source_type,
            "parent_id": parent_id,
            "tags": tags or [],
            "created_at": time.time(),
        }
        return audio_id

    def get_audio(self, audio_id: str) -> Optional[Audio]:
        """Retrieve the Audio object for an ID."""
        item = self._items.get(audio_id)
        return item["audio"] if item else None

    def get_item(self, audio_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the full registered item dictionary."""
        return self._items.get(audio_id)

    def list_all(self) -> List[Dict[str, Any]]:
        """List all registered items formatted for the frontend."""
        result = []
        for audio_id, item in sorted(
            self._items.items(), key=lambda x: x[1]["created_at"], reverse=True
        ):
            audio: Audio = item["audio"]
            meta = audio.metadata()
            result.append(
                {
                    "id": audio_id,
                    "source_id": audio.source_id,
                    "title": audio.title,
                    "path": str(audio.path),
                    "sample_rate": audio.sample_rate,
                    "native_sample_rate": audio.native_sample_rate,
                    "duration_s": audio.duration_s,
                    "channels": audio.channels,
                    "format": audio.format,
                    "history": list(audio.history),
                    "fingerprint": audio.fingerprint,
                    "source_type": item["source_type"],
                    "parent_id": item["parent_id"],
                    "tags": item["tags"],
                    "created_at": item["created_at"],
                    "file_size": audio.path.stat().st_size if audio.path.is_file() else 0,
                }
            )
        return result

    def get_cached_waveform(self, audio_id: str) -> Optional[List[float]]:
        return self._waveform_cache.get(audio_id)

    def cache_waveform(self, audio_id: str, peaks: List[float]) -> None:
        self._waveform_cache[audio_id] = peaks


class TaskManager:
    """Manages asynchronous background jobs (Separation, Crawl, Diarization)."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, task_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        self._tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "pending",  # pending, running, completed, failed
            "progress": 0.0,
            "message": "Task queued...",
            "error": None,
            "result": None,
            "start_time": time.time(),
            "end_time": None,
            "metadata": metadata or {},
        }
        return task_id

    def update_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Any] = None,
    ) -> None:
        if task_id in self._tasks:
            t = self._tasks[task_id]
            if status:
                t["status"] = status
            if progress is not None:
                t["progress"] = progress
            if message:
                t["message"] = message
            if error:
                t["error"] = error
            if result is not None:
                t["result"] = result
            if status in ("completed", "failed"):
                t["end_time"] = time.time()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._tasks.get(task_id)


# Global instances
registry = AudioRegistry()
task_manager = TaskManager()
live_reload_subscribers: List[asyncio.Queue] = []


def extract_waveform_peaks(audio_path: Path, num_points: int = 1200) -> List[float]:
    """Extract downsampled peak amplitudes for high-performance waveform drawing."""
    if not audio_path.is_file():
        return []
    try:
        data, _sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        # Convert to mono by averaging channels
        mono = data.mean(axis=1)
        total_samples = len(mono)
        if total_samples == 0:
            return []

        chunk_size = max(1, total_samples // num_points)
        peaks = []
        for i in range(0, total_samples, chunk_size):
            chunk = mono[i : i + chunk_size]
            if len(chunk) > 0:
                peak = float(np.max(np.abs(chunk)))
                peaks.append(round(peak, 4))
        return peaks
    except Exception as e:
        logger.error("Failed to extract waveform: %s", e)
        return []


# ==================== API HANDLERS ====================


async def handle_status(request: web.Request) -> web.Response:
    """Return system information and device status."""
    info = get_system_device_info()
    info["registered_audios"] = len(registry.list_all())
    return web.json_response(info)


async def handle_list_library(request: web.Request) -> web.Response:
    """Scan and list audio files available in project directories."""
    scan_dirs = [
        ("Benchmark Speech", ROOT_DIR / "benchmarks/separation/sources/speech"),
        ("Data Directory", ROOT_DIR / "data"),
        ("Quick Saves (temp)", ROOT_DIR / "temp"),
        ("Runtime Outputs (.data)", ROOT_DIR / ".data"),
    ]

    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = []

    for category, directory in scan_dirs:
        if directory.is_dir():
            for p in directory.rglob("*"):
                if p.is_file() and p.suffix.lower() in extensions:
                    try:
                        stat = p.stat()
                        files.append(
                            {
                                "category": category,
                                "name": p.name,
                                "path": str(p.relative_to(ROOT_DIR)),
                                "absolute_path": str(p.resolve()),
                                "size": stat.st_size,
                                "modified": stat.st_mtime,
                                "format": p.suffix.lstrip(".").lower(),
                            }
                        )
                    except Exception:
                        pass

    files.sort(key=lambda x: x["modified"], reverse=True)
    return web.json_response({"files": files})


async def handle_load_library_file(request: web.Request) -> web.Response:
    """Load a server file into active registry."""
    data = await request.json()
    file_path = data.get("path")
    if not file_path:
        return web.json_response({"error": "Path is required"}, status=400)

    resolved = Path(file_path)
    if not resolved.is_absolute():
        resolved = (ROOT_DIR / file_path).resolve()

    if not resolved.is_file():
        return web.json_response({"error": f"File not found: {resolved}"}, status=404)

    try:
        audio = Audio.from_file(resolved)
        audio_id = registry.register(audio, source_type="library", tags=["library"])
        return web.json_response({"audio_id": audio_id, "metadata": audio.metadata()})
    except Exception as e:
        logger.exception("Error loading audio file")
        return web.json_response({"error": str(e)}, status=500)


async def handle_upload_audio(request: web.Request) -> web.Response:
    """Handle multipart file upload from client."""
    reader = await request.multipart()
    field = await reader.next()
    if not field or field.name != "file":
        return web.json_response({"error": "Form field 'file' expected"}, status=400)

    filename = field.filename or f"upload_{int(time.time())}.wav"
    clean_name = Path(filename).name
    save_path = UPLOADS_DIR / f"{int(time.time())}_{clean_name}"

    with open(save_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)

    try:
        audio = Audio.from_file(save_path)
        audio_id = registry.register(audio, source_type="upload", tags=["upload"])
        return web.json_response({"audio_id": audio_id, "metadata": audio.metadata()})
    except Exception as e:
        logger.exception("Error processing uploaded file")
        return web.json_response({"error": str(e)}, status=500)


async def handle_youtube_inspect(request: web.Request) -> web.Response:
    """Inspect YouTube video metadata without downloading full media."""
    data = await request.json()
    url = data.get("url")
    if not url:
        return web.json_response({"error": "URL is required"}, status=400)

    def do_inspect():
        import subprocess
        crawler = YtCrawler()
        cmd = crawler._yt_dlp_prefix() + [
            "--dump-json",
            "--no-playlist",
            "--no-download",
            url,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            err = (res.stderr or res.stdout or "").strip()
            raise RuntimeError(f"Failed to inspect YouTube URL: {err[:500]}")
        info = json.loads(res.stdout)
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "duration": info.get("duration"),
            "uploader": info.get("uploader") or info.get("channel"),
            "thumbnail": info.get("thumbnail"),
            "view_count": info.get("view_count"),
            "webpage_url": info.get("webpage_url") or url,
            "description": (info.get("description") or "")[:300],
        }

    try:
        loop = asyncio.get_running_loop()
        meta = await loop.run_in_executor(None, do_inspect)
        return web.json_response(meta)
    except Exception as e:
        logger.exception("YouTube inspect failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_youtube_history(request: web.Request) -> web.Response:
    """List previously crawled YouTube audio files from disk."""
    yt_dir = DATA_DIR / "yt_crawler" / "downloads"
    files = []
    if yt_dir.is_dir():
        for p in yt_dir.glob("*.wav"):
            try:
                rate, dur, ch = _probe_wav(p)
                stat = p.stat()
                files.append({
                    "name": p.name,
                    "stem": p.stem,
                    "path": str(p.relative_to(ROOT_DIR)),
                    "absolute_path": str(p.resolve()),
                    "sample_rate": rate,
                    "duration_s": dur,
                    "channels": ch,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
            except Exception:
                pass
    files.sort(key=lambda x: x["modified"], reverse=True)
    return web.json_response({"downloads": files})


async def handle_youtube_ingest(request: web.Request) -> web.Response:
    """Start asynchronous YouTube audio ingestion."""
    data = await request.json()
    url = data.get("url")
    sample_rate = int(data.get("sample_rate", DEFAULT_SAMPLE_RATE))
    audio_format = data.get("audio_format", "wav")

    if not url:
        return web.json_response({"error": "URL is required"}, status=400)

    task_id = task_manager.create_task("youtube_crawl", {"url": url, "sample_rate": sample_rate})

    async def run_crawler():
        task_manager.update_task(task_id, status="running", progress=0.1, message="Downloading YouTube audio with yt-dlp...")
        loop = asyncio.get_running_loop()
        try:
            crawler = YtCrawler(
                output_dir=DATA_DIR / "yt_crawler" / "downloads",
                work_dir=DATA_DIR / "yt_crawler" / "work",
                audio_format=audio_format,
                sample_rate=sample_rate,
                channels=1,
            )
            # Run in worker thread
            audio = await loop.run_in_executor(None, crawler.download, url)
            audio_id = registry.register(audio, source_type="youtube", tags=["youtube", "crawled", f"{sample_rate}Hz"])
            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                message=f"Downloaded '{audio.title}' successfully!",
                result={"audio_id": audio_id, "metadata": audio.metadata()},
            )
        except Exception as e:
            logger.exception("YouTube ingest failed")
            task_manager.update_task(
                task_id,
                status="failed",
                error=str(e),
                message=f"YouTube ingestion error: {e}",
            )

    asyncio.create_task(run_crawler())
    return web.json_response({"task_id": task_id})


async def handle_list_audios(request: web.Request) -> web.Response:
    """Return list of all registered Audio objects."""
    return web.json_response({"audios": registry.list_all()})


async def handle_get_audio_metadata(request: web.Request) -> web.Response:
    """Get metadata for a specific Audio object."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)
    return web.json_response(audio.metadata())


async def handle_stream_audio(request: web.Request) -> web.Response:
    """Stream audio with full HTTP Range support for instant playback & seeking."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio or not audio.path.is_file():
        return web.Response(text="Audio file not found", status=404)

    return web.FileResponse(
        audio.path,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Content-Disposition": f'inline; filename="{audio.path.name}"',
        },
    )


async def handle_get_waveform(request: web.Request) -> web.Response:
    """Return waveform peaks data."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio or not audio.path.is_file():
        return web.json_response({"error": "Audio file not found"}, status=404)

    cached = registry.get_cached_waveform(audio_id)
    if cached is not None:
        return web.json_response({"peaks": cached, "duration_s": audio.duration_s})

    loop = asyncio.get_running_loop()
    peaks = await loop.run_in_executor(None, extract_waveform_peaks, audio.path)
    registry.cache_waveform(audio_id, peaks)
    return web.json_response({"peaks": peaks, "duration_s": audio.duration_s})


async def handle_get_spectrogram(request: web.Request) -> web.Response:
    """Generate and return a Mel Spectrogram image (PNG)."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio or not audio.path.is_file():
        return web.Response(text="Audio file not found", status=404)

    def generate_spec_png():
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt

        sr = 16000
        y, _ = librosa.load(str(audio.path), sr=sr, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#141721")
        ax.set_facecolor("#0e111a")
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
        )
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
        cbar.set_label("dB", color="#94a3b8")

        ax.set_title(f"Mel Spectrogram: {audio.title}", color="#e2e8f0", fontsize=11)
        ax.set_xlabel("Time (s)", color="#94a3b8")
        ax.set_ylabel("Hz", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    loop = asyncio.get_running_loop()
    png_bytes = await loop.run_in_executor(None, generate_spec_png)
    return web.Response(body=png_bytes, content_type="image/png")


async def handle_cut_audio(request: web.Request) -> web.Response:
    """Cut audio segment using AudioCutter."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    data = await request.json()
    start = data.get("start")
    end = data.get("end")
    unit = data.get("unit", "seconds")

    if start is None or end is None:
        return web.json_response({"error": "start and end bounds are required"}, status=400)

    out_dir = DATA_DIR / "audio_cutter" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cutter = AudioCutter(output_dir=out_dir)

    try:
        loop = asyncio.get_running_loop()
        cut_audio = await loop.run_in_executor(
            None,
            lambda: cutter.cut(audio, start, end, unit=unit),
        )
        new_id = registry.register(
            cut_audio,
            source_type="cut",
            parent_id=audio_id,
            tags=["cut", f"{unit}:{start}-{end}"],
        )
        return web.json_response({
            "audio_id": new_id,
            "metadata": cut_audio.metadata(),
        })
    except AudioCutterError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Audio cut failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_quick_save(request: web.Request) -> web.Response:
    """Perform Audio.quick_save() to temp/ or specified directory."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    data = await request.json() if request.can_read_body else {}
    name = data.get("name")
    prefix = data.get("prefix")
    suffix = data.get("suffix")
    tag = data.get("tag")

    try:
        loop = asyncio.get_running_loop()
        saved = await loop.run_in_executor(
            None,
            lambda: audio.quick_save(
                TEMP_DIR,
                name=name,
                prefix=prefix,
                suffix=suffix,
                tag=tag,
            ),
        )
        return web.json_response({
            "saved_path": str(saved.path),
            "fingerprint": saved.fingerprint,
            "metadata": saved.metadata(),
        })
    except Exception as e:
        logger.exception("Quick save failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_save_to(request: web.Request) -> web.Response:
    """Save audio to a specified destination path."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    data = await request.json()
    dest = data.get("dest")
    if not dest:
        return web.json_response({"error": "Destination path required"}, status=400)

    dest_path = Path(dest)
    if not dest_path.is_absolute():
        dest_path = (ROOT_DIR / dest).resolve()

    try:
        loop = asyncio.get_running_loop()
        saved = await loop.run_in_executor(None, audio.save_to, dest_path)
        return web.json_response({
            "saved_path": str(saved.path),
            "metadata": saved.metadata(),
        })
    except Exception as e:
        logger.exception("Save to failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_run_separation(request: web.Request) -> web.Response:
    """Run model source separation in background."""
    data = await request.json()
    audio_id = data.get("audio_id")
    model_type = data.get("model_type", "htdemucs").lower()  # htdemucs, bs_roformer, mel_roformer, mvsep_mdx23
    device = data.get("device", "auto")
    model_name = data.get("model_name")
    two_stems = data.get("two_stems", "vocals")

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    task_id = task_manager.create_task(
        "separation",
        {"audio_id": audio_id, "model_type": model_type, "device": device},
    )

    async def run_sep():
        task_manager.update_task(
            task_id,
            status="running",
            progress=0.1,
            message=f"Initializing {model_type.upper()} on {device}...",
        )
        loop = asyncio.get_running_loop()

        def do_separation():
            target_device = get_default_device() if device == "auto" else device
            
            if model_type == "htdemucs":
                sep = HTDemucs(
                    model=model_name or "htdemucs",
                    device=target_device,
                    two_stems=two_stems,
                    output_dir=DATA_DIR / "demucs" / "out",
                    work_dir=DATA_DIR / "demucs" / "work",
                )
                return sep.separate(audio)
            
            elif model_type == "bs_roformer":
                kwargs = {
                    "device": target_device,
                    "output_dir": DATA_DIR / "bs_roformer" / "out",
                    "work_dir": DATA_DIR / "bs_roformer" / "work",
                }
                if model_name:
                    kwargs["model"] = model_name
                sep = BSRoFormer(**kwargs)
                with sep:
                    return sep.separate(audio)
            
            elif model_type == "mel_roformer":
                kwargs = {
                    "device": target_device,
                    "output_dir": DATA_DIR / "mel_roformer" / "out",
                    "work_dir": DATA_DIR / "mel_roformer" / "work",
                }
                if model_name:
                    kwargs["model"] = model_name
                sep = MelRoFormer(**kwargs)
                with sep:
                    return sep.separate(audio)
            
            elif model_type == "mvsep_mdx23":
                sep = MVSepMDX23(
                    device=target_device,
                    output_dir=DATA_DIR / "mvsep_mdx23" / "out",
                    work_dir=DATA_DIR / "mvsep_mdx23" / "work",
                    repo_dir=DATA_DIR / "mvsep_mdx23" / "repo",
                )
                return sep.separate(audio)
            
            else:
                raise ValueError(f"Unknown separation model: {model_type}")

        try:
            start_time = time.time()
            separated_audio = await loop.run_in_executor(None, do_separation)
            elapsed = time.time() - start_time
            
            new_id = registry.register(
                separated_audio,
                source_type="separation",
                parent_id=audio_id,
                tags=["separated", model_type, two_stems],
            )
            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                message=f"Separation completed in {elapsed:.2f}s!",
                result={
                    "separated_audio_id": new_id,
                    "metadata": separated_audio.metadata(),
                    "elapsed_s": round(elapsed, 2),
                    "model_type": model_type,
                },
            )
        except Exception as e:
            logger.exception("Separation failed")
            task_manager.update_task(
                task_id,
                status="failed",
                error=str(e),
                message=f"Separation failed: {e}",
            )

    asyncio.create_task(run_sep())
    return web.json_response({"task_id": task_id})


async def handle_run_diarization(request: web.Request) -> web.Response:
    """Run speaker diarization in background."""
    data = await request.json()
    audio_id = data.get("audio_id")
    model_type = data.get("model_type", "pyannote").lower()  # pyannote, sortformer
    device = data.get("device", "auto")
    token = data.get("token") or os.getenv("HF_TOKEN")

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    task_id = task_manager.create_task(
        "diarization",
        {"audio_id": audio_id, "model_type": model_type, "device": device},
    )

    async def run_diar():
        task_manager.update_task(
            task_id,
            status="running",
            progress=0.1,
            message=f"Running speaker diarization with {model_type}...",
        )
        loop = asyncio.get_running_loop()

        def do_diarization():
            target_device = get_default_device() if device == "auto" else device
            
            if model_type == "pyannote":
                diarizer = PyannoteDiarizer(device=target_device, token=token)
                with diarizer:
                    return diarizer.diarize(audio)
            elif model_type == "sortformer":
                diarizer = SortformerDiarizer(device=target_device)
                with diarizer:
                    return diarizer.diarize(audio)
            else:
                raise ValueError(f"Unknown diarization model: {model_type}")

        try:
            start_time = time.time()
            result = await loop.run_in_executor(None, do_diarization)
            elapsed = time.time() - start_time

            # Format dataclass result to JSON
            result_dict = asdict(result)
            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                message=f"Diarization finished in {elapsed:.2f}s!",
                result={
                    "diarization": result_dict,
                    "elapsed_s": round(elapsed, 2),
                    "audio_id": audio_id,
                },
            )
        except Exception as e:
            logger.exception("Diarization failed")
            task_manager.update_task(
                task_id,
                status="failed",
                error=str(e),
                message=f"Diarization failed: {e}",
            )

    asyncio.create_task(run_diar())
    return web.json_response({"task_id": task_id})


async def handle_mix_audio(request: web.Request) -> web.Response:
    """Mix speech and music benchmark audio."""
    data = await request.json()
    speech_id = data.get("speech_id")
    music_id = data.get("music_id")
    target_smr_db = float(data.get("target_smr_db", 0.0))
    seed = int(data.get("seed", 42))

    speech_audio = registry.get_audio(speech_id)
    music_audio = registry.get_audio(music_id)

    if not speech_audio or not music_audio:
        return web.json_response({"error": "Both speech and music audios are required"}, status=400)

    mix_out = DATA_DIR / "benchmark_mixer" / f"mix_{int(time.time())}"
    mix_out.mkdir(parents=True, exist_ok=True)

    def do_mix():
        mixer = AudioMixer(sample_rate=44100, channels=1)
        return mixer.mix(
            speech=speech_audio,
            music=music_audio,
            target_smr_db=target_smr_db,
            seed=seed,
            output_dir=mix_out,
        )

    try:
        loop = asyncio.get_running_loop()
        mix_res = await loop.run_in_executor(None, do_mix)

        mixture_id = registry.register(mix_res.mixture, source_type="mix", tags=["mixture", f"smr_{target_smr_db}"])
        speech_ref_id = registry.register(mix_res.speech_reference, source_type="mix_ref", tags=["speech_ref"])
        music_ref_id = registry.register(mix_res.music_reference, source_type="mix_ref", tags=["music_ref"])

        return web.json_response({
            "mixture_id": mixture_id,
            "speech_ref_id": speech_ref_id,
            "music_ref_id": music_ref_id,
            "params": asdict(mix_res.parameters),
        })
    except Exception as e:
        logger.exception("Mixing failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_compare_spectrogram(request: web.Request) -> web.Response:
    """Run SpectrogramComparer on two audio objects and return comparison image."""
    data = await request.json()
    before_id = data.get("before_id")
    after_id = data.get("after_id")
    sample_rate = int(data.get("sample_rate", 16000))

    before = registry.get_audio(before_id)
    after = registry.get_audio(after_id)

    if not before or not after:
        return web.json_response({"error": "Both audio IDs are required"}, status=400)

    def run_compare():
        import matplotlib.pyplot as plt

        comparer = SpectrogramComparer(sample_rate=sample_rate)
        fig = comparer.compare(
            before=before,
            after=after,
            before_title=f"Before ({before.title})",
            after_title=f"After ({after.title})",
            residual_title="Residual (Mixture - Estimate)",
            show=False,
        )
        if fig is None:
            raise RuntimeError("Failed to generate figure")

        # Dark theme styling for figure
        fig.patch.set_facecolor("#141721")
        for ax in fig.axes:
            ax.set_facecolor("#0e111a")
            ax.tick_params(colors="#94a3b8")
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")
            ax.title.set_color("#e2e8f0")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    try:
        loop = asyncio.get_running_loop()
        img_bytes = await loop.run_in_executor(None, run_compare)
        return web.Response(body=img_bytes, content_type="image/png")
    except Exception as e:
        logger.exception("Spectrogram compare failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_compare_waveform(request: web.Request) -> web.Response:
    """Run WaveformComparer on two audio objects and return comparison image."""
    data = await request.json()
    before_id = data.get("before_id")
    after_id = data.get("after_id")
    sample_rate = int(data.get("sample_rate", 16000))

    before = registry.get_audio(before_id)
    after = registry.get_audio(after_id)

    if not before or not after:
        return web.json_response({"error": "Both audio IDs are required"}, status=400)

    def run_compare():
        import matplotlib.pyplot as plt

        comparer = WaveformComparer(sample_rate=sample_rate)
        fig = comparer.compare(
            before=before,
            after=after,
            before_title=f"Before ({before.title})",
            after_title=f"After ({after.title})",
            residual_title="Residual (Mixture - Estimate)",
            show=False,
        )
        if fig is None:
            raise RuntimeError("Failed to generate figure")

        fig.patch.set_facecolor("#141721")
        for ax in fig.axes:
            ax.set_facecolor("#0e111a")
            ax.tick_params(colors="#94a3b8")
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")
            ax.title.set_color("#e2e8f0")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    try:
        loop = asyncio.get_running_loop()
        img_bytes = await loop.run_in_executor(None, run_compare)
        return web.Response(body=img_bytes, content_type="image/png")
    except Exception as e:
        logger.exception("Waveform compare failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_get_task(request: web.Request) -> web.Response:
    """Get status and result for an asynchronous task."""
    task_id = request.match_info["id"]
    task = task_manager.get_task(task_id)
    if not task:
        return web.json_response({"error": "Task not found"}, status=404)
    return web.json_response(task)


# ==================== LIVE RELOAD SSE ====================


async def handle_live_reload_sse(request: web.Request) -> web.StreamResponse:
    """Server-Sent Events endpoint for hot live-reloading."""
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)
    queue: asyncio.Queue = asyncio.Queue()
    live_reload_subscribers.append(queue)

    # Send initial connection event
    await response.write(b"data: connected\n\n")

    try:
        while True:
            msg = await queue.get()
            try:
                await response.write(f"data: {msg}\n\n".encode("utf-8"))
            except (ConnectionResetError, aiohttp.ClientConnectionResetError):
                break
    except asyncio.CancelledError:
        pass
    finally:
        if queue in live_reload_subscribers:
            live_reload_subscribers.remove(queue)

    return response


async def file_watcher_loop(app: web.Application):
    """Background task to watch frontend files and trigger live reload."""
    mtimes: Dict[str, float] = {}

    def scan():
        changed = False
        for p in STATIC_DIR.rglob("*"):
            if p.is_file() and p.suffix in (".html", ".css", ".js"):
                try:
                    mt = p.stat().st_mtime
                    if str(p) in mtimes and mtimes[str(p)] != mt:
                        changed = True
                    mtimes[str(p)] = mt
                except Exception:
                    pass
        return changed

    scan()
    while True:
        await asyncio.sleep(0.5)
        if scan():
            logger.info("Static file modification detected! Broadcasting reload to %d clients...", len(live_reload_subscribers))
            for q in list(live_reload_subscribers):
                await q.put("reload")


async def start_background_tasks(app: web.Application):
    app["watcher"] = asyncio.create_task(file_watcher_loop(app))


async def cleanup_background_tasks(app: web.Application):
    watcher = app.get("watcher")
    if watcher and not watcher.done():
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass


# ==================== STATIC FILE HANDLERS ====================


async def handle_index(request: web.Request) -> web.Response:
    """Serve index.html with no-cache for instant development feedback."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return web.Response(text="index.html not found", status=404)
    with open(index_file, "r", encoding="utf-8") as f:
        content = f.read()
    return web.Response(text=content, content_type="text/html", headers={"Cache-Control": "no-cache"})


def create_app() -> web.Application:
    """Create and configure the aiohttp web application."""
    app = web.Application(client_max_size=1024 * 1024 * 500)  # 500 MB max upload

    # Routes
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/system/status", handle_status)
    app.router.add_get("/api/library", handle_list_library)
    app.router.add_post("/api/library/load", handle_load_library_file)
    app.router.add_post("/api/audio/upload", handle_upload_audio)
    app.router.add_post("/api/audio/youtube", handle_youtube_ingest)
    app.router.add_post("/api/crawler/inspect", handle_youtube_inspect)
    app.router.add_get("/api/crawler/history", handle_youtube_history)
    app.router.add_get("/api/audio", handle_list_audios)
    app.router.add_get("/api/audio/{id}", handle_get_audio_metadata)
    app.router.add_get("/api/audio/{id}/stream", handle_stream_audio)
    app.router.add_get("/api/audio/{id}/waveform", handle_get_waveform)
    app.router.add_get("/api/audio/{id}/spectrogram", handle_get_spectrogram)
    app.router.add_post("/api/audio/{id}/cut", handle_cut_audio)
    app.router.add_post("/api/audio/{id}/quick-save", handle_quick_save)
    app.router.add_post("/api/audio/{id}/save-to", handle_save_to)
    
    app.router.add_post("/api/separation/run", handle_run_separation)
    app.router.add_post("/api/diarization/run", handle_run_diarization)
    app.router.add_post("/api/benchmark/mix", handle_mix_audio)
    app.router.add_post("/api/compare/spectrogram", handle_compare_spectrogram)
    app.router.add_post("/api/compare/waveform", handle_compare_waveform)
    app.router.add_get("/api/tasks/{id}", handle_get_task)
    app.router.add_get("/api/live-reload", handle_live_reload_sse)

    # Static assets
    app.router.add_static("/static/", path=str(STATIC_DIR), name="static")

    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app


def main():
    """Main CLI entrypoint for running the web server."""
    import argparse
    parser = argparse.ArgumentParser(description="Audio Prepare Pipeline Web Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port number (default: 8765)")
    args = parser.parse_args()

    app = create_app()
    print("=" * 60)
    print("   🎙️  SONICSTUDIO - Audio Prepare & Separation Suite")
    print(f"   🚀 Running at: http://{args.host}:{args.port}")
    print(f"   ⚡ Compute Device: {get_default_device().upper()}")
    print("=" * 60)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
