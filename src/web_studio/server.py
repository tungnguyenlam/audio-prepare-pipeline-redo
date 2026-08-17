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
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
from aiohttp import web
import soundfile as sf
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".data" / "huggingface"))

from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio, _sanitize_filename_component
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
    device_count = 0
    devices = []
    
    if cuda_available:
        device_type = "cuda"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_name = f"CUDA ({device_count} GPUs: {torch.cuda.get_device_name(0)})"
        else:
            device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
        for i in range(device_count):
            devices.append({
                "index": i,
                "id": f"cuda:{i}",
                "name": torch.cuda.get_device_name(i),
            })
    elif mps_available:
        device_type = "mps"
        device_name = "Apple Silicon (MPS)"
        device_count = 1
        devices.append({"index": 0, "id": "mps", "name": "Apple Silicon (MPS)"})
        
    return {
        "device_type": device_type,
        "device_name": device_name,
        "device_count": device_count,
        "devices": devices,
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
        model_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register an Audio object and return a unique ID."""
        audio_id = f"aud_{uuid.uuid4().hex[:10]}"
        self._items[audio_id] = {
            "id": audio_id,
            "audio": audio,
            "source_type": source_type,
            "parent_id": parent_id,
            "tags": tags or [],
            "model_info": model_info or {},
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

    def unregister(self, audio_id: str) -> bool:
        """Remove an audio object from the in-memory registry."""
        if audio_id in self._items:
            del self._items[audio_id]
            self._waveform_cache.pop(audio_id, None)
            return True
        return False

    def unregister_path(self, path: str | Path) -> int:
        """Remove all registered audio objects that point at ``path``."""
        target = Path(path).resolve()
        matching_ids = [
            audio_id
            for audio_id, item in self._items.items()
            if Path(item["audio"].path).resolve() == target
        ]
        for audio_id in matching_ids:
            self.unregister(audio_id)
        return len(matching_ids)

    def clear_all(self) -> int:
        """Clear all registered items from in-memory session registry."""
        count = len(self._items)
        self._items.clear()
        self._waveform_cache.clear()
        return count

    def list_all(self) -> List[Dict[str, Any]]:
        """List all registered items formatted for the frontend."""
        result = []
        for audio_id, item in sorted(
            self._items.items(), key=lambda x: x[1]["created_at"], reverse=True
        ):
            audio: Audio = item["audio"]
            meta = audio.metadata()
            try:
                file_size = audio.path.stat().st_size if audio.path.is_file() else 0
            except OSError:
                file_size = 0
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
                    "model_info": item.get("model_info", {}),
                    "created_at": item["created_at"],
                    "file_size": file_size,
                }
            )
        return result

    def get_cached_waveform(self, audio_id: str) -> Optional[List[float]]:
        return self._waveform_cache.get(audio_id)

    def cache_waveform(self, audio_id: str, peaks: List[float]) -> None:
        self._waveform_cache[audio_id] = peaks


class TaskManager:
    """Run Studio background jobs through a bounded asynchronous queue."""

    def __init__(self, max_concurrency: int = 1) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._queue: asyncio.Queue[tuple[str, Callable[[], Awaitable[None]]]] = asyncio.Queue()
        self._queued_ids: List[str] = []
        self._workers: List[asyncio.Task[None]] = []
        self._running_ids: set[str] = set()
        self.max_concurrency = max(1, min(4, max_concurrency))

    async def start(self) -> None:
        """Start the fixed-size worker pool."""
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._worker_loop(), name=f"studio-task-worker-{index + 1}")
            for index in range(self.max_concurrency)
        ]
        logger.info("Studio task queue started with concurrency %d", self.max_concurrency)

    async def stop(self) -> None:
        """Stop queue workers during application shutdown."""
        workers = list(self._workers)
        self._workers.clear()
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)

    def enqueue(self, task_id: str, runner: Callable[[], Awaitable[None]]) -> None:
        """Add a previously created task to the worker queue."""
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task: {task_id}")
        self._queued_ids.append(task_id)
        self._queue.put_nowait((task_id, runner))
        self._refresh_queue_messages()

    async def _worker_loop(self) -> None:
        while True:
            task_id, runner = await self._queue.get()
            try:
                if task_id in self._queued_ids:
                    self._queued_ids.remove(task_id)
                task = self._tasks.get(task_id)
                if not task or task["status"] == "cancelled":
                    continue

                self._running_ids.add(task_id)
                self.update_task(task_id, status="running", message="Starting task...")
                await runner()
                task = self._tasks.get(task_id)
                if task and task["status"] == "running":
                    self.update_task(task_id, status="completed", progress=1.0)
            except asyncio.CancelledError:
                task = self._tasks.get(task_id)
                if task and task["status"] == "running":
                    self.update_task(
                        task_id,
                        status="cancelled",
                        message="Task interrupted by server shutdown.",
                    )
                raise
            except Exception as exc:
                logger.exception("Unhandled Studio task failure: %s", task_id)
                self.update_task(
                    task_id,
                    status="failed",
                    error=str(exc),
                    message=f"Task failed: {exc}",
                )
            finally:
                self._running_ids.discard(task_id)
                self._queue.task_done()
                self._refresh_queue_messages()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _refresh_queue_messages(self) -> None:
        for position, task_id in enumerate(self._queued_ids, start=1):
            task = self._tasks.get(task_id)
            if task and task["status"] == "pending":
                task["queue_position"] = position
                task["message"] = f"Queued — position {position}"

    def create_task(self, task_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        self._tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "pending",  # pending, running, completed, failed, cancelled
            "progress": 0.0,
            "message": "Task queued...",
            "error": None,
            "result": None,
            "created_at": time.time(),
            "start_time": None,
            "end_time": None,
            "queue_position": None,
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
                if status == "running" and t["start_time"] is None:
                    t["start_time"] = time.time()
                if status != "pending":
                    t["queue_position"] = None
            if progress is not None:
                t["progress"] = progress
            if message:
                t["message"] = message
            if error:
                t["error"] = error
            if result is not None:
                t["result"] = result
            if status in ("completed", "failed", "cancelled"):
                t["end_time"] = time.time()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self._tasks.get(task_id)
        return dict(task) if task else None

    def list_tasks(self) -> List[Dict[str, Any]]:
        """Return newest tasks first."""
        tasks = sorted(self._tasks.values(), key=lambda task: task["created_at"], reverse=True)
        return [dict(task) for task in tasks]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task without interrupting unsafe native model work."""
        task = self._tasks.get(task_id)
        if not task or task["status"] != "pending":
            return False
        if task_id in self._queued_ids:
            self._queued_ids.remove(task_id)
        self.update_task(task_id, status="cancelled", message="Cancelled while queued.")
        self._refresh_queue_messages()
        return True

    def clear_finished(self) -> int:
        """Remove completed, failed, or cancelled tasks from task memory."""
        to_delete = [
            tid for tid, task in self._tasks.items()
            if task.get("status") in ("completed", "failed", "cancelled")
        ]
        for tid in to_delete:
            del self._tasks[tid]
        return len(to_delete)

    def status(self) -> Dict[str, Any]:
        """Return a compact queue status summary."""
        return {
            "max_concurrency": self.max_concurrency,
            "running": len(self._running_ids),
            "queued": len(self._queued_ids),
        }


class EvaluationManager:
    """Persistent storage manager for human separation scoring & notes."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluations: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.file_path.is_file():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.evaluations = {item["id"]: item for item in data if isinstance(item, dict) and "id" in item}
                elif isinstance(data, dict):
                    self.evaluations = data
            except Exception as e:
                logger.error("Failed to load evaluations: %s", e)
                self.evaluations = {}

    def _save(self):
        try:
            self.file_path.write_text(
                json.dumps(list(self.evaluations.values()), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to save evaluations: %s", e)

    def get_all(self) -> List[Dict[str, Any]]:
        evals = list(self.evaluations.values())
        evals.sort(key=lambda x: x.get("updated_at", x.get("created_at", 0)), reverse=True)
        return evals

    def get_by_clip(self, clip_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.get_all() if e.get("clip_id") == clip_id]

    def save_evaluation(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        eval_id = eval_data.get("id") or f"eval-{uuid.uuid4().hex[:10]}"
        now = time.time()
        record = {
            "id": eval_id,
            "clip_id": str(eval_data.get("clip_id", "")),
            "clip_title": str(eval_data.get("clip_title", "")),
            "clip_path": str(eval_data.get("clip_path", "")),
            "model_id": str(eval_data.get("model_id", "")),
            "model_name": str(eval_data.get("model_name", "")),
            "stem": str(eval_data.get("stem", "vocals")),
            "separated_audio_id": str(eval_data.get("separated_audio_id", "")),
            "separated_audio_path": str(eval_data.get("separated_audio_path", "")),
            "score_overall": float(eval_data.get("score_overall", 5.0)),
            "score_vocal_clarity": int(eval_data.get("score_vocal_clarity", 5)),
            "score_bleed": int(eval_data.get("score_bleed", 5)),
            "score_artifacts": int(eval_data.get("score_artifacts", 5)),
            "notes": str(eval_data.get("notes", "")),
            "tags": list(eval_data.get("tags", [])),
            "created_at": float(eval_data.get("created_at", now)),
            "updated_at": now,
        }
        self.evaluations[eval_id] = record
        self._save()
        return record

    def delete_evaluation(self, eval_id: str) -> bool:
        if eval_id in self.evaluations:
            del self.evaluations[eval_id]
            self._save()
            return True
        return False


# Global instances
registry = AudioRegistry()
try:
    STUDIO_QUEUE_CONCURRENCY = int(os.getenv("STUDIO_QUEUE_CONCURRENCY", "1"))
except ValueError:
    logger.warning("Invalid STUDIO_QUEUE_CONCURRENCY; falling back to 1")
    STUDIO_QUEUE_CONCURRENCY = 1
task_manager = TaskManager(max_concurrency=STUDIO_QUEUE_CONCURRENCY)
evaluation_manager = EvaluationManager(DATA_DIR / "studio" / "evaluations.json")
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
    """Return system information and device status with shared GPU queue metrics."""
    info = get_system_device_info()
    info["registered_audios"] = len(registry.list_all())
    studio_q = task_manager.status()
    info["task_queue"] = studio_q
    try:
        from src.web_pipeline.queue_manager import queue_manager
        p_running = len(queue_manager.running_jobs)
        p_pending = len(queue_manager.pending_jobs)
        info["shared_queue"] = {
            "total_running": studio_q["running"] + p_running,
            "total_queued": studio_q["queued"] + p_pending,
            "studio_running": studio_q["running"],
            "studio_queued": studio_q["queued"],
            "pipeline_running": p_running,
            "pipeline_queued": p_pending,
        }
    except Exception:
        info["shared_queue"] = {
            "total_running": studio_q["running"],
            "total_queued": studio_q["queued"],
            "studio_running": studio_q["running"],
            "studio_queued": studio_q["queued"],
            "pipeline_running": 0,
            "pipeline_queued": 0,
        }
    try:
        from src.web_pipeline.hardware_monitor import hardware_monitor
        info["telemetry"] = hardware_monitor.get_system_telemetry()
    except Exception:
        info["telemetry"] = None
    return web.json_response(info)


async def handle_telemetry(request: web.Request) -> web.Response:
    """Return live system hardware telemetry."""
    from src.web_pipeline.hardware_monitor import hardware_monitor
    return web.json_response(hardware_monitor.get_system_telemetry())


def probe_audio_file_info(path: Path) -> dict[str, Any]:
    """Extract metadata (duration, sample rate, channels, title) from sidecar or audio headers."""
    sidecar_path = path.with_suffix(".json")
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if isinstance(sidecar, dict) and sidecar.get("kind") == "audio.sidecar":
                return {
                    "title": sidecar.get("title") or path.stem,
                    "source_id": sidecar.get("source_id") or path.stem,
                    "duration_s": float(sidecar.get("duration_s", 0.0)) if sidecar.get("duration_s") is not None else 0.0,
                    "sample_rate": int(sidecar.get("sample_rate", DEFAULT_SAMPLE_RATE)) if sidecar.get("sample_rate") is not None else DEFAULT_SAMPLE_RATE,
                    "channels": int(sidecar.get("channels", 1)) if sidecar.get("channels") is not None else 1,
                    "native_sample_rate": int(sidecar.get("native_sample_rate", sidecar.get("sample_rate", DEFAULT_SAMPLE_RATE))) if sidecar.get("native_sample_rate") is not None else DEFAULT_SAMPLE_RATE,
                    "history": sidecar.get("history", []),
                }
        except Exception:
            pass

    # Probe WAV with wave module (fast header read)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                sr = wf.getframerate()
                nframes = wf.getnframes()
                ch = wf.getnchannels()
                dur = nframes / float(sr) if sr else 0.0
                return {
                    "title": path.stem,
                    "source_id": path.stem,
                    "duration_s": round(dur, 2),
                    "sample_rate": sr,
                    "channels": ch,
                    "native_sample_rate": sr,
                    "history": [],
                }
        except Exception:
            pass

    # Fallback to soundfile.info for other formats (mp3, flac, ogg, m4a)
    try:
        info = sf.info(str(path))
        return {
            "title": path.stem,
            "source_id": path.stem,
            "duration_s": round(info.duration, 2),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "native_sample_rate": info.samplerate,
            "history": [],
        }
    except Exception:
        return {
            "title": path.stem,
            "source_id": path.stem,
            "duration_s": 0.0,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "channels": 1,
            "native_sample_rate": DEFAULT_SAMPLE_RATE,
            "history": [],
        }


def categorize_library_path(rel_path: str) -> str:
    """Determine clean, accurate library category based on relative project path."""
    p_lower = rel_path.lower()
    if "sources/speech" in p_lower or ("speech" in p_lower and "music" not in p_lower and "cuts" not in p_lower):
        return "Benchmark Speech"
    elif "sources/music" in p_lower or ("music" in p_lower and "speech" not in p_lower and "cuts" not in p_lower):
        return "Benchmark Music"
    elif "sources/cuts" in p_lower or "audio_cutter" in p_lower or "_cut_" in p_lower or "cuts" in p_lower:
        return "Audio Cuts"
    elif "stems" in p_lower or "separation" in p_lower or "demucs" in p_lower or "roformer" in p_lower or "mvsep" in p_lower:
        return "Separated Stems"
    elif "yt_crawler" in p_lower or "downloads" in p_lower:
        return "YouTube Downloads"
    elif "pipeline" in p_lower:
        return "Pipeline Assets"
    elif "temp" in p_lower or "quick_save" in p_lower:
        return "Quick Saves (temp)"
    elif "upload" in p_lower:
        return "Uploads"
    elif "data" in p_lower:
        return "Data Directory"
    else:
        return "Project Audio"


async def handle_list_library(request: web.Request) -> web.Response:
    """Scan and list audio files available in project directories with precise categorization and metadata."""
    # Ensure benchmark and output directories exist
    (ROOT_DIR / "benchmarks/separation/sources/speech").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "benchmarks/separation/sources/music").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "benchmarks/separation/sources/cuts").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "temp").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / ".data").mkdir(parents=True, exist_ok=True)

    scan_dirs = [
        ROOT_DIR / "benchmarks",
        ROOT_DIR / "data",
        ROOT_DIR / "temp",
        ROOT_DIR / ".data",
    ]

    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff"}
    files = []
    seen_paths = set()

    for directory in scan_dirs:
        if directory.is_dir():
            for p in directory.rglob("*"):
                # Ignore intermediate work directories and caches
                if any(part in ("work", ".cache", "checkpoints", "venv", ".git", "__pycache__") for part in p.parts):
                    continue
                if p.is_file() and p.suffix.lower() in extensions:
                    try:
                        resolved_str = str(p.resolve())
                        if resolved_str in seen_paths:
                            continue
                        seen_paths.add(resolved_str)

                        stat = p.stat()
                        # Zero byte files are unusable / corrupted downloads
                        if stat.st_size == 0:
                            continue

                        rel_path = str(p.relative_to(ROOT_DIR))
                        category = categorize_library_path(rel_path)
                        probe_meta = probe_audio_file_info(p)

                        files.append(
                            {
                                "category": category,
                                "name": p.name,
                                "title": probe_meta.get("title") or p.stem,
                                "source_id": probe_meta.get("source_id") or p.stem,
                                "path": rel_path,
                                "absolute_path": resolved_str,
                                "size": stat.st_size,
                                "modified": stat.st_mtime,
                                "format": p.suffix.lstrip(".").lower(),
                                "duration_s": probe_meta.get("duration_s", 0.0),
                                "sample_rate": probe_meta.get("sample_rate", DEFAULT_SAMPLE_RATE),
                                "channels": probe_meta.get("channels", 1),
                                "native_sample_rate": probe_meta.get("native_sample_rate", DEFAULT_SAMPLE_RATE),
                            }
                        )
                    except Exception:
                        pass

    files.sort(key=lambda x: x["modified"], reverse=True)
    return web.json_response({"files": files, "total": len(files)})


async def handle_stream_library_file(request: web.Request) -> web.Response:
    """Stream any permissible library audio file directly for preview/playback."""
    rel_or_abs = request.query.get("path")
    if not rel_or_abs:
        return web.Response(text="Path is required", status=400)
    target = Path(rel_or_abs)
    if not target.is_absolute():
        target = (ROOT_DIR / rel_or_abs).resolve()
    else:
        target = target.resolve()

    allowed_roots = [
        (ROOT_DIR / ".data").resolve(),
        (ROOT_DIR / "data").resolve(),
        (ROOT_DIR / "temp").resolve(),
        (ROOT_DIR / "benchmarks").resolve(),
    ]
    if not any(target.is_relative_to(root) for root in allowed_roots) or not target.is_file():
        return web.Response(text="Audio file not found or access denied", status=404)

    return web.FileResponse(
        target,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Content-Disposition": f'inline; filename="{target.name}"',
        },
    )


async def handle_download_library_file(request: web.Request) -> web.Response:
    """Download any permissible library audio file."""
    rel_or_abs = request.query.get("path")
    if not rel_or_abs:
        return web.Response(text="Path is required", status=400)
    target = Path(rel_or_abs)
    if not target.is_absolute():
        target = (ROOT_DIR / rel_or_abs).resolve()
    else:
        target = target.resolve()

    allowed_roots = [
        (ROOT_DIR / ".data").resolve(),
        (ROOT_DIR / "data").resolve(),
        (ROOT_DIR / "temp").resolve(),
        (ROOT_DIR / "benchmarks").resolve(),
    ]
    if not any(target.is_relative_to(root) for root in allowed_roots) or not target.is_file():
        return web.Response(text="Audio file not found or access denied", status=404)

    return web.FileResponse(
        target,
        headers={
            "Content-Disposition": f'attachment; filename="{target.name}"',
        },
    )


async def handle_delete_library_file(request: web.Request) -> web.Response:
    """Delete an audio file and its matching sidecar JSON from disk."""
    data = await request.json()
    rel_or_abs_path = data.get("path")
    if not rel_or_abs_path:
        return web.json_response({"error": "File path is required"}, status=400)

    target_path = Path(rel_or_abs_path)
    if not target_path.is_absolute():
        target_path = (ROOT_DIR / rel_or_abs_path).resolve()
    else:
        target_path = target_path.resolve()

    # Security check: Ensure target path is inside ROOT_DIR and within permitted directories
    try:
        target_path.relative_to(ROOT_DIR)
    except ValueError:
        return web.json_response({"error": "Cannot delete files outside project workspace"}, status=403)

    allowed_roots = [
        (ROOT_DIR / ".data").resolve(),
        (ROOT_DIR / "data").resolve(),
        (ROOT_DIR / "temp").resolve(),
        (ROOT_DIR / "benchmarks").resolve(),
    ]
    if not any(target_path.is_relative_to(ar) for ar in allowed_roots):
        return web.json_response({"error": "Path is outside permissible data/benchmark folders"}, status=403)

    if not target_path.is_file():
        return web.json_response({"error": f"File not found: {target_path.name}"}, status=404)

    try:
        # Delete the media file
        target_path.unlink()

        # Delete matching sidecar JSON if present
        sidecar = target_path.with_suffix(".json")
        if sidecar.is_file():
            sidecar.unlink()

        # A deleted source must not remain in the in-memory session registry.
        registry.unregister_path(target_path)

        logger.info("Deleted library file and sidecar: %s", target_path)
        return web.json_response({
            "status": "success",
            "deleted_file": str(target_path.name),
            "path": str(target_path.relative_to(ROOT_DIR)),
        })
    except Exception as e:
        logger.exception("Failed to delete file: %s", target_path)
        return web.json_response({"error": str(e)}, status=500)


async def handle_bulk_delete_library_files(request: web.Request) -> web.Response:
    """Delete multiple audio files and matching sidecar JSONs from disk."""
    data = await request.json()
    paths = data.get("paths", [])
    if not paths:
        return web.json_response({"error": "List of file paths is required"}, status=400)

    allowed_roots = [
        (ROOT_DIR / ".data").resolve(),
        (ROOT_DIR / "data").resolve(),
        (ROOT_DIR / "temp").resolve(),
        (ROOT_DIR / "benchmarks").resolve(),
    ]

    deleted_count = 0
    errors = []

    for p_str in paths:
        target = Path(p_str)
        if not target.is_absolute():
            target = (ROOT_DIR / p_str).resolve()
        else:
            target = target.resolve()

        if any(target.is_relative_to(ar) for ar in allowed_roots) and target.is_file():
            try:
                target.unlink()
                sidecar = target.with_suffix(".json")
                if sidecar.is_file():
                    sidecar.unlink()
                registry.unregister_path(target)
                deleted_count += 1
            except Exception as e:
                errors.append({"path": p_str, "error": str(e)})

    return web.json_response({
        "status": "success",
        "deleted_count": deleted_count,
        "errors": errors,
    })


async def handle_load_library_file(request: web.Request) -> web.Response:
    """Load a server file into active registry."""
    data = await request.json()
    file_path = data.get("path")
    if not file_path:
        return web.json_response({"error": "Path is required"}, status=400)

    resolved = Path(file_path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / file_path
    resolved = resolved.resolve()

    allowed_roots = [
        (ROOT_DIR / ".data").resolve(),
        (ROOT_DIR / "data").resolve(),
        (ROOT_DIR / "temp").resolve(),
        (ROOT_DIR / "benchmarks").resolve(),
    ]
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        return web.json_response({"error": "Path is outside permissible project audio folders"}, status=403)

    if not resolved.is_file():
        return web.json_response({"error": f"File not found: {resolved}"}, status=404)

    try:
        audio = Audio.from_file(resolved)
        category = categorize_library_path(str(resolved.relative_to(ROOT_DIR)))
        audio_id = registry.register(audio, source_type="library", tags=["library", category.lower().replace(" ", "_")])
        return web.json_response({"audio_id": audio_id, "metadata": audio.metadata()})
    except Exception as e:
        logger.exception("Error loading audio file")
        return web.json_response({"error": str(e)}, status=500)


async def handle_clear_session_audios(request: web.Request) -> web.Response:
    """Clear all active in-memory session registered audios."""
    count = registry.clear_all()
    return web.json_response({"status": "success", "cleared_count": count})


async def handle_upload_audio(request: web.Request) -> web.Response:
    """Handle multipart file upload from client."""
    reader = await request.multipart()
    field = await reader.next()
    if not field or field.name != "file":
        return web.json_response({"error": "Form field 'file' expected"}, status=400)

    filename = field.filename or f"upload_{int(time.time())}.wav"
    clean_name = Path(filename).name
    save_path = UPLOADS_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_name}"

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
        for p in yt_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff"}:
                continue
            try:
                audio = Audio.from_file(p)
                stat = p.stat()
                files.append({
                    "name": p.name,
                    "stem": p.stem,
                    "path": str(p.relative_to(ROOT_DIR)),
                    "absolute_path": str(p.resolve()),
                    "sample_rate": audio.sample_rate or 0,
                    "duration_s": audio.duration_s or 0,
                    "channels": audio.channels or 0,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
            except Exception:
                pass
    files.sort(key=lambda x: x["modified"], reverse=True)
    return web.json_response({"downloads": files})


async def handle_delete_youtube_file(request: web.Request) -> web.Response:
    """Delete a crawled YouTube audio file from disk."""
    data = await request.json()
    file_path = data.get("path")
    if not file_path:
        return web.json_response({"error": "Path is required"}, status=400)
    p = Path(file_path)
    if not p.is_absolute():
        p = ROOT_DIR / file_path
    p = p.resolve()
    if p.is_file() and p.is_relative_to((DATA_DIR / "yt_crawler").resolve()):
        p.unlink()
        sidecar = p.with_suffix(".json")
        if sidecar.is_file():
            sidecar.unlink()
        return web.json_response({"status": "success", "file": str(p.name)})
    return web.json_response({"error": "File not found or not in downloads directory"}, status=404)


async def handle_delete_audio(request: web.Request) -> web.Response:
    """Unregister an audio object from the in-memory registry."""
    audio_id = request.match_info.get("id")
    if not audio_id:
        return web.json_response({"error": "Audio ID is required"}, status=400)

    success = registry.unregister(audio_id)
    if success:
        return web.json_response({"status": "success", "audio_id": audio_id})
    return web.json_response({"error": "Audio not found"}, status=404)


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

    task_manager.enqueue(task_id, run_crawler)
    return web.json_response({"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202)


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
            
            model_label = "HTDemucs (Fine-Tuned)" if (model_type == "htdemucs" and model_name == "htdemucs_ft") else (
                "HTDemucs (Default)" if model_type == "htdemucs" else (
                    "BS-RoFormer" if model_type == "bs_roformer" else (
                        "Mel-RoFormer" if model_type == "mel_roformer" else (
                            "MVSep MDX23" if model_type == "mvsep_mdx23" else model_type.upper()
                        )
                    )
                )
            )

            new_id = registry.register(
                separated_audio,
                source_type="separation",
                parent_id=audio_id,
                tags=["separated", model_type, model_name or "default", two_stems],
                model_info={
                    "model_type": model_type,
                    "model_name": model_name or "default",
                    "model_label": model_label,
                    "stem": two_stems,
                    "parent_title": audio.title,
                    "elapsed_s": round(elapsed, 2),
                },
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
                    "model_label": model_label,
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

    task_manager.enqueue(task_id, run_sep)
    return web.json_response({"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202)


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

    task_manager.enqueue(task_id, run_diar)
    return web.json_response({"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202)


async def handle_extract_speaker_audio(request: web.Request) -> web.Response:
    """Extract all turns for a specific speaker and concatenate into a new Audio item."""
    data = await request.json()
    audio_id = data.get("audio_id")
    speaker_id = data.get("speaker_id")
    speaker_name = data.get("speaker_name") or speaker_id
    turns = data.get("turns", [])

    if not audio_id or not speaker_id:
        return web.json_response({"error": "audio_id and speaker_id are required"}, status=400)

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    # Filter turns for this speaker
    spk_turns = [t for t in turns if t.get("speaker_id") == speaker_id]
    if not spk_turns:
        return web.json_response({"error": f"No turns found for speaker {speaker_id}"}, status=400)

    out_dir = DATA_DIR / "diarization" / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    def do_extract():
        src_path = Path(audio.path)
        waveform, sr = sf.read(str(src_path), always_2d=True)
        total_frames = waveform.shape[0]

        valid_intervals = []
        for t in spk_turns:
            s_sec = float(t.get("start_s", 0))
            e_sec = float(t.get("end_s", 0))
            if e_sec > s_sec:
                s_frame = max(0, min(int(round(s_sec * sr)), total_frames))
                e_frame = max(0, min(int(round(e_sec * sr)), total_frames))
                if e_frame > s_frame:
                    valid_intervals.append((s_frame, e_frame))

        if not valid_intervals:
            raise ValueError(f"No valid audio samples found for speaker {speaker_id}")

        valid_intervals.sort(key=lambda x: x[0])
        segments = [waveform[s:e] for s, e in valid_intervals]
        combined = np.concatenate(segments, axis=0)

        sanitized_title = _sanitize_filename_component(audio.title or audio.source_id)
        sanitized_spk = _sanitize_filename_component(speaker_name or speaker_id)
        out_filename = f"{sanitized_title}_{sanitized_spk}.wav"
        out_path = out_dir / out_filename
        sf.write(str(out_path), combined, sr)

        extracted_audio = Audio.from_file(
            out_path,
            source_id=f"{audio.source_id}_{sanitized_spk}",
            title=f"{audio.title or audio.source_id} [{speaker_name}]",
            native_sample_rate=audio.native_sample_rate,
            history=(*audio.history, f"diar_extract_{speaker_id}"),
        )
        return extracted_audio

    try:
        loop = asyncio.get_running_loop()
        extracted = await loop.run_in_executor(None, do_extract)
        new_id = registry.register(
            extracted,
            source_type="speaker_stem",
            parent_id=audio_id,
            tags=["diarization", f"speaker:{speaker_id}"],
        )
        return web.json_response({
            "audio_id": new_id,
            "metadata": extracted.metadata(),
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "duration_s": extracted.duration_s,
        })
    except Exception as e:
        logger.exception("Speaker extraction failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_extract_all_speakers(request: web.Request) -> web.Response:
    """Extract audio for each speaker present in turns and register them all."""
    data = await request.json()
    audio_id = data.get("audio_id")
    turns = data.get("turns", [])
    speaker_names = data.get("speaker_names", {})  # map spk_id -> custom_name

    if not audio_id or not turns:
        return web.json_response({"error": "audio_id and non-empty turns are required"}, status=400)

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    unique_speakers = sorted(list(set(t.get("speaker_id") for t in turns if t.get("speaker_id"))))
    if not unique_speakers:
        return web.json_response({"error": "No valid speakers found in turns"}, status=400)

    out_dir = DATA_DIR / "diarization" / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    def do_extract_all():
        src_path = Path(audio.path)
        waveform, sr = sf.read(str(src_path), always_2d=True)
        total_frames = waveform.shape[0]
        results = []

        for spk_id in unique_speakers:
            spk_name = speaker_names.get(spk_id, spk_id)
            spk_turns = [t for t in turns if t.get("speaker_id") == spk_id]
            valid_intervals = []
            for t in spk_turns:
                s_sec = float(t.get("start_s", 0))
                e_sec = float(t.get("end_s", 0))
                if e_sec > s_sec:
                    s_frame = max(0, min(int(round(s_sec * sr)), total_frames))
                    e_frame = max(0, min(int(round(e_sec * sr)), total_frames))
                    if e_frame > s_frame:
                        valid_intervals.append((s_frame, e_frame))

            if not valid_intervals:
                continue

            valid_intervals.sort(key=lambda x: x[0])
            segments = [waveform[s:e] for s, e in valid_intervals]
            combined = np.concatenate(segments, axis=0)

            sanitized_title = _sanitize_filename_component(audio.title or audio.source_id)
            sanitized_spk = _sanitize_filename_component(spk_name or spk_id)
            out_filename = f"{sanitized_title}_{sanitized_spk}.wav"
            out_path = out_dir / out_filename
            sf.write(str(out_path), combined, sr)

            extracted_audio = Audio.from_file(
                out_path,
                source_id=f"{audio.source_id}_{sanitized_spk}",
                title=f"{audio.title or audio.source_id} [{spk_name}]",
                native_sample_rate=audio.native_sample_rate,
                history=(*audio.history, f"diar_extract_{spk_id}"),
            )
            results.append((spk_id, spk_name, extracted_audio))
        return results

    try:
        loop = asyncio.get_running_loop()
        extracted_list = await loop.run_in_executor(None, do_extract_all)
        registered = []
        for spk_id, spk_name, extracted_audio in extracted_list:
            new_id = registry.register(
                extracted_audio,
                source_type="speaker_stem",
                parent_id=audio_id,
                tags=["diarization", f"speaker:{spk_id}"],
            )
            registered.append({
                "audio_id": new_id,
                "speaker_id": spk_id,
                "speaker_name": spk_name,
                "metadata": extracted_audio.metadata(),
                "duration_s": extracted_audio.duration_s,
            })
        return web.json_response({"extracted": registered, "total_speakers": len(registered)})
    except Exception as e:
        logger.exception("Bulk speaker extraction failed")
        return web.json_response({"error": str(e)}, status=500)


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


async def handle_list_tasks(request: web.Request) -> web.Response:
    """List Studio tasks and the current queue state."""
    return web.json_response({"tasks": task_manager.list_tasks(), "queue": task_manager.status()})


async def handle_cancel_task(request: web.Request) -> web.Response:
    """Cancel a task that has not started yet."""
    task_id = request.match_info["id"]
    task = task_manager.get_task(task_id)
    if not task:
        return web.json_response({"error": "Task not found"}, status=404)
    if not task_manager.cancel_task(task_id):
        return web.json_response(
            {"error": "Only queued tasks can be cancelled safely", "task": task},
            status=409,
        )
async def handle_clear_tasks(request: web.Request) -> web.Response:
    """Clear completed, failed, and cancelled tasks from Studio queue memory."""
    cleared = task_manager.clear_finished()
    return web.json_response({"cleared": cleared, "tasks": task_manager.list_tasks(), "queue": task_manager.status()})


def get_shared_queue_data() -> Dict[str, Any]:
    """Aggregate shared GPU and task queue status across SonicStudio and SonicPipeline."""
    import torch
    telemetry: Dict[str, Any] = {}
    try:
        from src.web_pipeline.hardware_monitor import hardware_monitor
        telemetry = hardware_monitor.get_telemetry()
    except Exception:
        pass

    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    studio_tasks = task_manager.list_tasks()
    studio_status = task_manager.status()

    pipeline_jobs: List[Dict[str, Any]] = []
    pipeline_status = {"running": 0, "pending": 0, "is_paused": False, "max_concurrency": 2}
    try:
        from src.web_pipeline.queue_manager import queue_manager
        pipeline_jobs = queue_manager.list_jobs(limit=50)
        pipeline_status = {
            "running": len(queue_manager.running_jobs),
            "pending": len(queue_manager.pending_jobs),
            "is_paused": queue_manager.is_paused,
            "max_concurrency": queue_manager.max_concurrency,
        }
    except Exception:
        pass

    unified_items = []

    for t in studio_tasks:
        unified_items.append({
            "id": t["id"],
            "source": "studio",
            "source_label": "SonicStudio",
            "title": t.get("metadata", {}).get("title") or t.get("metadata", {}).get("model") or t.get("type", "Studio Task"),
            "type": t.get("type", "studio_task"),
            "status": t.get("status", "pending"),
            "progress": t.get("progress", 0.0),
            "message": t.get("message", ""),
            "error": t.get("error"),
            "created_at": t.get("created_at"),
            "start_time": t.get("start_time"),
            "end_time": t.get("end_time"),
            "queue_position": t.get("queue_position"),
            "metadata": t.get("metadata", {}),
        })

    for j in pipeline_jobs:
        unified_items.append({
            "id": j["id"],
            "source": "pipeline",
            "source_label": "SonicPipeline",
            "title": j.get("title", "Batch Job"),
            "type": j.get("type", "batch_job"),
            "status": j.get("status", "pending"),
            "progress": j.get("progress", 0.0),
            "message": j.get("current_step", ""),
            "error": j.get("error"),
            "created_at": j.get("created_at"),
            "start_time": j.get("started_at"),
            "end_time": j.get("finished_at"),
            "processed_items": j.get("processed_items", 0),
            "total_items": j.get("total_items", 0),
            "metadata": j.get("params", {}),
        })

    unified_items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)

    total_running = studio_status["running"] + pipeline_status["running"]
    total_queued = studio_status["queued"] + pipeline_status["pending"]

    return {
        "device": {
            "name": device_name,
            "cuda_available": torch.cuda.is_available(),
            "gpu_load_pct": telemetry.get("gpu", {}).get("load_percent") if telemetry.get("gpu") else None,
            "vram_used_mb": telemetry.get("gpu", {}).get("vram_used_mb") if telemetry.get("gpu") else None,
            "vram_total_mb": telemetry.get("gpu", {}).get("vram_total_mb") if telemetry.get("gpu") else None,
            "vram_pct": telemetry.get("gpu", {}).get("vram_percent") if telemetry.get("gpu") else None,
            "devices": telemetry.get("gpu", {}).get("devices", []),
        },
        "telemetry": telemetry,
        "summary": {
            "total_running": total_running,
            "total_queued": total_queued,
            "studio_running": studio_status["running"],
            "studio_queued": studio_status["queued"],
            "pipeline_running": pipeline_status["running"],
            "pipeline_queued": pipeline_status["pending"],
            "pipeline_paused": pipeline_status["is_paused"],
        },
        "items": unified_items,
        "studio": {"tasks": studio_tasks, "queue": studio_status},
        "pipeline": {"jobs": pipeline_jobs, "queue": pipeline_status},
    }


async def handle_shared_queue(request: web.Request) -> web.Response:
    """Return unified GPU workload queue across Studio and Pipeline."""
    return web.json_response(get_shared_queue_data())


async def handle_shared_queue_cancel(request: web.Request) -> web.Response:
    """Cancel a task or job across either SonicStudio or SonicPipeline."""
    item_id = request.match_info["id"]
    if item_id.startswith("task_") or task_manager.get_task(item_id):
        if not task_manager.cancel_task(item_id):
            return web.json_response({"error": "Only queued studio tasks can be cancelled"}, status=409)
        return web.json_response({"id": item_id, "source": "studio", "status": "cancelled"})
    else:
        try:
            from src.web_pipeline.queue_manager import queue_manager
            success = queue_manager.cancel_job(item_id)
            if not success:
                return web.json_response({"error": "Could not cancel pipeline job"}, status=409)
            return web.json_response({"id": item_id, "source": "pipeline", "status": "cancelled"})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)


# ==================== HUMAN EVALUATION & SCORING API ====================


async def handle_list_evaluations(request: web.Request) -> web.Response:
    """List all human evaluation records or filter by clip_id."""
    clip_id = request.query.get("clip_id")
    if clip_id:
        records = evaluation_manager.get_by_clip(clip_id)
    else:
        records = evaluation_manager.get_all()
    return web.json_response({"evaluations": records})


async def handle_save_evaluation(request: web.Request) -> web.Response:
    """Save or update human scoring and notes for an audio separation."""
    data = await request.json()
    if not data:
        return web.json_response({"error": "Payload required"}, status=400)
    saved = evaluation_manager.save_evaluation(data)
    return web.json_response({"evaluation": saved, "status": "success"})


async def handle_delete_evaluation(request: web.Request) -> web.Response:
    """Delete an evaluation record."""
    eval_id = request.match_info["id"]
    success = evaluation_manager.delete_evaluation(eval_id)
    if not success:
        return web.json_response({"error": "Evaluation not found"}, status=404)
    return web.json_response({"status": "success", "deleted_id": eval_id})


async def handle_export_evaluations(request: web.Request) -> web.Response:
    """Export human evaluations as JSON or CSV."""
    fmt = request.query.get("format", "csv").lower()
    evals = evaluation_manager.get_all()

    if fmt == "json":
        return web.Response(
            text=json.dumps(evals, indent=2, ensure_ascii=False),
            content_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="separation_evaluations.json"'},
        )

    # CSV export
    output = io.StringIO()
    import csv
    fieldnames = [
        "id", "clip_title", "clip_id", "model_name", "model_id", "stem",
        "score_overall", "score_vocal_clarity", "score_bleed", "score_artifacts",
        "notes", "tags", "created_at", "updated_at", "clip_path", "separated_audio_path"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in evals:
        row_copy = dict(row)
        if isinstance(row_copy.get("tags"), list):
            row_copy["tags"] = ";".join(str(t) for t in row_copy["tags"])
        writer.writerow(row_copy)

    return web.Response(
        text=output.getvalue(),
        content_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="separation_evaluations.csv"'},
    )


async def handle_batch_separation_compare(request: web.Request) -> web.Response:
    """Run multiple separation models sequentially on the same cut audio clip for side-by-side evaluation."""
    data = await request.json()
    audio_id = data.get("audio_id")
    models = data.get("models", [])
    device = data.get("device", "auto")
    two_stems = data.get("two_stems", "vocals")

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    if not models:
        models = [
            {"model_type": "htdemucs", "model_name": "htdemucs", "label": "HTDemucs (Default)"},
            {"model_type": "htdemucs", "model_name": "htdemucs_ft", "label": "HTDemucs (Fine-Tuned)"},
            {"model_type": "bs_roformer", "model_name": None, "label": "BS-RoFormer"},
            {"model_type": "mel_roformer", "model_name": None, "label": "Mel-RoFormer"},
        ]

    task_id = task_manager.create_task(
        "multi_model_separation",
        {"audio_id": audio_id, "models_count": len(models), "clip_title": audio.title},
    )

    async def run_batch():
        results = []
        target_device = get_default_device() if device == "auto" else device
        total = len(models)
        
        for idx, m_spec in enumerate(models, start=1):
            m_type = m_spec.get("model_type", "htdemucs")
            m_name = m_spec.get("model_name")
            m_label = m_spec.get("label") or f"{m_type}_{m_name or 'default'}"
            
            task_manager.update_task(
                task_id,
                status="running",
                progress=round((idx - 1) / total, 2),
                message=f"[{idx}/{total}] Separating with {m_label} on {target_device}...",
            )

            try:
                loop = asyncio.get_running_loop()
                def do_sep(curr_type=m_type, curr_name=m_name):
                    if curr_type == "htdemucs":
                        sep = HTDemucs(
                            model=curr_name or "htdemucs",
                            device=target_device,
                            two_stems=two_stems,
                            output_dir=DATA_DIR / "demucs" / "out",
                            work_dir=DATA_DIR / "demucs" / "work",
                        )
                        return sep.separate(audio)
                    elif curr_type == "bs_roformer":
                        kwargs = {
                            "device": target_device,
                            "output_dir": DATA_DIR / "bs_roformer" / "out",
                            "work_dir": DATA_DIR / "bs_roformer" / "work",
                        }
                        if curr_name:
                            kwargs["model"] = curr_name
                        sep = BSRoFormer(**kwargs)
                        with sep:
                            return sep.separate(audio)
                    elif curr_type == "mel_roformer":
                        kwargs = {
                            "device": target_device,
                            "output_dir": DATA_DIR / "mel_roformer" / "out",
                            "work_dir": DATA_DIR / "mel_roformer" / "work",
                        }
                        if curr_name:
                            kwargs["model"] = curr_name
                        sep = MelRoFormer(**kwargs)
                        with sep:
                            return sep.separate(audio)
                    elif curr_type == "mvsep_mdx23":
                        sep = MVSepMDX23(
                            device=target_device,
                            output_dir=DATA_DIR / "mvsep_mdx23" / "out",
                            work_dir=DATA_DIR / "mvsep_mdx23" / "work",
                            repo_dir=DATA_DIR / "mvsep_mdx23" / "repo",
                        )
                        return sep.separate(audio)
                    else:
                        raise ValueError(f"Unknown model type: {curr_type}")

                t0 = time.time()
                sep_audio = await loop.run_in_executor(None, do_sep)
                elapsed = time.time() - t0

                new_id = registry.register(
                    sep_audio,
                    source_type="separation",
                    parent_id=audio_id,
                    tags=["separated", m_type, m_name or "default", two_stems],
                    model_info={
                        "model_type": m_type,
                        "model_name": m_name or "default",
                        "model_label": m_label,
                        "stem": two_stems,
                        "parent_title": audio.title,
                        "elapsed_s": round(elapsed, 2),
                    },
                )
                results.append({
                    "model_id": f"{m_type}_{m_name or 'default'}",
                    "model_type": m_type,
                    "model_name": m_name,
                    "label": m_label,
                    "stem": two_stems,
                    "audio_id": new_id,
                    "title": sep_audio.title,
                    "path": str(sep_audio.path),
                    "elapsed_s": round(elapsed, 2),
                    "metadata": sep_audio.metadata(),
                })
            except Exception as e:
                logger.exception("Failed model separation: %s", m_label)
                results.append({
                    "model_id": f"{m_type}_{m_name or 'default'}",
                    "label": m_label,
                    "error": str(e),
                })

        task_manager.update_task(
            task_id,
            status="completed",
            progress=1.0,
            message=f"Completed {len(results)} model separations for '{audio.title}'!",
            result={
                "clip_id": audio_id,
                "clip_title": audio.title,
                "clip_path": str(audio.path),
                "results": results,
            },
        )

    task_manager.enqueue(task_id, run_batch)
    return web.json_response({"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202)


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


@web.middleware
async def no_cache_middleware(request: web.Request, handler):
    """Ensure static files and HTML are never served from browser stale cache in development."""
    response = await handler(request)
    if request.path.startswith("/static/") or request.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


async def file_watcher_loop(app: web.Application):
    """Background task to watch frontend and studio backend files and trigger live reload."""
    mtimes: Dict[str, float] = {}

    def scan():
        changed = False
        # Watch static frontend files
        for p in STATIC_DIR.rglob("*"):
            if p.is_file() and p.suffix in (".html", ".css", ".js"):
                try:
                    mt = p.stat().st_mtime
                    if str(p) in mtimes and mtimes[str(p)] != mt:
                        changed = True
                    mtimes[str(p)] = mt
                except Exception:
                    pass

        # Watch Python server/router files
        for p in (ROOT_DIR / "src" / "web_studio").rglob("*.py"):
            if p.is_file():
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
            logger.info("Modification detected! Broadcasting hot reload to %d client(s)...", len(live_reload_subscribers))
            for q in list(live_reload_subscribers):
                await q.put("reload")


async def start_background_tasks(app: web.Application):
    await task_manager.start()
    app["studio_watcher"] = asyncio.create_task(file_watcher_loop(app))


async def cleanup_background_tasks(app: web.Application):
    await task_manager.stop()
    watcher = app.get("studio_watcher")
    if watcher and not watcher.done():
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass


# ==================== STATIC FILE HANDLERS ====================


async def handle_index(request: web.Request) -> web.Response:
    """Serve modular index.html with composed partials and no-cache for instant development feedback."""
    from src.web_backend.html_composer import compose_html
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return web.Response(text="index.html not found", status=404)
    content = compose_html(index_file)
    return web.Response(
        text=content,
        content_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"},
    )


def register_api_routes(app: web.Application) -> None:
    """Register SonicStudio API routes on an application.

    Args:
        app: Aiohttp application that owns the shared backend.
    """
    app.router.add_get("/api/system/status", handle_status)
    app.router.add_get("/api/library", handle_list_library)
    app.router.add_get("/api/library/stream", handle_stream_library_file)
    app.router.add_get("/api/library/download", handle_download_library_file)
    app.router.add_post("/api/library/load", handle_load_library_file)
    app.router.add_post("/api/library/delete", handle_delete_library_file)
    app.router.add_post("/api/library/bulk-delete", handle_bulk_delete_library_files)
    app.router.add_post("/api/audio/clear-all", handle_clear_session_audios)
    app.router.add_post("/api/audio/upload", handle_upload_audio)
    app.router.add_post("/api/audio/youtube", handle_youtube_ingest)
    app.router.add_post("/api/crawler/inspect", handle_youtube_inspect)
    app.router.add_get("/api/crawler/history", handle_youtube_history)
    app.router.add_post("/api/crawler/delete", handle_delete_youtube_file)
    app.router.add_get("/api/audio", handle_list_audios)
    app.router.add_get("/api/audio/{id}", handle_get_audio_metadata)
    app.router.add_delete("/api/audio/{id}", handle_delete_audio)
    app.router.add_get("/api/audio/{id}/stream", handle_stream_audio)
    app.router.add_get("/api/audio/{id}/waveform", handle_get_waveform)
    app.router.add_get("/api/audio/{id}/spectrogram", handle_get_spectrogram)
    app.router.add_post("/api/audio/{id}/cut", handle_cut_audio)
    app.router.add_post("/api/audio/{id}/quick-save", handle_quick_save)
    app.router.add_post("/api/audio/{id}/save-to", handle_save_to)
    
    app.router.add_post("/api/separation/run", handle_run_separation)
    app.router.add_post("/api/separation/batch-compare", handle_batch_separation_compare)
    app.router.add_get("/api/evaluations", handle_list_evaluations)
    app.router.add_post("/api/evaluations", handle_save_evaluation)
    app.router.add_delete("/api/evaluations/{id}", handle_delete_evaluation)
    app.router.add_get("/api/evaluations/export", handle_export_evaluations)
    app.router.add_post("/api/diarization/run", handle_run_diarization)
    app.router.add_post("/api/diarization/extract-speaker", handle_extract_speaker_audio)
    app.router.add_post("/api/diarization/extract-all-speakers", handle_extract_all_speakers)
    app.router.add_post("/api/benchmark/mix", handle_mix_audio)
    app.router.add_post("/api/compare/spectrogram", handle_compare_spectrogram)
    app.router.add_post("/api/compare/waveform", handle_compare_waveform)
    app.router.add_get("/api/tasks", handle_list_tasks)
    app.router.add_post("/api/tasks/clear", handle_clear_tasks)
    app.router.add_get("/api/tasks/{id}", handle_get_task)
    app.router.add_delete("/api/tasks/{id}", handle_cancel_task)
    app.router.add_get("/api/queue/shared", handle_shared_queue)
    app.router.add_delete("/api/queue/shared/{id}", handle_shared_queue_cancel)
    app.router.add_post("/api/queue/shared/{id}/cancel", handle_shared_queue_cancel)
    app.router.add_get("/api/telemetry", handle_telemetry)
    app.router.add_get("/api/live-reload", handle_live_reload_sse)


def register_lifecycle(app: web.Application) -> None:
    """Register SonicStudio background services on an application.

    Args:
        app: Aiohttp application that owns the shared backend.
    """
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)


def create_app() -> web.Application:
    """Create a standalone SonicStudio application for compatibility."""
    app = web.Application(
        client_max_size=1024 * 1024 * 500,  # 500 MB max upload
        middlewares=[no_cache_middleware],
    )

    register_api_routes(app)

    app.router.add_get("/", handle_index)

    # Static assets
    app.router.add_static("/static/", path=str(STATIC_DIR), name="static")

    register_lifecycle(app)

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
