"""Web server application for the audio preparation pipeline.

Provides a full REST API, background task management, audio streaming,
waveform extraction, spectrogram generation, and static file serving for
the frontend studio.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
import wave
import zipfile
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

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

from src.data_paths import (
    portable_data_path,
    portable_data_payload,
    resolve_data_path,
    resolve_data_payload,
)
from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio, _sanitize_filename_component
from src.utils.AudioCutter import AudioCutter, AudioCutterError
from src.utils.SpectrogramComparer import SpectrogramComparer
from src.separation import HTDemucs, BSRoFormer, MelRoFormer, MVSepMDX23
from src.diarization import (
    ClusteringDiarizer,
    ClusteringWorkerDiarizer,
    DEFAULT_GEMINI_FLASH_LITE_MODEL_ID,
    DEFAULT_GEMINI_MODEL_ID,
    DEFAULT_GEMMA4_MODEL_ID,
    DEFAULT_JITTER_MAX_DURATION_S,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MERGE_SAME_SPEAKER_GAP_S,
    DEFAULT_MIN_SECONDARY_SPEECH_S,
    DEFAULT_VIBEVOICE_BATCH_SIZE,
    DEFAULT_MIN_TURN_DURATION_S,
    DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS,
    DEFAULT_UNSLOTH_ENDPOINT,
    DEFAULT_VIBEVOICE_MODEL_ID,
    DiariZenWorkerDiarizer,
    OVERLAP_PROMPT,
    PyannoteDiarizer,
    SortformerWorkerDiarizer,
    SpeakerVerifier,
    SpeakerVerifierError,
    SpeakerPurityResult,
    ThreeDSpeakerDiarizer,
    ThreeDSpeakerWorkerDiarizer,
    VibeVoicePurityWorkerVerifier,
    clean_speaker_turns,
    create_overlap_verifier,
    evaluate_diarization,
    is_overlap_readiness_error,
    pad_and_merge_intervals,
    vibevoice_studio_models,
)
from src.diarization.OverlapVerifier import OverlapVerifierError
from src.diarization.SortformerDiarizer import (
    DEFAULT_OFFSET as DEFAULT_SORTFORMER_OFFSET,
    DEFAULT_ONSET as DEFAULT_SORTFORMER_ONSET,
    DEFAULT_PAD_OFFSET_S as DEFAULT_SORTFORMER_PAD_OFFSET_S,
    DEFAULT_PAD_ONSET_S as DEFAULT_SORTFORMER_PAD_ONSET_S,
)
from src.diarization.schemas import (
    DIARIZATION_RESULT_KIND,
    DiarizationResult,
    Speaker,
    SpeakerTurn,
)
from src.yt_crawler.YtCrawlerClass import YtCrawler, parse_crawl_sample_rate

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
LIBRARY_ALLOWED_ROOTS = (
    DATA_DIR.resolve(),
    (ROOT_DIR / "data").resolve(),
    TEMP_DIR.resolve(),
    (ROOT_DIR / "benchmarks").resolve(),
)
LIBRARY_SKIP_DIR_NAMES = {
    "work",
    ".cache",
    "checkpoints",
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    "huggingface",
    "node_modules",
    ".uv",
}
LIBRARY_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff"}
LIBRARY_CATEGORY_ORDER = (
    "speech",
    "music",
    "cuts",
    "stems",
    "verified",
    "diarized",
    "ingest",
    "pipeline",
    "uploads",
    "temp",
    "data",
    "other",
)
DIARIZATION_RESULTS_DIR = DATA_DIR / "diarization" / "results"
DIARIZATION_VERIFICATIONS_DIR = DATA_DIR / "diarization" / "verifications"
DIARIZATION_PREVIEW_DIR = DATA_DIR / "diarization" / "preview"
DIARIZATION_ANNOTATIONS_DIR = DATA_DIR / "diarization" / "annotations"
AUDIO_SEGMENT_DIR = DATA_DIR / "audio_cutter" / "segments"
AUDIO_REGISTRY_PATH = DATA_DIR / "studio" / "audio_registry.json"
MAX_SEGMENT_ZIP_ITEMS = 2000

DEFAULT_EXTRACTION_PRE_ROLL_S = DEFAULT_SORTFORMER_PAD_ONSET_S
DEFAULT_EXTRACTION_POST_ROLL_S = DEFAULT_SORTFORMER_PAD_OFFSET_S

# Ensure runtime directories exist
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DIARIZATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DIARIZATION_VERIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)
DIARIZATION_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
DIARIZATION_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_default_device() -> str:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        best_index = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.get_device_properties(index).total_memory,
        )
        return f"cuda:{best_index}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"




def normalize_queue_device(device: str | None) -> str:
    """Map a requested compute device to a dedicated queue lane key.

    Args:
        device: Requested device string such as ``cuda:0``, ``auto``, or ``cpu``.

    Returns:
        Stable lane id used for per-GPU / CPU queue routing.
    """
    raw = (device or "").strip().lower()
    if not raw or raw in {"auto", "cuda"}:
        return get_default_device()
    if raw.startswith("cuda:"):
        if torch.cuda.is_available():
            return raw
        return "cpu"
    if raw in {"cpu", "mps"}:
        return raw
    return get_default_device()


def discover_queue_devices() -> List[str]:
    """Return the default set of queue lanes for this host."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(f"cuda:{index}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices

def get_system_device_info() -> dict[str, Any]:
    """Return hardware accelerator and environment details."""
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    device_name = "CPU"
    device_type = "cpu"
    device_count = 0
    devices = []

    telemetry_gpu: Dict[str, Any] = {}
    try:
        from src.web_pipeline.hardware_monitor import hardware_monitor
        telemetry_gpu = hardware_monitor.get_gpu_info()
    except Exception:
        pass
    
    if cuda_available:
        device_type = "cuda"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_name = f"CUDA ({device_count} GPUs: {torch.cuda.get_device_name(0)})"
        else:
            device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
        if telemetry_gpu and telemetry_gpu.get("devices"):
            devices = telemetry_gpu["devices"]
        else:
            for i in range(device_count):
                devices.append({
                    "index": i,
                    "id": f"cuda:{i}",
                    "name": torch.cuda.get_device_name(i),
                    "power_w": None,
                    "power_limit_w": None,
                    "power_percent": None,
                })
    elif mps_available:
        device_type = "mps"
        device_name = "Apple Silicon (MPS)"
        device_count = 1
        devices.append({"index": 0, "id": "mps", "name": "Apple Silicon (MPS)", "power_w": None, "power_limit_w": None})
        
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


def _get_device_power_w(target_device: str) -> Optional[float]:
    """Helper to query current power draw (watts) for a target device."""
    try:
        from src.web_pipeline.hardware_monitor import hardware_monitor
        gpu_info = hardware_monitor.get_gpu_info()
        if not gpu_info or not gpu_info.get("available"):
            return None
        if target_device.startswith("cuda:"):
            try:
                idx = int(target_device.split(":")[1])
                for dev in gpu_info.get("devices", []):
                    if dev.get("index") == idx:
                        return dev.get("power_w")
            except (ValueError, IndexError):
                pass
        return gpu_info.get("power_w")
    except Exception:
        return None


def _task_is_cancelled(task_id: str) -> bool:
    task = task_manager.get_task(task_id)
    return bool(task and task.get("status") == "cancelled")


def _apply_cli_progress(task_id: str, message: str, prefix: str = "") -> None:
    """Update a Studio task from a backend log line when numeric progress exists."""
    from src.web_pipeline.queue_manager import parse_progress_text

    task = task_manager.get_task(task_id)
    if not task or task["status"] != "running":
        return
    display = f"{prefix}{message}" if prefix else message
    percent = parse_progress_text(message)
    kwargs: Dict[str, Any] = {"message": display[:240]}
    if percent is not None:
        kwargs["progress"] = percent / 100.0
        kwargs["progress_known"] = True
    task_manager.update_task(task_id, **kwargs)


def _cli_progress_reporter(
    task_id: str,
    loop: asyncio.AbstractEventLoop,
    prefix: str,
) -> Callable[[str], None]:
    def report(message: str) -> None:
        loop.call_soon_threadsafe(_apply_cli_progress, task_id, message, prefix)

    return report



class AudioRegistry:
    """File-backed store mapping audio IDs to Audio objects and metadata."""

    def __init__(self, persist_path: Path = AUDIO_REGISTRY_PATH) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._waveform_cache: Dict[str, Dict[str, Any]] = {}
        self._persist_path = persist_path
        self._restoring = False

    def register(
        self,
        audio: Audio,
        *,
        source_type: str = "local",
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        audio_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> str:
        """Register an Audio object and return a stable ID.

        The same file path reuses its previous ID so diarization history can
        find the source after a server restart.
        """
        existing_id = self.find_id_by_path(audio.path)
        if existing_id:
            return existing_id
        requested_id = str(audio_id or "").strip()
        if (
            requested_id
            and requested_id not in self._items
            and "/" not in requested_id
            and "\\" not in requested_id
        ):
            resolved_id = requested_id
        else:
            resolved_id = f"aud_{uuid.uuid4().hex[:10]}"
        raw_tags = [str(tag) for tag in (tags or [])]
        system_tags = {
            tag for tag in raw_tags
            if tag.startswith(("type:", "stage:", "speaker:", "profile:", "verification:"))
        }
        custom_tags = [tag for tag in raw_tags if tag not in system_tags]
        type_tag = {
            "separation": "type:stem",
            "cut": "type:cut",
            "speaker_stem": "type:cut",
            "purity_stem": "type:stem",
        }.get(source_type, "type:source")
        stage_tag = {
            "separation": "stage:separated",
            "diarization": "stage:diarized",
            "speaker_stem": "stage:diarized",
            "purity_stem": "stage:verified",
        }.get(source_type, "stage:ingested")
        system_tags.update({type_tag, stage_tag})
        if source_type == "purity_stem":
            system_tags.add("verification:passed")
        self._items[resolved_id] = {
            "id": resolved_id,
            "audio": audio,
            "source_type": source_type,
            "parent_id": parent_id,
            "custom_tags": sorted(set(custom_tags)),
            "system_tags": sorted(system_tags),
            "model_info": model_info or {},
            "created_at": float(created_at) if created_at is not None else time.time(),
        }
        self._persist()
        return resolved_id

    def get_audio(self, audio_id: str) -> Optional[Audio]:
        """Retrieve the Audio object for an ID."""
        if not audio_id:
            return None
        item = self._items.get(audio_id)
        if item:
            return item["audio"]
        for it in self._items.values():
            if it["audio"].source_id == audio_id:
                return it["audio"]
        found_id = self.find_id_by_path(audio_id)
        if found_id:
            return self._items[found_id]["audio"]
        raw_id = audio_id[4:] if audio_id.startswith("lib:") else audio_id
        lib_path = resolve_library_path(raw_id)
        if lib_path and lib_path.is_file():
            try:
                return Audio.from_file(lib_path)
            except Exception:
                pass
        return None

    def get_item(self, audio_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the full registered item dictionary."""
        if not audio_id:
            return None
        item = self._items.get(audio_id)
        if item:
            return item
        for it in self._items.values():
            if it["audio"].source_id == audio_id:
                return it
        found_id = self.find_id_by_path(audio_id)
        if found_id:
            return self._items[found_id]
        return None

    def find_id_by_path(self, path: str | Path) -> str | None:
        """Return the registered ID for a file path, if present."""
        try:
            target = Path(path).expanduser().resolve()
        except OSError:
            return None
        for audio_id, item in self._items.items():
            try:
                if Path(item["audio"].path).resolve() == target:
                    return audio_id
            except OSError:
                continue
        return None

    def unregister(self, audio_id: str, *, persist: bool = True) -> bool:
        """Remove an audio object from the session registry."""
        if audio_id in self._items:
            del self._items[audio_id]
            self._waveform_cache.pop(audio_id, None)
            if persist:
                self._persist()
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
            self.unregister(audio_id, persist=False)
        if matching_ids:
            self._persist()
        return len(matching_ids)

    def clear_all(self) -> int:
        """Clear all registered items from the session registry."""
        count = len(self._items)
        self._items.clear()
        self._waveform_cache.clear()
        self._persist()
        return count

    def restore(self) -> int:
        """Reload session items whose audio files still exist on disk.

        Returns:
            Number of restored session items.
        """
        try:
            if not self._persist_path.is_file():
                return 0
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not restore session audio registry: %s", exc)
            return 0
        records = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            logger.warning("Ignoring invalid session audio registry payload")
            return 0
        restored = 0
        self._restoring = True
        try:
            for record in records:
                if not isinstance(record, dict):
                    continue
                raw_path = record.get("path")
                if not raw_path:
                    continue
                path = resolve_data_path(str(raw_path))
                try:
                    exists = path.is_file()
                except OSError as exc:
                    logger.warning("Skipping inaccessible session audio %s: %s", path, exc)
                    continue
                if not exists:
                    continue
                try:
                    audio = Audio.from_file(path)
                    self.register(
                        audio,
                        source_type=str(record.get("source_type") or "local"),
                        parent_id=record.get("parent_id"),
                        tags=[
                            *list(record.get("system_tags") or []),
                            *list(record.get("custom_tags") or []),
                        ],
                        model_info=record.get("model_info") or {},
                        audio_id=str(record.get("id") or ""),
                        created_at=record.get("created_at"),
                    )
                    restored += 1
                except Exception as exc:
                    logger.warning("Skipping persisted session audio %s: %s", path, exc)
        finally:
            self._restoring = False
        self._persist()
        return restored

    def _persist(self) -> None:
        """Atomically write the session registry, skipping missing files."""
        if self._restoring:
            return
        records = []
        for audio_id, item in self._items.items():
            audio: Audio = item["audio"]
            try:
                path = Path(audio.path).resolve()
                exists = path.is_file()
            except OSError:
                continue
            if not exists:
                continue
            stored_path = portable_data_path(path)
            if Path(stored_path).is_absolute():
                logger.info("Not persisting machine-local audio path: %s", path)
                continue
            try:
                model_info = json.loads(
                    json.dumps(item.get("model_info") or {}, default=str, allow_nan=False)
                )
            except (TypeError, ValueError):
                model_info = {}
            records.append(
                {
                    "id": audio_id,
                    "path": stored_path,
                    "source_type": item["source_type"],
                    "parent_id": item["parent_id"],
                    "custom_tags": list(item["custom_tags"]),
                    "system_tags": list(item["system_tags"]),
                    "model_info": model_info,
                    "created_at": item["created_at"],
                }
            )
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._persist_path.with_suffix(".json.tmp")
            temp_path.write_text(
                json.dumps({"items": records}, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            temp_path.replace(self._persist_path)
        except Exception as exc:
            logger.warning("Could not persist session audio registry: %s", exc)

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
                    "source_url": audio.source_url,
                    "channel_id": audio.channel_id,
                    "channel_name": audio.channel_name,
                    "channel_url": audio.channel_url,
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
                    "custom_tags": item["custom_tags"],
                    "system_tags": item["system_tags"],
                    "tags": [*item["system_tags"], *item["custom_tags"]],
                    "model_info": item.get("model_info", {}),
                    "created_at": item["created_at"],
                    "file_size": file_size,
                }
            )
        return result

    def get_cached_waveform(self, audio_id: str, bins: int) -> Optional[Dict[str, Any]]:
        cached = self._waveform_cache.get(audio_id)
        if cached and cached.get("requested_bins") == bins:
            return cached
        return None

    def cache_waveform(self, audio_id: str, waveform: Dict[str, Any]) -> None:
        """Cache the latest full-track overview, never a zoomed window."""
        self._waveform_cache[audio_id] = waveform


class TaskManager:
    """Per-device Studio task queues with independent worker pools.

    Each GPU (and CPU/MPS) owns its own FIFO lane so work targeting
    ``cuda:0`` never blocks ``cuda:1``. Default concurrency is one worker
    per device to avoid VRAM contention on the same accelerator.
    """

    def __init__(self, workers_per_device: int = 1) -> None:
        self.workers_per_device = max(1, min(4, workers_per_device))
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._device_queues: Dict[str, asyncio.Queue[tuple[str, Callable[[], Awaitable[None]]]]] = {}
        self._device_queued_ids: Dict[str, List[str]] = {}
        self._device_running_ids: Dict[str, set[str]] = {}
        self._device_workers: Dict[str, List[asyncio.Task[None]]] = {}
        self._runner_tasks: Dict[str, asyncio.Task[None]] = {}
        self._cancel_callbacks: Dict[str, Callable[[], None]] = {}
        self._started = False

    @property
    def max_concurrency(self) -> int:
        """Compatibility alias: workers available per device lane."""
        return self.workers_per_device

    def _ensure_device_lane(self, device: str) -> None:
        """Create the FIFO queue and workers for ``device`` if missing."""
        if device in self._device_queues:
            return
        self._device_queues[device] = asyncio.Queue()
        self._device_queued_ids[device] = []
        self._device_running_ids[device] = set()
        workers: List[asyncio.Task[None]] = []
        if self._started:
            for index in range(self.workers_per_device):
                workers.append(
                    asyncio.create_task(
                        self._worker_loop(device),
                        name=f"studio-task-worker-{device}-{index + 1}",
                    )
                )
        self._device_workers[device] = workers
        logger.info(
            "Studio queue lane ready for %s (%d worker(s))",
            device,
            self.workers_per_device,
        )

    async def start(self) -> None:
        """Start per-device worker pools for all known accelerators."""
        if self._started:
            return
        self._started = True
        for device in discover_queue_devices():
            self._ensure_device_lane(device)
            workers = self._device_workers[device]
            if not workers:
                self._device_workers[device] = [
                    asyncio.create_task(
                        self._worker_loop(device),
                        name=f"studio-task-worker-{device}-{index + 1}",
                    )
                    for index in range(self.workers_per_device)
                ]
        logger.info(
            "Studio task queues started (%d worker(s)/device, lanes=%s)",
            self.workers_per_device,
            ", ".join(sorted(self._device_queues)),
        )

    async def stop(self) -> None:
        """Cancel queued and running tasks, then stop all device workers."""
        for task_id, task in list(self._tasks.items()):
            if task.get("status") in ("pending", "running"):
                self.update_task(
                    task_id,
                    status="cancelled",
                    message="Task interrupted by server shutdown.",
                )
        for callback in list(self._cancel_callbacks.values()):
            try:
                callback()
            except Exception:
                logger.exception("Studio cancel callback failed during shutdown")
        self._cancel_callbacks.clear()
        for runner in list(self._runner_tasks.values()):
            runner.cancel()
        for queued in self._device_queued_ids.values():
            queued.clear()

        workers: List[asyncio.Task[None]] = []
        for device_workers in self._device_workers.values():
            workers.extend(w for w in device_workers if not w.done())
        self._device_workers.clear()
        self._started = False
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.wait(workers, timeout=2.0)

    def enqueue(
        self,
        task_id: str,
        runner: Callable[[], Awaitable[None]],
        device: str | None = None,
    ) -> None:
        """Add a previously created task to the queue lane for its device."""
        if task_id not in self._tasks:
            raise KeyError(f"Unknown task: {task_id}")
        task = self._tasks[task_id]
        metadata = dict(task.get("metadata") or {})
        requested = device if device is not None else metadata.get("device")
        # CPU-only work (YouTube crawl, etc.) stays off the GPU lanes.
        if task.get("type") == "youtube_crawl":
            lane = "cpu"
        else:
            lane = normalize_queue_device(requested if requested is not None else "auto")
        metadata["device"] = lane
        metadata["queue_device"] = lane
        task["metadata"] = metadata
        task["queue_device"] = lane

        self._ensure_device_lane(lane)
        self._device_queued_ids[lane].append(task_id)
        self._device_queues[lane].put_nowait((task_id, runner))
        self._refresh_queue_messages(lane)

    async def _worker_loop(self, device: str) -> None:
        queue = self._device_queues[device]
        while True:
            task_id, runner = await queue.get()
            try:
                queued_ids = self._device_queued_ids[device]
                if task_id in queued_ids:
                    queued_ids.remove(task_id)
                task = self._tasks.get(task_id)
                if not task or task["status"] == "cancelled":
                    continue

                self._device_running_ids[device].add(task_id)
                self.update_task(task_id, status="running", message=f"Starting on {device}...")
                runner_task = asyncio.create_task(runner())
                self._runner_tasks[task_id] = runner_task
                try:
                    await asyncio.wait({runner_task})
                    if runner_task.cancelled():
                        task = self._tasks.get(task_id)
                        if task and task["status"] == "running":
                            self.update_task(
                                task_id,
                                status="cancelled",
                                message="Task interrupted by server shutdown.",
                            )
                    else:
                        exc = runner_task.exception()
                        if isinstance(exc, asyncio.CancelledError):
                            task = self._tasks.get(task_id)
                            if task and task["status"] == "running":
                                self.update_task(
                                    task_id,
                                    status="cancelled",
                                    message="Task interrupted by server shutdown.",
                                )
                        elif exc is not None:
                            raise exc
                        else:
                            task = self._tasks.get(task_id)
                            if task and task["status"] == "running":
                                self.update_task(
                                    task_id,
                                    status="completed",
                                    progress=1.0,
                                    progress_known=True,
                                )
                finally:
                    self._runner_tasks.pop(task_id, None)
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
                self._cancel_callbacks.pop(task_id, None)
                self._device_running_ids[device].discard(task_id)
                queue.task_done()
                self._refresh_queue_messages(device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _refresh_queue_messages(self, device: str | None = None) -> None:
        devices = [device] if device else list(self._device_queued_ids)
        for lane in devices:
            for position, task_id in enumerate(self._device_queued_ids.get(lane, []), start=1):
                task = self._tasks.get(task_id)
                if task and task["status"] == "pending":
                    task["queue_position"] = position
                    task["queue_device"] = lane
                    task["message"] = f"Queued on {lane} — position {position}"

    def create_task(self, task_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        meta = dict(metadata or {})
        if task_type == "youtube_crawl":
            meta.setdefault("device", "cpu")
        self._tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "progress": 0.0,
            "progress_known": False,
            "message": "Task queued...",
            "error": None,
            "result": None,
            "created_at": time.time(),
            "start_time": None,
            "end_time": None,
            "queue_position": None,
            "queue_device": None,
            "metadata": meta,
        }
        return task_id

    def update_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        progress_known: Optional[bool] = None,
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
            if progress_known is not None:
                t["progress_known"] = progress_known
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

    def set_cancel_callback(
        self,
        task_id: str,
        callback: Callable[[], None] | None,
    ) -> None:
        """Register or clear safe cancellation for a running task."""
        if callback is None:
            self._cancel_callbacks.pop(task_id, None)
        else:
            self._cancel_callbacks[task_id] = callback

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or running Studio task."""
        task = self._tasks.get(task_id)
        if not task or task["status"] not in ("pending", "running"):
            return False
        lane = task.get("queue_device") or (task.get("metadata") or {}).get("queue_device")
        if task["status"] == "pending":
            if lane and task_id in self._device_queued_ids.get(lane, []):
                self._device_queued_ids[lane].remove(task_id)
            self.update_task(task_id, status="cancelled", message="Cancelled while queued.")
        else:
            self.update_task(task_id, status="cancelled", message="Stopping running task...")
            callback = self._cancel_callbacks.get(task_id)
            if callback is not None:
                try:
                    callback()
                except Exception:
                    logger.exception("Studio cancel callback failed for %s", task_id)
            runner = self._runner_tasks.get(task_id)
            if runner is not None and not runner.done():
                runner.cancel()
        if lane:
            self._refresh_queue_messages(lane)
        else:
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
        """Return aggregate and per-device queue status."""
        device_queues: Dict[str, Dict[str, Any]] = {}
        total_running = 0
        total_queued = 0
        for device, running_ids in self._device_running_ids.items():
            queued = len(self._device_queued_ids.get(device, []))
            running = len(running_ids)
            total_running += running
            total_queued += queued
            device_queues[device] = {
                "device": device,
                "running": running,
                "queued": queued,
                "workers": self.workers_per_device,
            }
        return {
            "max_concurrency": self.workers_per_device,
            "workers_per_device": self.workers_per_device,
            "running": total_running,
            "queued": total_queued,
            "device_queues": device_queues,
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
                data = resolve_data_payload(
                    json.loads(self.file_path.read_text(encoding="utf-8"))
                )
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
                json.dumps(
                    portable_data_payload(list(self.evaluations.values())),
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
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
            "evaluation_type": str(eval_data.get("evaluation_type", "separation")),
            "channel_id": eval_data.get("channel_id"),
            "channel_name": eval_data.get("channel_name"),
            "profile_name": eval_data.get("profile_name"),
            "threshold": eval_data.get("threshold"),
            "min_duration_s": eval_data.get("min_duration_s"),
            "exclude_overlap": eval_data.get("exclude_overlap"),
            "qualified_segments": eval_data.get("qualified_segments"),
            "reviewed_segments": eval_data.get("reviewed_segments"),
            "total_segments": eval_data.get("total_segments"),
            "qualified_duration_s": eval_data.get("qualified_duration_s"),
            "total_duration_s": eval_data.get("total_duration_s"),
            "qualified_percent": eval_data.get("qualified_percent"),
            "segment_labels": dict(eval_data.get("segment_labels", {})),
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
task_manager = TaskManager(workers_per_device=STUDIO_QUEUE_CONCURRENCY)
evaluation_manager = EvaluationManager(DATA_DIR / "studio" / "evaluations.json")


def _json_response(payload: Any, status: int = 200) -> web.Response:
    """Serialize JSON without NaN/Infinity so browsers can parse the body."""
    return web.Response(
        text=json.dumps(payload, ensure_ascii=False, allow_nan=False),
        status=status,
        content_type="application/json",
        charset="utf-8",
        headers={"Cache-Control": "no-store"},
    )


def _source_audio_path(result: DiarizationResult) -> Path | None:
    """Return the resolved source audio path when the snapshot exists."""
    if result.source_audio is None:
        return None
    path = Path(result.source_audio.path).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    try:
        return path.resolve()
    except OSError:
        return None


def _session_audio_id_for(result: DiarizationResult) -> str | None:
    """Return the live session ID for a result's source file, if registered."""
    path = _source_audio_path(result)
    if path is None:
        return None
    return registry.find_id_by_path(path)


def _ensure_source_registered(result: DiarizationResult) -> str | None:
    """Register a result's source audio into the session when the file exists."""
    path = _source_audio_path(result)
    if path is None or not path.is_file() or result.source_audio is None:
        return None
    existing = registry.find_id_by_path(path)
    if existing:
        return existing
    source = result.source_audio
    if Path(source.path).resolve() != path:
        source = Audio.from_file(
            path,
            source_id=source.source_id,
            title=source.title,
            source_url=source.source_url,
            channel_id=source.channel_id,
            channel_name=source.channel_name,
            channel_url=source.channel_url,
            native_sample_rate=source.native_sample_rate,
            history=source.history,
        )
    return registry.register(
        source,
        source_type="library",
        tags=["library", "diarization_source"],
    )


def _diarization_result_path(result_id: str) -> Path:
    """Resolve a result ID without allowing directory traversal."""
    if not result_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in result_id):
        raise ValueError("Invalid diarization result ID")
    return DIARIZATION_RESULTS_DIR / f"{result_id}.json"


def _load_diarization_result(result_id: str) -> DiarizationResult:
    path = _diarization_result_path(result_id)
    if not path.is_file():
        raise FileNotFoundError(f"Diarization result not found: {result_id}")
    result = DiarizationResult.load(path)
    if result.result_id != result_id:
        raise ValueError("Diarization result ID does not match its filename")
    return result


def _parse_overlap_verifier_request(
    overlap_settings: Any,
) -> tuple[dict[str, Any], str]:
    """Parse the required LLM ``overlap_verifier`` request object.

    Returns:
        ``(config, failure_policy)``.

    Raises:
        TypeError: Numeric fields cannot be coerced.
        ValueError: Payload shape is invalid, or the request tried to
            disable the LLM verifier.
    """
    if overlap_settings is None:
        overlap_settings = {}
    if not isinstance(overlap_settings, dict):
        raise ValueError("overlap_verifier must be an object")
    failure_policy = str(overlap_settings.get("failure_policy", "fail_closed"))
    if failure_policy not in {"fail_closed", "fail_open"}:
        raise ValueError("Invalid overlap verifier failure_policy")
    if overlap_settings.get("enabled") is False:
        raise ValueError(
            "Speaker Purity uses the LLM verifier only; speaker embeddings "
            "are not used on this tab"
        )
    backend = str(overlap_settings.get("backend", "gemma4")).strip().lower()
    backend = backend.replace("_", "-")
    if backend in {"vibevoice", "vibevoice-asr"}:
        batch_size = int(
            overlap_settings.get("batch_size", DEFAULT_VIBEVOICE_BATCH_SIZE)
        )
        if batch_size < 1 or batch_size > 256:
            raise ValueError("batch_size must be an integer from 1 to 256")
        return {
            "backend": "vibevoice",
            "model": overlap_settings.get("model") or DEFAULT_VIBEVOICE_MODEL_ID,
            "min_secondary_speech_s": float(
                overlap_settings.get(
                    "min_secondary_speech_s", DEFAULT_MIN_SECONDARY_SPEECH_S
                )
            ),
            "max_new_tokens": int(
                overlap_settings.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
            ),
            "batch_size": batch_size,
        }, failure_policy
    config: dict[str, Any] = {
        "backend": overlap_settings.get("backend", "gemma4"),
        "model": overlap_settings.get("model") or None,
        "api_key": overlap_settings.get("api_key") or None,
        "timeout_s": float(overlap_settings.get("timeout_s", 120.0)),
        "prompt": overlap_settings.get("prompt") or OVERLAP_PROMPT,
        "max_output_tokens": int(
            overlap_settings.get(
                "max_output_tokens", DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS
            )
        ),
    }
    if str(config["backend"]).strip().lower() in {
        "gemma",
        "gemma4",
        "gemma-4",
        "unsloth",
    }:
        config["endpoint"] = overlap_settings.get("endpoint") or None
    config = {key: value for key, value in config.items() if value is not None}
    create_overlap_verifier(config)
    return config, failure_policy


def _audio_duration_s(audio: Audio) -> float:
    """Return a positive duration for ``audio``, probing the file if needed."""
    if audio.duration_s is not None:
        duration_s = float(audio.duration_s)
        if isfinite(duration_s) and duration_s > 0:
            return duration_s
    path = Path(audio.path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file is missing: {path}")
    duration_s = float(sf.info(str(path)).duration)
    if not isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"Audio has no duration: {path}")
    return duration_s


def _direct_audio_purity_item(
    audio: Audio,
    diarization: DiarizationResult,
    turn: SpeakerTurn,
    profile_name: str,
) -> dict[str, Any]:
    """Build a purity-report row for one turn judged by the direct-audio verifier."""
    duration_s = turn.end_s - turn.start_s
    overlap_duration_s = SpeakerVerifier._other_speaker_overlap_duration(
        diarization,
        speaker_id=turn.speaker_id,
        start_s=turn.start_s,
        end_s=turn.end_s,
    )
    return {
        "schema_version": "1.0",
        "audio_id": audio.source_id,
        "profile_name": profile_name,
        "speaker_id": turn.speaker_id,
        "start_s": turn.start_s,
        "end_s": turn.end_s,
        "decision": "pass",
        "reason": None,
        "error": None,
        "overlap_duration_s": overlap_duration_s,
        "overlap_ratio": (
            min(1.0, overlap_duration_s / duration_s) if duration_s > 0 else 0.0
        ),
        "windows": [],
        "model": None,
        "duration_s": duration_s,
        "min_target_similarity": None,
        "direct_overlap": None,
        "passed": True,
    }


def _apply_direct_overlap_decision(
    item: dict[str, Any],
    *,
    backend: str,
    model: str,
    overlap: bool | None,
    reason: str | None,
    error: str | None,
    failure_policy: str,
) -> None:
    """Record a direct-audio answer onto a purity-report row."""
    item["direct_overlap"] = {
        "backend": backend,
        "model": model,
        "overlap": overlap,
        "reason": reason,
        "error": error,
    }
    if error:
        item["error"] = error
        if failure_policy == "fail_closed":
            item["decision"] = "error"
            item["reason"] = "direct_overlap_verification_failed"
            item["passed"] = False
        return
    if overlap:
        item["decision"] = "reject"
        item["reason"] = "direct_overlap_detected"
        item["passed"] = False


def _overlap_verifier_report_settings(
    overlap_config: dict[str, Any] | None,
    failure_policy: str,
    verifier: Any = None,
) -> dict[str, Any]:
    if not overlap_config:
        return {"enabled": False}
    if str(overlap_config.get("backend")) == "vibevoice":
        return {
            "enabled": True,
            "backend": "vibevoice",
            "model": overlap_config.get("model") or DEFAULT_VIBEVOICE_MODEL_ID,
            "min_secondary_speech_s": overlap_config.get("min_secondary_speech_s"),
            "max_new_tokens": overlap_config.get("max_new_tokens"),
            "failure_policy": failure_policy,
        }
    return {
        "enabled": True,
        "backend": overlap_config["backend"],
        "model": getattr(verifier, "model", overlap_config.get("model")),
        "endpoint": getattr(verifier, "endpoint", overlap_config.get("endpoint")),
        "timeout_s": overlap_config["timeout_s"],
        "prompt": overlap_config["prompt"],
        "max_output_tokens": overlap_config["max_output_tokens"],
        "failure_policy": failure_policy,
        "api_key_configured": bool(
            overlap_config.get("api_key") or getattr(verifier, "api_key", None)
        ),
    }


def _probe_overlap_verifier_or_raise(verifier: Any, backend: str) -> dict[str, Any]:
    """Fail the job before any candidate if the LLM backend is not ready."""
    check = getattr(verifier, "check_ready", None)
    if check is None:
        return {
            "ready": True,
            "message": f"{backend} has no readiness probe.",
            "models": [],
        }
    status = check()
    if not status.get("ready"):
        raise OverlapVerifierError(
            str(status.get("message") or f"{backend} is not ready"),
            readiness=True,
        )
    return status


def _verifier_error_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect request-error counts and sample messages for the report."""
    errors = []
    for item in items:
        direct = item.get("direct_overlap") or {}
        message = item.get("error") or direct.get("error")
        if message:
            errors.append(str(message))
    unique = list(dict.fromkeys(errors))
    return {
        "error_count": len(errors),
        "error_samples": unique[:5],
    }


def _vibevoice_verifier_from_config(
    overlap_config: dict[str, Any],
    *,
    device: str,
    token: str | None,
) -> VibeVoicePurityWorkerVerifier:
    return VibeVoicePurityWorkerVerifier(
        model_id=str(overlap_config.get("model") or DEFAULT_VIBEVOICE_MODEL_ID),
        device=device,
        token=token,
        min_secondary_speech_s=float(
            overlap_config.get(
                "min_secondary_speech_s", DEFAULT_MIN_SECONDARY_SPEECH_S
            )
        ),
        max_new_tokens=int(
            overlap_config.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        ),
        batch_size=int(
            overlap_config.get("batch_size", DEFAULT_VIBEVOICE_BATCH_SIZE)
        ),
        attn_implementation="eager",
    )


def _apply_vibevoice_purity_item(item: dict[str, Any], result: Any) -> None:
    """Copy a VibeVoice speaker-count decision onto a purity-report row."""
    item["decision"] = result.decision
    item["reason"] = None if result.decision == "pass" else result.reason
    item["passed"] = result.passed
    item["error"] = result.error
    item["windows"] = []
    item["min_target_similarity"] = None
    item["vibevoice"] = {
        "num_speakers": result.num_speakers,
        "dominant_speaker_id": result.dominant_speaker_id,
        "secondary_speech_s": result.secondary_speech_s,
        "reason": result.reason,
        "speaker_turns": [
            {
                "start_s": turn.start_s,
                "end_s": turn.end_s,
                "speaker_id": turn.speaker_id,
            }
            for turn in result.speaker_turns
        ],
    }


def _annotation_path(annotation_id: str) -> Path:
    """Resolve a manual annotation ID without allowing directory traversal."""
    if not annotation_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in annotation_id
    ):
        raise ValueError("Invalid annotation ID")
    return DIARIZATION_ANNOTATIONS_DIR / f"{annotation_id}.json"


def _load_annotation(annotation_id: str) -> dict[str, Any]:
    """Load and validate the outer shape of one manual annotation."""
    path = _annotation_path(annotation_id)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation not found: {annotation_id}")
    payload = resolve_data_payload(json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict) or payload.get("annotation_id") != annotation_id:
        raise ValueError("Annotation ID does not match its filename")
    return payload


def _annotation_session_audio_id(annotation: dict[str, Any]) -> str | None:
    """Find the live session ID for an annotation source, if registered."""
    source = annotation.get("source_audio") or {}
    raw_path = source.get("path")
    if not raw_path:
        return None
    return registry.find_id_by_path(raw_path)


def _ensure_annotation_source_registered(annotation: dict[str, Any]) -> str | None:
    """Restore an annotation source into the Studio registry when available."""
    existing = _annotation_session_audio_id(annotation)
    if existing:
        return existing
    source = annotation.get("source_audio") or {}
    raw_path = source.get("path")
    if not raw_path:
        return None
    path = resolve_data_path(str(raw_path))
    try:
        path = path.resolve()
    except OSError:
        return None
    if not path.is_file():
        return None
    audio = Audio.from_file(
        path,
        source_id=str(annotation.get("audio_id") or path.stem),
        title=source.get("title"),
    )
    return registry.register(
        audio,
        source_type="library",
        tags=["library", "diarization_annotation_source"],
    )


def _annotation_summary(annotation: dict[str, Any]) -> dict[str, Any]:
    """Return catalog fields without the full turn payload."""
    speakers = annotation.get("speakers") or []
    turns = annotation.get("turns") or []
    return {
        "kind": annotation.get("kind"),
        "schema_version": annotation.get("schema_version"),
        "annotation_id": annotation.get("annotation_id"),
        "revision": annotation.get("revision"),
        "created_at": annotation.get("created_at"),
        "updated_at": annotation.get("updated_at"),
        "name": annotation.get("name"),
        "audio_id": annotation.get("audio_id"),
        "session_audio_id": _annotation_session_audio_id(annotation),
        "source_audio": annotation.get("source_audio"),
        "seed": annotation.get("seed"),
        "speaker_count": len(speakers),
        "turn_count": len(turns),
        "speech_duration_s": round(
            sum(float(turn["end_s"]) - float(turn["start_s"]) for turn in turns),
            6,
        ),
    }


def _validated_annotation_payload(
    data: dict[str, Any],
    *,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize a manual ground-truth annotation request."""
    if not isinstance(data, dict):
        raise TypeError("Annotation payload must be an object")
    now = time.time()
    annotation_id = (
        str(existing["annotation_id"])
        if existing is not None
        else str(data.get("annotation_id") or f"ann_{uuid.uuid4().hex}")
    )
    _annotation_path(annotation_id)

    if existing is not None:
        requested_revision = data.get("revision")
        if isinstance(requested_revision, bool):
            raise TypeError("revision must be an integer")
        try:
            requested_revision = int(requested_revision)
        except (TypeError, ValueError) as exc:
            raise TypeError("revision must be an integer") from exc
        if requested_revision != int(existing.get("revision", 0)):
            raise RuntimeError(
                "This annotation changed in another browser tab. Reload it before saving."
            )
        source_audio = dict(existing.get("source_audio") or {})
        audio_id = str(existing.get("audio_id") or "")
        session_audio_id = _annotation_session_audio_id(existing)
    else:
        session_audio_id = str(data.get("session_audio_id") or "").strip()
        audio = registry.get_audio(session_audio_id)
        if audio is None:
            raise FileNotFoundError("Select a registered source audio before annotating")
        source_audio = {
            "path": str(Path(audio.path).resolve()),
            "fingerprint": audio.fingerprint,
            "title": audio.title,
            "duration_s": audio.duration_s,
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "format": audio.format,
        }
        audio_id = audio.source_id

    seed = existing.get("seed") if existing is not None else None
    if existing is None and data.get("seed_result_id"):
        seed_result_id = str(data.get("seed_result_id") or "").strip()
        seed_result = _load_diarization_result(seed_result_id)
        candidate_annotation = {
            "audio_id": audio_id,
            "source_audio": source_audio,
        }
        matches, reason = _annotation_matches_result(candidate_annotation, seed_result)
        if not matches:
            raise ValueError(
                f"Diarization result {seed_result_id} cannot seed this audio: {reason}"
            )
        model = asdict(seed_result.model) if seed_result.model else None
        model_name = (
            seed_result.model.model_id
            if seed_result.model is not None
            else "Unknown model"
        )
        colors = (
            "#168aad", "#2f9e6f", "#c98200", "#dc3656", "#805ad5",
            "#2574c8", "#65a30d", "#d95f20", "#0891b2", "#be185d",
        )
        data = dict(data)
        data["name"] = data.get("name") or (
            f"{source_audio.get('title') or audio_id} — {model_name} assisted reference"
        )
        data["speakers"] = [
            {
                "speaker_id": speaker.speaker_id,
                "name": speaker.global_speaker_id or f"Speaker {index + 1}",
                "color": colors[index % len(colors)],
                "global_speaker_id": speaker.global_speaker_id,
            }
            for index, speaker in enumerate(seed_result.speakers)
        ]
        merged_by_speaker: dict[str, list[dict[str, Any]]] = {}
        for turn in sorted(
            seed_result.turns,
            key=lambda item: (item.speaker_id, item.start_s, item.end_s),
        ):
            start_s = round(float(turn.start_s), 3)
            end_s = round(float(turn.end_s), 3)
            speaker_turns = merged_by_speaker.setdefault(turn.speaker_id, [])
            if speaker_turns and start_s < speaker_turns[-1]["end_s"]:
                speaker_turns[-1]["end_s"] = max(speaker_turns[-1]["end_s"], end_s)
            else:
                speaker_turns.append(
                    {
                        "turn_id": f"turn_{uuid.uuid4().hex}",
                        "speaker_id": turn.speaker_id,
                        "start_s": start_s,
                        "end_s": end_s,
                    }
                )
        data["turns"] = [
            turn for speaker_turns in merged_by_speaker.values() for turn in speaker_turns
        ]
        seed = {"result_id": seed_result_id, "model": model, "created_at": now}

    try:
        duration_s = float(source_audio.get("duration_s"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Source audio requires a known duration") from exc
    if not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Source audio requires a finite positive duration")

    raw_speakers = data.get("speakers")
    raw_turns = data.get("turns")
    if not isinstance(raw_speakers, list) or not isinstance(raw_turns, list):
        raise TypeError("speakers and turns must be arrays")
    speakers = []
    speaker_ids: set[str] = set()
    for index, raw_speaker in enumerate(raw_speakers):
        if not isinstance(raw_speaker, dict):
            raise TypeError("Every speaker must be an object")
        speaker_id = str(raw_speaker.get("speaker_id") or "").strip()
        if not speaker_id:
            raise ValueError(f"Speaker {index + 1} requires speaker_id")
        if speaker_id in speaker_ids:
            raise ValueError(f"Duplicate speaker_id: {speaker_id}")
        speaker_ids.add(speaker_id)
        name = str(raw_speaker.get("name") or speaker_id).strip()
        if not name:
            raise ValueError(f"Speaker {speaker_id} requires a display name")
        global_speaker_id = str(raw_speaker.get("global_speaker_id") or "").strip() or None
        speakers.append(
            {
                "speaker_id": speaker_id,
                "name": name[:120],
                "color": str(raw_speaker.get("color") or "#4f7cff")[:32],
                "global_speaker_id": global_speaker_id,
            }
        )

    turns = []
    turn_ids: set[str] = set()
    for index, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, dict):
            raise TypeError("Every turn must be an object")
        turn_id = str(raw_turn.get("turn_id") or f"turn_{uuid.uuid4().hex}").strip()
        speaker_id = str(raw_turn.get("speaker_id") or "").strip()
        if turn_id in turn_ids:
            raise ValueError(f"Duplicate turn_id: {turn_id}")
        if speaker_id not in speaker_ids:
            raise ValueError(f"Turn {index + 1} references unknown speaker: {speaker_id}")
        try:
            start_s = round(float(raw_turn.get("start_s")), 6)
            end_s = round(float(raw_turn.get("end_s")), 6)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Turn {index + 1} requires numeric timestamps") from exc
        if (
            not isfinite(start_s)
            or not isfinite(end_s)
            or start_s < 0
            or end_s <= start_s
            or end_s > duration_s + 0.001
        ):
            raise ValueError(
                f"Turn {index + 1} must satisfy 0 <= start_s < end_s <= {duration_s:.3f}"
            )
        turn_ids.add(turn_id)
        turns.append(
            {
                "turn_id": turn_id,
                "speaker_id": speaker_id,
                "start_s": start_s,
                "end_s": min(end_s, duration_s),
            }
        )
    turns.sort(key=lambda turn: (turn["start_s"], turn["end_s"], turn["speaker_id"]))
    previous_by_speaker: dict[str, dict[str, Any]] = {}
    for turn in turns:
        previous = previous_by_speaker.get(turn["speaker_id"])
        if previous is not None and turn["start_s"] < previous["end_s"] - 0.000001:
            raise ValueError(
                f"{turn['speaker_id']} has overlapping turns at "
                f"{turn['start_s']:.3f}s; use another speaker lane for simultaneous speech"
            )
        previous_by_speaker[turn["speaker_id"]] = turn

    name = str(data.get("name") or source_audio.get("title") or "Ground truth").strip()
    if not name:
        raise ValueError("Annotation name cannot be empty")
    payload = {
        "kind": "diarization.annotation",
        "schema_version": "1.0",
        "annotation_id": annotation_id,
        "revision": int(existing.get("revision", 0)) + 1 if existing else 1,
        "created_at": float(existing.get("created_at", now)) if existing else now,
        "updated_at": now,
        "name": name[:200],
        "audio_id": audio_id,
        "session_audio_id": session_audio_id,
        "source_audio": source_audio,
        "speakers": speakers,
        "turns": turns,
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


_CUT_SOURCE_ID_SUFFIX = re.compile(r"_\d+\.\d{3}-\d+\.\d{3}$")


def _audio_is_cut(
    *,
    audio_id: str | None = None,
    history: Any = None,
    fingerprint: str | None = None,
) -> bool:
    """Return True when identity describes an AudioCutter excerpt.

    Cuts occupy a different clock from the full-length YouTube mix and its
    stems, so they must not family-match a mixture annotation.
    """
    steps = history if isinstance(history, (list, tuple)) else ()
    if any(str(step).startswith("cut_") for step in steps):
        return True
    if fingerprint and "__cut_" in str(fingerprint):
        return True
    if audio_id and _CUT_SOURCE_ID_SUFFIX.search(str(audio_id)):
        return True
    return False


def _annotation_matches_result(
    annotation: dict[str, Any], result: DiarizationResult
) -> tuple[bool, str]:
    """Check that reference and hypothesis share a scoring timeline.

    Exact fingerprint or resolved path still wins. When those differ, a
    full-length derivative (separator stem) matches if it keeps the same
    ``audio_id`` and duration. Cuts are excluded from that family rule.
    """
    source = annotation.get("source_audio") or {}
    result_source = result.source_audio
    if result_source is None:
        return False, "Model result has no source-audio identity"
    annotation_fingerprint = str(source.get("fingerprint") or "")
    result_fingerprint = str(result_source.fingerprint or "")
    if (
        annotation_fingerprint
        and result_fingerprint
        and annotation_fingerprint == result_fingerprint
    ):
        return True, "Audio fingerprints match"
    try:
        annotation_path = Path(str(source.get("path") or "")).expanduser().resolve()
        result_path = Path(result_source.path).expanduser().resolve()
        if annotation_path == result_path:
            return True, "Source paths match"
    except OSError:
        pass
    annotation_duration = source.get("duration_s")
    result_duration = result_source.duration_s
    annotation_is_cut = _audio_is_cut(
        audio_id=str(annotation.get("audio_id") or ""),
        history=source.get("history"),
        fingerprint=annotation_fingerprint,
    )
    result_is_cut = _audio_is_cut(
        audio_id=result.audio_id,
        history=result_source.history,
        fingerprint=result_fingerprint,
    )
    if (
        annotation.get("audio_id") == result.audio_id
        and annotation_duration is not None
        and result_duration is not None
        and abs(float(annotation_duration) - float(result_duration)) <= 0.05
        and not annotation_is_cut
        and not result_is_cut
    ):
        return True, "Same-timeline source identity and duration match"
    if annotation_fingerprint and result_fingerprint:
        return False, "Audio fingerprints differ and timelines are not the same"
    return False, "Result was produced from different audio"


def _turn_key(speaker_id: str, start_s: float, end_s: float) -> str:
    return f"{speaker_id}|{float(start_s):.6f}|{float(end_s):.6f}"


def _validated_float(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    """Parse one finite numeric request setting within inclusive bounds."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number") from exc
    if not isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return parsed


def _sortformer_settings(data: dict[str, Any]) -> dict[str, float]:
    """Read Sortformer boundary-detection settings from an API request."""
    settings = {
        "onset": _validated_float(
            data.get("sortformer_onset", DEFAULT_SORTFORMER_ONSET),
            "sortformer_onset",
            minimum=0.0,
            maximum=1.0,
        ),
        "offset": _validated_float(
            data.get("sortformer_offset", DEFAULT_SORTFORMER_OFFSET),
            "sortformer_offset",
            minimum=0.0,
            maximum=1.0,
        ),
        "pad_onset_s": _validated_float(
            data.get("sortformer_pad_onset_s", DEFAULT_SORTFORMER_PAD_ONSET_S),
            "sortformer_pad_onset_s",
            minimum=0.0,
        ),
        "pad_offset_s": _validated_float(
            data.get("sortformer_pad_offset_s", DEFAULT_SORTFORMER_PAD_OFFSET_S),
            "sortformer_pad_offset_s",
            minimum=0.0,
        ),
    }
    if settings["onset"] < settings["offset"]:
        raise ValueError(
            "sortformer_onset must be greater than or equal to "
            "sortformer_offset"
        )
    return settings


def _validated_bool(value: Any, name: str, *, default: bool = False) -> bool:
    """Parse one optional boolean request setting."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a boolean")


def _extraction_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Read optional stem-export post-processing. Default is raw labeled turns."""
    settings = data.get("extraction_settings") or {}
    if not isinstance(settings, dict):
        raise TypeError("extraction_settings must be an object")
    add_extra = _validated_bool(
        settings.get("add_extra"),
        "extraction_settings.add_extra",
        default=False,
    )
    stop_at_other_speakers = _validated_bool(
        settings.get("stop_at_other_speakers"),
        "extraction_settings.stop_at_other_speakers",
        default=False,
    )
    pre_roll_s = _validated_float(
        settings.get("pre_roll_s", DEFAULT_EXTRACTION_PRE_ROLL_S),
        "extraction_settings.pre_roll_s",
        minimum=0.0,
    )
    post_roll_s = _validated_float(
        settings.get("post_roll_s", DEFAULT_EXTRACTION_POST_ROLL_S),
        "extraction_settings.post_roll_s",
        minimum=0.0,
    )
    if not add_extra:
        pre_roll_s = 0.0
        post_roll_s = 0.0
        stop_at_other_speakers = False
    return {
        "add_extra": add_extra,
        "stop_at_other_speakers": stop_at_other_speakers,
        "pre_roll_s": pre_roll_s,
        "post_roll_s": post_roll_s,
    }


def _turn_time_windows(turns: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Collect finite ``[start, end)`` pairs from turn-like JSON objects."""
    windows: list[tuple[float, float]] = []
    for turn in turns:
        try:
            start_s = float(turn.get("start_s", 0))
            end_s = float(turn.get("end_s", 0))
        except (TypeError, ValueError):
            continue
        if isfinite(start_s) and isfinite(end_s) and end_s > start_s:
            windows.append((start_s, end_s))
    return windows


def _request_blocker_turns(
    data: dict[str, Any],
    turns: list[dict[str, Any]],
    *,
    exclude_speaker_id: str | None,
) -> list[dict[str, Any]]:
    """Other-speaker windows used when stopping extra at a neighbor."""
    raw = data.get("blocker_turns")
    if isinstance(raw, list) and raw:
        return [item for item in raw if isinstance(item, dict)]
    if exclude_speaker_id is None:
        return []
    return [turn for turn in turns if turn.get("speaker_id") != exclude_speaker_id]


def _padded_audio_intervals(
    turns: list[dict[str, Any]],
    *,
    sample_rate: int,
    total_frames: int,
    pre_roll_s: float,
    post_roll_s: float,
    add_extra: bool = False,
    stop_at_other_speakers: bool = False,
    blocker_turns: list[dict[str, Any]] | None = None,
) -> list[tuple[int, int]]:
    """Resolve optional extra around turns and merge intervals that now overlap."""
    duration_s = total_frames / sample_rate if sample_rate else 0.0
    blockers = (
        _turn_time_windows(blocker_turns or [])
        if add_extra and stop_at_other_speakers
        else None
    )
    windows = pad_and_merge_intervals(
        _turn_time_windows(turns),
        pre_roll_s=pre_roll_s if add_extra else 0.0,
        post_roll_s=post_roll_s if add_extra else 0.0,
        end_bound_s=duration_s,
        blocker_intervals=blockers,
    )
    intervals: list[tuple[int, int]] = []
    for start_s, end_s in windows:
        start_frame = max(0, min(int(round(start_s * sample_rate)), total_frames))
        end_frame = max(0, min(int(round(end_s * sample_rate)), total_frames))
        if end_frame > start_frame:
            if intervals and start_frame <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end_frame))
            else:
                intervals.append((start_frame, end_frame))
    return intervals


def _speaker_turns_from_payload(raw_turns: Any) -> list[SpeakerTurn]:
    """Parse JSON turn objects without accepting frontend-only fields."""
    if not isinstance(raw_turns, list):
        raise TypeError("turns must be a list")
    turns: list[SpeakerTurn] = []
    for raw_turn in raw_turns:
        if not isinstance(raw_turn, dict):
            raise TypeError("each turn must be an object")
        confidence = raw_turn.get("confidence")
        turns.append(
            SpeakerTurn(
                speaker_id=str(raw_turn.get("speaker_id") or ""),
                start_s=float(raw_turn.get("start_s", 0)),
                end_s=float(raw_turn.get("end_s", 0)),
                confidence=(float(confidence) if confidence is not None else None),
                overlaps_other_speaker=bool(
                    raw_turn.get(
                        "overlaps_other_speaker",
                        raw_turn.get("has_overlap", False),
                    )
                ),
            )
        )
    return turns


def _clean_turn_settings(data: dict[str, Any]) -> dict[str, float]:
    """Read optional clean-turn settings from an API request.

    Studio listen/export defaults the boundary collar to zero so cleanup does
    not clip syllables. The library default of 40 ms remains available for
    high-purity identity clips when a client sends it explicitly.
    """
    settings = data.get("settings") or {}
    if not isinstance(settings, dict):
        raise TypeError("settings must be an object")
    return {
        "min_turn_duration_s": _validated_float(
            settings.get("min_turn_duration_s", DEFAULT_MIN_TURN_DURATION_S),
            "settings.min_turn_duration_s",
            minimum=0.0,
        ),
        "merge_same_speaker_gap_s": _validated_float(
            settings.get(
                "merge_same_speaker_gap_s",
                DEFAULT_MERGE_SAME_SPEAKER_GAP_S,
            ),
            "settings.merge_same_speaker_gap_s",
            minimum=0.0,
        ),
        "boundary_collar_s": _validated_float(
            settings.get("boundary_collar_s", 0.0),
            "settings.boundary_collar_s",
            minimum=0.0,
        ),
        "jitter_max_duration_s": _validated_float(
            settings.get("jitter_max_duration_s", DEFAULT_JITTER_MAX_DURATION_S),
            "settings.jitter_max_duration_s",
            minimum=0.0,
        ),
    }


def _clean_turn_payloads(
    raw_turns: Any,
    settings: dict[str, float],
) -> list[dict[str, Any]]:
    turns = clean_speaker_turns(_speaker_turns_from_payload(raw_turns), **settings)
    return [{**asdict(turn), "duration_s": turn.duration_s} for turn in turns]


def _verification_state_index(profile_name: str | None = None) -> dict[str, dict[str, Any]]:
    """Read the latest persisted decision for each result and turn."""
    by_result: dict[str, dict[str, Any]] = {}
    reports = sorted(
        DIARIZATION_VERIFICATIONS_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    for path in reports:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if profile_name is not None and report.get("profile") != profile_name:
            continue
        for item in report.get("results", []):
            result_id = item.get("result_id")
            if not result_id:
                continue
            state = by_result.setdefault(
                str(result_id), {"state": "unverified", "turns": {}, "report_id": None}
            )
            decision = str(item.get("decision", "error"))
            state["turns"][_turn_key(item.get("speaker_id", ""), item.get("start_s", 0), item.get("end_s", 0))] = decision
            state["report_id"] = report.get("verification_id")
    for state in by_result.values():
        decisions = set(state["turns"].values())
        if "error" in decisions:
            state["state"] = "error"
        elif "reject" in decisions:
            state["state"] = "rejected"
        elif "pass" in decisions:
            state["state"] = "passed"
    return by_result


def _result_catalog_item(
    result: DiarizationResult,
    verification: dict[str, Any] | None = None,
    *,
    complete: bool = False,
    hydrate_source: bool = False,
) -> dict[str, Any]:
    """Build a JSON-safe catalog entry for a durable diarization result.

    The list endpoint returns summaries only. The single-result endpoint
    returns the complete canonical payload plus session hydration fields.
    """
    session_audio_id = None
    if hydrate_source:
        try:
            session_audio_id = _ensure_source_registered(result)
        except Exception as exc:
            logger.warning(
                "Could not re-register diarization source for %s: %s",
                result.result_id,
                exc,
            )
            session_audio_id = _session_audio_id_for(result)
    else:
        session_audio_id = _session_audio_id_for(result)
    source_path = _source_audio_path(result)
    extras = {
        "verification": verification or {
            "state": "unverified", "turns": {}, "report_id": None
        },
        "source_available": bool(source_path and source_path.is_file()),
        "session_audio_id": session_audio_id,
    }
    if complete:
        payload = result.to_dict()
        if payload.get("source_audio") is not None and result.source_audio is not None:
            try:
                payload["source_audio"]["fingerprint"] = result.source_audio.fingerprint
            except Exception:
                payload["source_audio"]["fingerprint"] = None
        payload.update(extras)
        return payload
    source = result.source_audio.metadata() if result.source_audio else None
    if source is not None:
        try:
            source["fingerprint"] = result.source_audio.fingerprint
        except Exception:
            source["fingerprint"] = None
    return {
        "kind": DIARIZATION_RESULT_KIND,
        "schema_version": result.schema_version,
        "result_id": result.result_id,
        "created_at": result.created_at,
        "audio_id": result.audio_id,
        "source_audio": source,
        "speakers": [asdict(speaker) for speaker in result.speakers],
        "model": asdict(result.model) if result.model else None,
        "channel_id": result.channel_id,
        "channel_name": result.channel_name,
        "channel_url": result.channel_url,
        "summary": {
            "speaker_count": result.speaker_count,
            "turn_count": result.turn_count,
            "total_speech_duration_s": result.total_speech_duration_s,
            "duration_per_speaker_s": result.duration_per_speaker_s,
        },
        **extras,
    }


def _raw_result_catalog_item(
    path: Path,
    verification: dict[str, Any] | None = None,
    *,
    complete: bool = False,
) -> dict[str, Any]:
    """Build a catalog entry from raw JSON when schema load fails."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Diarization result is not an object")
    result_id = str(data.get("result_id") or path.stem)
    source = data.get("source_audio") if isinstance(data.get("source_audio"), dict) else None
    speakers = data.get("speakers") if isinstance(data.get("speakers"), list) else []
    turns = data.get("turns") if isinstance(data.get("turns"), list) else []
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    source_path: Path | None = None
    if source and source.get("path"):
        try:
            source_path = Path(str(source["path"])).expanduser()
            if not source_path.is_absolute():
                source_path = ROOT_DIR / source_path
            source_path = source_path.resolve()
        except OSError:
            source_path = None
    extras = {
        "verification": verification or {
            "state": "unverified", "turns": {}, "report_id": None
        },
        "source_available": bool(source_path and source_path.is_file()),
        "session_audio_id": registry.find_id_by_path(source_path) if source_path else None,
    }
    if complete and not extras["session_audio_id"] and extras["source_available"] and source_path:
        try:
            extras["session_audio_id"] = registry.register(
                Audio.from_file(source_path),
                source_type="library",
                tags=["library", "diarization_source"],
            )
        except Exception as exc:
            logger.warning("Could not re-register raw diarization source %s: %s", path, exc)
    if complete:
        payload = dict(data)
        payload.update(extras)
        return payload
    return {
        "kind": data.get("kind") or DIARIZATION_RESULT_KIND,
        "schema_version": data.get("schema_version") or "1.0",
        "result_id": result_id,
        "created_at": data.get("created_at") or 0,
        "audio_id": data.get("audio_id") or (source or {}).get("source_id"),
        "source_audio": source,
        "speakers": speakers,
        "model": data.get("model"),
        "channel_id": data.get("channel_id"),
        "channel_name": data.get("channel_name"),
        "channel_url": data.get("channel_url"),
        "summary": {
            "speaker_count": summary.get("speaker_count", len(speakers)),
            "turn_count": summary.get("turn_count", len(turns)),
            "total_speech_duration_s": summary.get("total_speech_duration_s", 0),
            "duration_per_speaker_s": summary.get("duration_per_speaker_s") or {},
        },
        **extras,
    }


def _catalog_item_from_path(
    path: Path,
    verification: dict[str, dict[str, Any]],
    *,
    complete: bool = False,
    hydrate_source: bool = False,
) -> dict[str, Any]:
    """Load one result file as a catalog item, falling back to raw JSON."""
    try:
        result = DiarizationResult.load(path)
        item = _result_catalog_item(
            result,
            verification.get(result.result_id),
            complete=complete,
            hydrate_source=hydrate_source,
        )
        json.dumps(item, allow_nan=False)
        return item
    except Exception as exc:
        logger.warning("Could not load diarization result %s: %s", path, exc)
        item = _raw_result_catalog_item(
            path,
            verification.get(str(path.stem)),
            complete=complete,
        )
        json.dumps(item, allow_nan=False)
        return item


def _created_at_value(item: dict[str, Any]) -> float:
    try:
        return float(item.get("created_at") or 0)
    except (TypeError, ValueError):
        return 0.0


def extract_waveform_envelope(
    audio_path: Path,
    start_frame: int,
    end_frame: int,
    bins: int,
) -> Dict[str, Any]:
    """Read a bounded window and return signed min/max envelopes per channel."""
    with sf.SoundFile(str(audio_path)) as source:
        sample_rate = int(source.samplerate)
        channel_count = int(source.channels)
        total_frames = int(source.frames)
        bounded_start = max(0, min(int(start_frame), total_frames))
        bounded_end = max(bounded_start, min(int(end_frame), total_frames))
        frame_count = bounded_end - bounded_start
        output_bins = min(int(bins), frame_count) if frame_count else 0
        channels = [
            {"min": [], "max": []}
            for _ in range(channel_count)
        ]

        if output_bins:
            boundaries = np.linspace(0, frame_count, output_bins + 1, dtype=np.int64)
            source.seek(bounded_start)
            for index in range(output_bins):
                frames_left = int(boundaries[index + 1] - boundaries[index])
                bin_min = np.full(channel_count, np.inf, dtype=np.float32)
                bin_max = np.full(channel_count, -np.inf, dtype=np.float32)
                while frames_left > 0:
                    block = source.read(
                        min(frames_left, 262_144),
                        dtype="float32",
                        always_2d=True,
                    )
                    if not len(block):
                        break
                    block = np.nan_to_num(block, nan=0.0, posinf=1.0, neginf=-1.0)
                    bin_min = np.minimum(bin_min, block.min(axis=0))
                    bin_max = np.maximum(bin_max, block.max(axis=0))
                    frames_left -= len(block)
                for channel_index in range(channel_count):
                    minimum = bin_min[channel_index]
                    maximum = bin_max[channel_index]
                    channels[channel_index]["min"].append(
                        float(minimum) if np.isfinite(minimum) else 0.0
                    )
                    channels[channel_index]["max"].append(
                        float(maximum) if np.isfinite(maximum) else 0.0
                    )

    return {
        "sample_rate": sample_rate,
        "duration_s": total_frames / sample_rate if sample_rate else 0.0,
        "total_frames": total_frames,
        "start_frame": bounded_start,
        "end_frame": bounded_end,
        "start_s": bounded_start / sample_rate if sample_rate else 0.0,
        "end_s": bounded_end / sample_rate if sample_rate else 0.0,
        "frame_count": frame_count,
        "channel_count": channel_count,
        "requested_bins": int(bins),
        "bins": output_bins,
        "channels": channels,
    }


# ==================== API HANDLERS ====================


async def handle_status(request: web.Request) -> web.Response:
    """Return system information and device status with shared GPU queue metrics."""
    info = get_system_device_info()
    info["registered_audios"] = len(registry.list_all())
    studio_q = task_manager.status()
    info["task_queue"] = studio_q
    try:
        from src.web_pipeline.queue_manager import queue_manager
        pipe_q = queue_manager.status()
        info["shared_queue"] = {
            "total_running": studio_q["running"] + pipe_q["running"],
            "total_queued": studio_q["queued"] + pipe_q["queued"],
            "studio_running": studio_q["running"],
            "studio_queued": studio_q["queued"],
            "pipeline_running": pipe_q["running"],
            "pipeline_queued": pipe_q["queued"],
            "device_queues": {
                **{
                    device: {
                        "device": device,
                        "running": int(lane.get("running", 0)),
                        "queued": int(lane.get("queued", 0)),
                        "workers": int(lane.get("workers", 1)),
                    }
                    for device, lane in (studio_q.get("device_queues") or {}).items()
                }
            },
        }
        # Merge pipeline lanes into the same map.
        merged = info["shared_queue"]["device_queues"]
        for device, lane in (pipe_q.get("device_queues") or {}).items():
            bucket = merged.setdefault(
                device,
                {"device": device, "running": 0, "queued": 0, "workers": int(lane.get("workers", 1))},
            )
            bucket["running"] += int(lane.get("running", 0))
            bucket["queued"] += int(lane.get("queued", 0))
            bucket["workers"] = max(int(bucket.get("workers", 1)), int(lane.get("workers", 1)))
    except Exception:
        info["shared_queue"] = {
            "total_running": studio_q["running"],
            "total_queued": studio_q["queued"],
            "studio_running": studio_q["running"],
            "studio_queued": studio_q["queued"],
            "pipeline_running": 0,
            "pipeline_queued": 0,
            "device_queues": studio_q.get("device_queues") or {},
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
                    "source_url": sidecar.get("source_url"),
                    "channel_id": sidecar.get("channel_id"),
                    "channel_name": sidecar.get("channel_name"),
                    "channel_url": sidecar.get("channel_url"),
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


def _session_library_tags(source_type: str) -> set[str]:
    """Return only tags implied by a session source type, without inventing ingest."""
    return {
        "separation": {"type:stem", "stage:separated"},
        "cut": {"type:cut"},
        "speaker_stem": {"type:cut", "stage:diarized"},
        "purity_stem": {"type:stem", "stage:verified", "verification:passed"},
        "youtube": {"type:source", "stage:ingested"},
        "upload": {"type:source", "stage:ingested"},
        "diarization": {"stage:diarized"},
    }.get(source_type, set())


def resolve_library_path(rel_or_abs: str | Path | None) -> Path | None:
    """Resolve a user-supplied path if it is inside a permitted library root."""
    if not rel_or_abs:
        return None
    target = Path(rel_or_abs)
    try:
        if not target.is_absolute():
            target = (ROOT_DIR / target).resolve()
        else:
            target = target.resolve()
    except OSError:
        return None
    if not any(target.is_relative_to(root) for root in LIBRARY_ALLOWED_ROOTS):
        return None
    return target


def should_skip_library_path(path: Path) -> bool:
    """Skip caches, VCS, and model-weight trees while scanning the library."""
    return any(part in LIBRARY_SKIP_DIR_NAMES for part in path.parts)


def library_relative_path(path: Path) -> str:
    """Return a POSIX relative path from the repo root when possible."""
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def categorize_library_path(rel_path: str) -> tuple[str, str]:
    """Fallback category id and label for files that have no structured metadata."""
    p_lower = rel_path.lower()
    if "sources/speech" in p_lower or ("speech" in p_lower and "music" not in p_lower and "cuts" not in p_lower):
        return "speech", "Benchmark Speech"
    if "sources/music" in p_lower or ("music" in p_lower and "speech" not in p_lower and "cuts" not in p_lower):
        return "music", "Benchmark Music"
    if "sources/cuts" in p_lower or "audio_cutter" in p_lower or "_cut_" in p_lower or "cuts" in p_lower:
        return "cuts", "Audio Cuts"
    if "stems" in p_lower or "separation" in p_lower or "demucs" in p_lower or "roformer" in p_lower or "mvsep" in p_lower:
        return "stems", "Separated Stems"
    if "yt_crawler" in p_lower or "downloads" in p_lower:
        return "ingest", "YouTube Downloads"
    if "pipeline" in p_lower:
        return "pipeline", "Pipeline Assets"
    if "temp" in p_lower or "quick_save" in p_lower:
        return "temp", "Quick Saves"
    if "upload" in p_lower:
        return "uploads", "Uploads"
    if "data" in p_lower:
        return "data", "Data Directory"
    return "other", "Project Audio"


def categorize_library_tags(system_tags: list[str], fallback_path: str) -> tuple[str, str]:
    """Classify a library asset from namespaced metadata before legacy paths."""
    tags = set(system_tags)
    if "type:cut" in tags:
        return "cuts", "Audio Cuts"
    if "stage:verified" in tags or "verification:passed" in tags:
        return "verified", "Verified Speech"
    if "type:stem" in tags or "stage:separated" in tags:
        return "stems", "Separated Stems"
    if "stage:diarized" in tags:
        return "diarized", "Diarized Sources"
    if "stage:ingested" in tags:
        return "ingest", "Ingested Sources"
    return categorize_library_path(fallback_path)


def ensure_library_dirs() -> None:
    """Create the directories the sample library is expected to browse."""
    (ROOT_DIR / "benchmarks/separation/sources/speech").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "benchmarks/separation/sources/music").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "benchmarks/separation/sources/cuts").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _unregister_pipeline_path(target: Path) -> None:
    """Drop a deleted file from the Pipeline dataset registry without deleting again."""
    try:
        from src.web_pipeline.dataset_manager import dataset_manager
        item = dataset_manager.find_item_by_path(target)
        if item:
            dataset_manager.delete_items([item.id], delete_files=False)
    except Exception:
        logger.debug("Pipeline registry skip for %s", target, exc_info=True)


def delete_library_file(target: Path) -> str:
    """Delete a library audio file, sidecar, and registry entries.

    Args:
        target: Resolved path already confirmed to be inside a library root.

    Returns:
        Repo-relative POSIX path of the deleted file.

    Raises:
        FileNotFoundError: If the audio file is not present.
        OSError: If the filesystem delete fails.
    """
    if not target.is_file():
        raise FileNotFoundError(target.name)
    target.unlink()
    sidecar = target.with_suffix(".json")
    if sidecar.is_file():
        sidecar.unlink()
    registry.unregister_path(target)
    _unregister_pipeline_path(target)
    rel_path = library_relative_path(target)
    logger.info("Deleted library file and sidecar: %s", target)
    return rel_path


def collect_library_files(
    pipeline_items: dict[str, Any],
    active_items: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scan permitted directories and return library file records."""
    scan_dirs = [
        ROOT_DIR / "benchmarks",
        ROOT_DIR / "data",
        TEMP_DIR,
        DATA_DIR,
    ]
    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for directory in scan_dirs:
        if not directory.is_dir():
            continue
        for root, dirnames, filenames in os.walk(directory, followlinks=False):
            dirnames[:] = [name for name in dirnames if name not in LIBRARY_SKIP_DIR_NAMES]
            root_path = Path(root)
            if should_skip_library_path(root_path):
                dirnames[:] = []
                continue
            for name in filenames:
                p = root_path / name
                if p.suffix.lower() not in LIBRARY_AUDIO_EXTENSIONS:
                    continue
                try:
                    resolved_str = str(p.resolve())
                    if resolved_str in seen_paths:
                        continue
                    seen_paths.add(resolved_str)

                    stat = p.stat()
                    if stat.st_size == 0:
                        continue

                    rel_path = library_relative_path(p)
                    probe_meta = probe_audio_file_info(p)
                    pipeline_item = pipeline_items.get(resolved_str)
                    active_item = active_items.get(resolved_str)
                    system_tags = list(pipeline_item.system_tags if pipeline_item else [])
                    custom_tags = list(pipeline_item.custom_tags if pipeline_item else [])
                    dataset = pipeline_item.dataset if pipeline_item else None
                    registry_item_id = pipeline_item.id if pipeline_item else None
                    if active_item:
                        source_type = str(active_item.get("source_type") or "local")
                        session_tags = set(active_item.get("system_tags", []))
                        if source_type not in {"youtube", "upload"}:
                            session_tags -= {"type:source", "stage:ingested"}
                        system_tags = sorted(
                            set(system_tags)
                            | session_tags
                            | _session_library_tags(source_type)
                        )
                        custom_tags = sorted(
                            set(custom_tags) | set(active_item.get("custom_tags", []))
                        )
                    history = probe_meta.get("history") or []
                    normalized_history = [str(step).lower() for step in history]
                    extra_tags: set[str] = set()
                    if any("diar" in step for step in normalized_history):
                        extra_tags.add("stage:diarized")
                    if any(
                        marker in step
                        for step in normalized_history
                        for marker in ("demucs", "roformer", "mvsep", "separ")
                    ):
                        extra_tags.update({"type:stem", "stage:separated"})
                    if any("cut" in step for step in normalized_history):
                        extra_tags.add("type:cut")
                    if any("purity" in step or "verified" in step for step in normalized_history):
                        extra_tags.add("stage:verified")
                    if extra_tags:
                        system_tags = sorted(set(system_tags) | extra_tags)
                    category_id, category = categorize_library_tags(system_tags, rel_path)
                    if registry_item_id and category_id == "other":
                        category_id, category = "pipeline", "Pipeline Assets"

                    files.append(
                        {
                            "category": category,
                            "category_id": category_id,
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
                            "source_url": probe_meta.get("source_url"),
                            "channel_id": probe_meta.get("channel_id"),
                            "channel_name": probe_meta.get("channel_name"),
                            "channel_url": probe_meta.get("channel_url"),
                            "history": history,
                            "dataset": dataset,
                            "registry_item_id": registry_item_id,
                            "system_tags": system_tags,
                            "custom_tags": custom_tags,
                        }
                    )
                except OSError:
                    logger.debug("Skipping unreadable library file %s", p, exc_info=True)

    files.sort(key=lambda item: item["modified"], reverse=True)
    return files


async def handle_list_library(request: web.Request) -> web.Response:
    """Scan and list audio files available in project directories with precise categorization and metadata."""
    ensure_library_dirs()
    try:
        from src.web_pipeline.dataset_manager import dataset_manager
        pipeline_items = dataset_manager.items_by_path()
    except Exception:
        pipeline_items = {}
    active_items = {
        str(Path(item["audio"].path).resolve()): item
        for item in registry._items.values()
    }
    loop = asyncio.get_running_loop()
    files = await loop.run_in_executor(
        None, collect_library_files, pipeline_items, active_items
    )
    category_counts: dict[str, int] = {key: 0 for key in LIBRARY_CATEGORY_ORDER}
    for item in files:
        category_counts[item["category_id"]] = category_counts.get(item["category_id"], 0) + 1
    return web.json_response({
        "files": files,
        "total": len(files),
        "category_counts": category_counts,
    })


async def handle_stream_library_file(request: web.Request) -> web.Response:
    """Stream any permissible library audio file directly for preview/playback."""
    target = resolve_library_path(request.query.get("path"))
    if target is None:
        return web.Response(text="Path is required", status=400) if not request.query.get("path") else web.Response(
            text="Audio file not found or access denied", status=404
        )
    if not target.is_file():
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
    target = resolve_library_path(request.query.get("path"))
    if target is None:
        return web.Response(text="Path is required", status=400) if not request.query.get("path") else web.Response(
            text="Audio file not found or access denied", status=404
        )
    if not target.is_file():
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
    target_path = resolve_library_path(data.get("path"))
    if target_path is None:
        status = 400 if not data.get("path") else 403
        error = "File path is required" if status == 400 else "Path is outside permissible data/benchmark folders"
        return web.json_response({"error": error}, status=status)

    try:
        rel_path = delete_library_file(target_path)
        return web.json_response({
            "status": "success",
            "deleted_file": target_path.name,
            "path": rel_path,
        })
    except FileNotFoundError:
        return web.json_response({"error": f"File not found: {target_path.name}"}, status=404)
    except Exception as e:
        logger.exception("Failed to delete file: %s", target_path)
        return web.json_response({"error": str(e)}, status=500)


async def handle_bulk_delete_library_files(request: web.Request) -> web.Response:
    """Delete multiple audio files and matching sidecar JSONs from disk."""
    data = await request.json()
    paths = data.get("paths", [])
    if not paths:
        return web.json_response({"error": "List of file paths is required"}, status=400)

    deleted_count = 0
    errors = []

    for p_str in paths:
        target = resolve_library_path(p_str)
        if target is None or not target.is_file():
            errors.append({"path": p_str, "error": "not found or not permitted"})
            continue
        try:
            delete_library_file(target)
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
    resolved = resolve_library_path(data.get("path"))
    if resolved is None:
        status = 400 if not data.get("path") else 403
        error = "Path is required" if status == 400 else "Path is outside permissible project audio folders"
        return web.json_response({"error": error}, status=status)

    if not resolved.is_file():
        return web.json_response({"error": f"File not found: {resolved}"}, status=404)

    try:
        existing_id = registry.find_id_by_path(resolved)
        if existing_id:
            audio = registry.get_audio(existing_id)
            return web.json_response({
                "audio_id": existing_id,
                "metadata": audio.metadata() if audio else {},
                "reused": True,
            })

        audio = Audio.from_file(resolved)
        _category_id, category = categorize_library_path(library_relative_path(resolved))
        audio_id = registry.register(
            audio,
            source_type="library",
            tags=["library", category.lower().replace(" ", "_")],
        )
        return web.json_response({"audio_id": audio_id, "metadata": audio.metadata(), "reused": False})
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
    try:
        sample_rate = parse_crawl_sample_rate(data.get("sample_rate", DEFAULT_SAMPLE_RATE))
    except (TypeError, ValueError):
        return web.json_response(
            {"error": "sample_rate must be 'native', 16000, or 44100"},
            status=400,
        )
    if sample_rate not in (None, 16000, DEFAULT_SAMPLE_RATE):
        return web.json_response(
            {"error": "sample_rate must be 'native', 16000, or 44100"},
            status=400,
        )
    audio_format = data.get("audio_format", "wav")

    if not url:
        return web.json_response({"error": "URL is required"}, status=400)

    rate_label = "native" if sample_rate is None else f"{sample_rate}Hz"
    task_id = task_manager.create_task(
        "youtube_crawl",
        {"url": url, "sample_rate": sample_rate, "sample_rate_label": rate_label},
    )

    async def run_crawler():
        task_manager.update_task(
            task_id,
            status="running",
            message=f"Downloading YouTube audio ({rate_label}) with yt-dlp...",
        )
        loop = asyncio.get_running_loop()
        try:
            crawler = YtCrawler(
                output_dir=DATA_DIR / "yt_crawler" / "downloads",
                work_dir=DATA_DIR / "yt_crawler" / "work",
                audio_format=audio_format,
                sample_rate=sample_rate,
                channels=1,
                progress_callback=_cli_progress_reporter(task_id, loop, "yt-dlp: "),
            )
            task_manager.set_cancel_callback(task_id, crawler.cancel)
            try:
                audio = await loop.run_in_executor(None, crawler.download, url)
            finally:
                task_manager.set_cancel_callback(task_id, None)
            if _task_is_cancelled(task_id):
                return
            audio_id = registry.register(
                audio,
                source_type="youtube",
                tags=["youtube", "crawled", rate_label],
            )
            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                progress_known=True,
                message=f"Downloaded '{audio.title}' successfully!",
                result={"audio_id": audio_id, "metadata": audio.metadata()},
            )
        except Exception as e:
            if _task_is_cancelled(task_id):
                return
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
    """Return a signed, per-channel envelope for one audio time window."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio or not audio.path.is_file():
        return web.json_response({"error": "Audio file not found"}, status=404)

    try:
        info = sf.info(str(audio.path))
        duration_s = info.frames / info.samplerate if info.samplerate else 0.0
        start_s = float(request.query.get("start_s", 0.0))
        end_s = float(request.query.get("end_s", duration_s))
        bins = int(request.query.get("bins", 1200))
    except (TypeError, ValueError, RuntimeError) as exc:
        return web.json_response({"error": f"Invalid waveform query: {exc}"}, status=400)

    if not all(isfinite(value) for value in (start_s, end_s)):
        return web.json_response({"error": "start_s and end_s must be finite"}, status=400)
    if not 0 <= start_s < end_s <= duration_s:
        return web.json_response(
            {"error": f"Expected 0 <= start_s < end_s <= {duration_s:.9f}"},
            status=400,
        )
    if not 1 <= bins <= 8192:
        return web.json_response({"error": "bins must be between 1 and 8192"}, status=400)

    start_frame = max(0, min(info.frames - 1, int(np.floor(start_s * info.samplerate))))
    end_frame = max(start_frame + 1, min(info.frames, int(np.ceil(end_s * info.samplerate))))
    is_full_track = start_frame == 0 and end_frame == info.frames
    if is_full_track:
        cached = registry.get_cached_waveform(audio_id, bins)
        if cached is not None:
            return web.json_response(cached)

    loop = asyncio.get_running_loop()
    try:
        waveform = await loop.run_in_executor(
            None,
            extract_waveform_envelope,
            audio.path,
            start_frame,
            end_frame,
            bins,
        )
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        logger.error("Failed to extract waveform for %s: %s", audio.path, exc)
        return web.json_response({"error": "Could not read audio waveform"}, status=500)
    if is_full_track:
        registry.cache_waveform(audio_id, waveform)
    return web.json_response(waveform)


async def handle_get_spectrogram(request: web.Request) -> web.Response:
    """Generate a marginless, linear-Hz STFT image for one visible window."""
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio or not audio.path.is_file():
        return web.Response(text="Audio file not found", status=404)

    try:
        info = sf.info(str(audio.path))
        duration_s = info.frames / info.samplerate if info.samplerate else 0.0
        start_s = float(request.query.get("start_s", 0.0))
        end_s = float(request.query.get("end_s", duration_s))
        width = int(request.query.get("width", 1200))
        height = int(request.query.get("height", 320))
    except (TypeError, ValueError, RuntimeError) as exc:
        return web.Response(text=f"Invalid spectrogram query: {exc}", status=400)
    if not all(isfinite(value) for value in (start_s, end_s)):
        return web.Response(text="start_s and end_s must be finite", status=400)
    if not 0 <= start_s < end_s <= duration_s:
        return web.Response(
            text=f"Expected 0 <= start_s < end_s <= {duration_s:.9f}",
            status=400,
        )
    if not 32 <= width <= 4096 or not 32 <= height <= 2048:
        return web.Response(text="width must be 32..4096 and height must be 32..2048", status=400)

    start_frame = max(0, min(info.frames - 1, int(np.floor(start_s * info.samplerate))))
    end_frame = max(start_frame + 1, min(info.frames, int(np.ceil(end_s * info.samplerate))))

    def generate_spec_png() -> bytes:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        frame_count = end_frame - start_frame
        n_fft = min(4096, max(32, 2 ** int(np.floor(np.log2(max(32, frame_count))))))
        power = np.empty((n_fft // 2 + 1, width), dtype=np.float32)
        window = np.hanning(n_fft).astype(np.float32)[:, np.newaxis]
        with sf.SoundFile(str(audio.path)) as source:
            for column in range(width):
                center = start_frame + int((column + 0.5) * frame_count / width)
                desired_start = center - n_fft // 2
                read_start = max(start_frame, desired_start)
                read_end = min(end_frame, desired_start + n_fft)
                source.seek(read_start)
                block = source.read(read_end - read_start, dtype="float32", always_2d=True)
                padded = np.zeros((n_fft, info.channels), dtype=np.float32)
                offset = read_start - desired_start
                usable = min(len(block), n_fft - offset)
                if usable:
                    padded[offset : offset + usable] = block[:usable]
                spectrum = np.fft.rfft(np.nan_to_num(padded) * window, axis=0)
                power[:, column] = np.mean(np.abs(spectrum) ** 2, axis=1)
        peak = float(np.max(power)) if power.size else 1.0
        power_db = 10.0 * np.log10(np.maximum(power, max(peak, 1e-20) * 1e-8))
        power_db -= 10.0 * np.log10(max(peak, 1e-20))

        dpi = 100
        figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_axes((0, 0, 1, 1))
        axis.imshow(
            power_db,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=-80,
            vmax=0,
            extent=(start_s, end_s, 0, info.samplerate / 2),
        )
        axis.set_axis_off()
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        return buffer.getvalue()

    loop = asyncio.get_running_loop()
    try:
        png_bytes = await loop.run_in_executor(None, generate_spec_png)
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        logger.error("Failed to generate spectrogram for %s: %s", audio.path, exc)
        return web.Response(text="Could not generate spectrogram", status=500)
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


def _download_filename(name: str | None, default_stem: str, suffix: str) -> str:
    """Return a filesystem-safe download name with the requested suffix."""
    stem = _sanitize_filename_component(Path(name or "").stem) or default_stem
    ext = Path(name or "").suffix.lower() or suffix
    if not ext.startswith("."):
        ext = f".{ext}"
    return f"{stem}{ext}"


def _attachment_headers(filename: str, content_type: str | None = None) -> dict[str, str]:
    """Build Content-Disposition headers for a browser file download."""
    ascii_name = filename.encode("ascii", "ignore").decode("ascii") or "download"
    headers = {
        "Content-Disposition": (
            f'attachment; filename="{ascii_name}"; '
            f"filename*=UTF-8''{quote(filename)}"
        ),
        "Cache-Control": "no-store",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _slice_audio_frames(
    waveform: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Return ``[start_s, end_s)`` of a 2-D waveform."""
    start_frame = int(round(float(start_s) * sample_rate))
    end_frame = int(round(float(end_s) * sample_rate))
    start_frame = max(0, min(start_frame, waveform.shape[0]))
    end_frame = max(start_frame, min(end_frame, waveform.shape[0]))
    if end_frame <= start_frame:
        raise AudioCutterError(
            f"resolved empty cut: start={start_s:.6f}s end={end_s:.6f}s"
        )
    return waveform[start_frame:end_frame]


def _unique_zip_entry(name: str, used: set[str]) -> str:
    """Return a zip entry name that does not collide with ``used``."""
    base = _sanitize_filename_component(Path(name).stem) or "turn"
    suffix = Path(name).suffix or ".wav"
    candidate = f"{base}{suffix}"
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


async def handle_download_audio_segment(request: web.Request) -> web.StreamResponse:
    """Cut ``[start, end)`` and return it as a downloadable WAV.

    Does not register a new session audio item. Repeated downloads of the
    same range reuse a cached cut under ``.data/audio_cutter/segments/``.
    """
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio or not Path(audio.path).is_file():
        return web.json_response({"error": "Audio not found"}, status=404)

    try:
        start = float(request.query.get("start", ""))
        end = float(request.query.get("end", ""))
    except (TypeError, ValueError):
        return web.json_response(
            {"error": "start and end query params (seconds) are required"},
            status=400,
        )
    if not isfinite(start) or not isfinite(end) or end <= start:
        return web.json_response(
            {"error": "end must be greater than start"}, status=400
        )

    default_stem = _sanitize_filename_component(
        f"{audio.title or audio_id}_{start:.3f}-{end:.3f}"
    ) or "turn"
    filename = _download_filename(
        request.query.get("filename"), default_stem, ".wav"
    )
    cache_dir = AUDIO_SEGMENT_DIR / (_sanitize_filename_component(audio_id) or "audio")
    output_path = cache_dir / f"{start:.3f}_{end:.3f}.wav"
    if not output_path.is_file():
        cutter = AudioCutter(output_dir=cache_dir)
        try:
            await asyncio.to_thread(
                cutter.cut, audio, start, end, output_path=output_path
            )
        except (AudioCutterError, FileNotFoundError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            logger.exception("Failed to cut audio segment: %s", exc)
            return web.json_response({"error": f"Failed to cut segment: {exc}"}, status=500)

    inline = request.query.get("inline") in ("1", "true")
    if inline:
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Content-Disposition": f'inline; filename="{output_path.name}"',
            "Content-Type": "audio/wav",
        }
        return web.FileResponse(output_path, headers=headers)

    return web.FileResponse(
        output_path, headers=_attachment_headers(filename, content_type="audio/wav")
    )


async def handle_download_audio_segments_zip(request: web.Request) -> web.StreamResponse:
    """Cut many ``[start, end)`` ranges into one zip without registering cuts.

    Reads the source audio once. Intended for the Turns Inspector bulk
    download of the currently filtered table.
    """
    audio_id = request.match_info["id"]
    audio = registry.get_audio(audio_id)
    if not audio or not Path(audio.path).is_file():
        return web.json_response({"error": "Audio not found"}, status=404)

    try:
        data = await request.json()
    except (json.JSONDecodeError, aiohttp.ContentTypeError):
        return web.json_response({"error": "JSON body is required"}, status=400)

    raw_segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(raw_segments, list) or not raw_segments:
        return web.json_response({"error": "segments must be a non-empty list"}, status=400)
    if len(raw_segments) > MAX_SEGMENT_ZIP_ITEMS:
        return web.json_response(
            {"error": f"At most {MAX_SEGMENT_ZIP_ITEMS} segments can be downloaded at once"},
            status=400,
        )

    parsed: list[tuple[float, float, str]] = []
    try:
        for index, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                raise ValueError(f"segments[{index}] must be an object")
            start = float(item.get("start"))
            end = float(item.get("end"))
            if not isfinite(start) or not isfinite(end) or end <= start:
                raise ValueError(f"segments[{index}] needs end > start")
            parsed.append((start, end, str(item.get("filename") or f"turn_{index + 1:03d}.wav")))
    except (TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)

    zip_name = _download_filename(
        data.get("filename") if isinstance(data, dict) else None,
        _sanitize_filename_component(f"{audio.title or audio_id}_turns") or "turns",
        ".zip",
    )
    cache_dir = AUDIO_SEGMENT_DIR / (_sanitize_filename_component(audio_id) or "audio")
    zip_path = cache_dir / f"download_{uuid.uuid4().hex}.zip"

    def build_zip() -> None:
        waveform, sample_rate = sf.read(str(audio.path), always_2d=True)
        if waveform.shape[0] == 0:
            raise AudioCutterError(f"Audio source is empty: {audio.path}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for start, end, given_name in parsed:
                frames = _slice_audio_frames(waveform, int(sample_rate), start, end)
                buf = io.BytesIO()
                sf.write(buf, frames, int(sample_rate), format="WAV")
                archive.writestr(_unique_zip_entry(given_name, used), buf.getvalue())

    try:
        await asyncio.to_thread(build_zip)
    except (AudioCutterError, FileNotFoundError, OSError) as exc:
        zip_path.unlink(missing_ok=True)
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        zip_path.unlink(missing_ok=True)
        logger.exception("Turn-segment zip failed")
        return web.json_response({"error": "Failed to build turn download zip"}, status=500)

    response = web.StreamResponse(
        status=200,
        headers={
            **_attachment_headers(zip_name, "application/zip"),
            "Content-Length": str(zip_path.stat().st_size),
        },
    )
    try:
        await response.prepare(request)
        with zip_path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                await response.write(chunk)
        await response.write_eof()
        return response
    finally:
        zip_path.unlink(missing_ok=True)


async def handle_quick_save(request: web.Request) -> web.Response:
    """Perform Audio.quick_save() in canonical runtime storage."""
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
                DATA_DIR / "quick_save",
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
        target_device = get_default_device() if device == "auto" else device
        power_w = _get_device_power_w(target_device)
        power_msg = f" (⚡ {power_w}W)" if power_w is not None else ""
        task_manager.update_task(
            task_id,
            status="running",
            message=f"Initializing {model_type.upper()} on {target_device}{power_msg}...",
        )
        loop = asyncio.get_running_loop()

        def do_separation():
            if target_device.startswith("cuda:") and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(int(target_device.split(":")[1]))
                except Exception:
                    pass

            if model_type == "htdemucs":
                sep = HTDemucs(
                    model=model_name or "htdemucs",
                    device=target_device,
                    two_stems=two_stems,
                    output_dir=DATA_DIR / "demucs" / "out",
                    work_dir=DATA_DIR / "demucs" / "work",
                    progress_callback=_cli_progress_reporter(task_id, loop, "Demucs: "),
                )
                task_manager.set_cancel_callback(task_id, sep.cancel)
                try:
                    return sep.separate(audio)
                finally:
                    task_manager.set_cancel_callback(task_id, None)
                    sep.close()
            
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
                def report_mvsep_progress(message: str) -> None:
                    def update_progress() -> None:
                        task = task_manager.get_task(task_id)
                        if task and task["status"] == "running":
                            task_manager.update_task(
                                task_id,
                                message=f"MVSep-MDX23: {message}",
                            )

                    loop.call_soon_threadsafe(update_progress)

                sep = MVSepMDX23(
                    device=target_device,
                    output_dir=DATA_DIR / "mvsep_mdx23" / "out",
                    work_dir=DATA_DIR / "mvsep_mdx23" / "work",
                    repo_dir=DATA_DIR / "mvsep_mdx23" / "repo",
                    progress_callback=report_mvsep_progress,
                )
                task_manager.set_cancel_callback(task_id, sep.cancel)
                try:
                    return sep.separate(audio)
                finally:
                    task_manager.set_cancel_callback(task_id, None)
                    sep.close()
            
            else:
                raise ValueError(f"Unknown separation model: {model_type}")

        try:
            start_time = time.time()
            separated_audio = await loop.run_in_executor(None, do_separation)
            if _task_is_cancelled(task_id):
                return
            elapsed = time.time() - start_time
            end_power_w = _get_device_power_w(target_device)
            
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
                    "device": target_device,
                    "power_w": end_power_w,
                },
            )
            complete_msg = (
                f"Separation completed in {elapsed:.2f}s on {target_device} (⚡ {end_power_w}W)!"
                if end_power_w is not None
                else f"Separation completed in {elapsed:.2f}s on {target_device}!"
            )
            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                progress_known=True,
                message=complete_msg,
                result={
                    "separated_audio_id": new_id,
                    "metadata": separated_audio.metadata(),
                    "elapsed_s": round(elapsed, 2),
                    "model_type": model_type,
                    "model_label": model_label,
                    "device": target_device,
                    "power_w": end_power_w,
                },
            )
        except Exception as e:
            task = task_manager.get_task(task_id)
            if task and task["status"] == "cancelled":
                return
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
    model_type = data.get("model_type", "pyannote").lower()
    model_id = data.get("model_id")
    device = data.get("device", "auto")
    token = data.get("token") or os.getenv("HF_TOKEN")
    num_speakers = data.get("num_speakers")
    min_speakers = data.get("min_speakers")
    max_speakers = data.get("max_speakers")
    try:
        batch_size = int(data.get("batch_size", 1))
        if batch_size < 1 or batch_size > 256:
            raise ValueError
    except (TypeError, ValueError):
        return web.json_response(
            {"error": "batch_size must be an integer from 1 to 256"}, status=400
        )
    enrollment_profile_name = data.get("enrollment_profile")
    include_overlap = bool(data.get("include_overlap", False))
    try:
        sortformer_settings = _sortformer_settings(data)
    except (TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    try:
        vad_onset = float(data.get("vad_onset", 0.5)) if data.get("vad_onset") is not None else 0.5
        vad_offset = float(data.get("vad_offset", 0.3)) if data.get("vad_offset") is not None else 0.3
    except (TypeError, ValueError):
        vad_onset, vad_offset = 0.5, 0.3
    try:
        chunk_duration_s = float(data.get("chunk_duration_s", 1.5))
        chunk_step_s = float(data.get("chunk_step_s", 0.75))
    except (TypeError, ValueError):
        return web.json_response(
            {"error": "chunk_duration_s and chunk_step_s must be numbers"},
            status=400,
        )

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    enrollment_profile = None
    if enrollment_profile_name:
        if model_type != "sortformer":
            return web.json_response(
                {
                    "error": (
                        f"{model_type} does not support pre-inference speaker "
                        "enrollment. Select NeMo Sortformer or run without an "
                        "enrolled speaker."
                    )
                },
                status=400,
            )
        verifier = SpeakerVerifier()
        try:
            enrollment_profile = verifier.load_profile(enrollment_profile_name)
        except SpeakerVerifierError as exc:
            return web.json_response({"error": str(exc)}, status=404)

    task_id = task_manager.create_task(
        "diarization",
        {
            "audio_id": audio_id,
            "model_type": model_type,
            "device": device,
            "enrollment_profile": enrollment_profile_name,
            "sortformer_settings": sortformer_settings,
        },
    )

    async def run_diar():
        target_device = get_default_device() if device == "auto" else device
        power_w = _get_device_power_w(target_device)
        power_msg = f" (⚡ {power_w}W)" if power_w is not None else ""
        task_manager.update_task(
            task_id,
            status="running",
            message=(
                f"Running speaker diarization with {model_type} on "
                f"{target_device}{power_msg}"
                + (
                    f" using enrolled speaker '{enrollment_profile_name}'..."
                    if enrollment_profile_name
                    else "..."
                )
            ),
        )
        loop = asyncio.get_running_loop()

        def do_diarization():
            if target_device.startswith("cuda:") and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(int(target_device.split(":")[1]))
                except Exception:
                    pass
            
            if model_type in {"pyannote", "pyannote_community"}:
                target_model_id = model_id or "pyannote/speaker-diarization-community-1"
                diarizer = PyannoteDiarizer(
                    model_id=target_model_id,
                    device=target_device,
                    token=token,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    batch_size=batch_size,
                )
                with diarizer:
                    return diarizer.diarize(audio)
            elif model_type in {"pyannote_31", "pyannote_3", "pyannote_3.1"}:
                target_model_id = model_id or "pyannote/speaker-diarization-3.1"
                diarizer = PyannoteDiarizer(
                    model_id=target_model_id,
                    device=target_device,
                    token=token,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    batch_size=batch_size,
                )
                with diarizer:
                    return diarizer.diarize(audio)
            elif model_type == "sortformer":
                diarizer = SortformerWorkerDiarizer(
                    device=target_device,
                    batch_size=batch_size,
                    **sortformer_settings,
                )
                task_manager.set_cancel_callback(task_id, diarizer.cancel)
                try:
                    with diarizer:
                        return diarizer.diarize(
                            audio,
                            enrollment_name=(
                                enrollment_profile.name
                                if enrollment_profile is not None
                                else None
                            ),
                            enrollment_clips=(
                                enrollment_profile.clip_paths
                                if enrollment_profile is not None
                                else None
                            ),
                        )
                finally:
                    task_manager.set_cancel_callback(task_id, None)
            elif model_type in {"clustering", "nemo-clustering"}:
                oracle_speakers, max_num_speakers = (
                    ClusteringDiarizer.resolve_speaker_settings(
                        num_speakers,
                        min_speakers,
                        max_speakers,
                    )
                )
                diarizer = ClusteringWorkerDiarizer(
                    device=target_device,
                    num_speakers=oracle_speakers,
                    max_num_speakers=max_num_speakers,
                    vad_onset=vad_onset,
                    vad_offset=vad_offset,
                    batch_size=batch_size,
                )
                task_manager.set_cancel_callback(task_id, diarizer.cancel)
                try:
                    with diarizer:
                        return diarizer.diarize(audio)
                finally:
                    task_manager.set_cancel_callback(task_id, None)
            elif model_type in {"3d_speaker", "3d-speaker", "threed_speaker", "speakerlab"}:
                oracle_speakers = ThreeDSpeakerDiarizer.resolve_speaker_settings(
                    num_speakers,
                    min_speakers,
                    max_speakers,
                )
                diarizer = ThreeDSpeakerWorkerDiarizer(
                    device=target_device,
                    num_speakers=oracle_speakers,
                    include_overlap=include_overlap,
                    batch_size=batch_size,
                    chunk_duration_s=chunk_duration_s,
                    chunk_step_s=chunk_step_s,
                    token=token if include_overlap else None,
                )
                task_manager.set_cancel_callback(task_id, diarizer.cancel)
                try:
                    with diarizer:
                        return diarizer.diarize(audio)
                finally:
                    task_manager.set_cancel_callback(task_id, None)
            elif model_type in {"diarizen", "diarizen_large_s80_v2"}:
                diarizer = DiariZenWorkerDiarizer(
                    model_id=(
                        model_id
                        or "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
                    ),
                    device=target_device,
                    token=token,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    batch_size=batch_size,
                )
                task_manager.set_cancel_callback(task_id, diarizer.cancel)
                try:
                    with diarizer:
                        return diarizer.diarize(audio)
                finally:
                    task_manager.set_cancel_callback(task_id, None)
            else:
                raise ValueError(f"Unknown diarization model: {model_type}")

        try:
            start_time = time.time()
            result = await loop.run_in_executor(None, do_diarization)
            if _task_is_cancelled(task_id):
                return
            elapsed = time.time() - start_time
            end_power_w = _get_device_power_w(target_device)

            result_path = result.save(DIARIZATION_RESULTS_DIR)
            result_dict = result.to_dict()
            active_item = registry.get_item(audio_id)
            if active_item:
                retained_tags = {
                    tag for tag in active_item.get("system_tags", [])
                    if not tag.startswith("speaker:")
                    and not tag.startswith("profile:")
                    and tag not in {
                        "stage:verified",
                        "verification:passed",
                        "verification:rejected",
                    }
                }
                active_item["system_tags"] = sorted(
                    retained_tags
                    | {"stage:diarized", "verification:unverified"}
                    | {f"speaker:{speaker.speaker_id}" for speaker in result.speakers}
                )
            try:
                from src.web_pipeline.dataset_manager import dataset_manager
                pipeline_item = dataset_manager.find_item_by_path(audio.path)
                if pipeline_item:
                    dataset_manager.attach_diarization(pipeline_item.id, result_dict)
            except Exception:
                logger.exception("Could not synchronize diarization metadata to Pipeline")
            complete_msg = (
                f"Diarization finished in {elapsed:.2f}s on {target_device} (⚡ {end_power_w}W)!"
                if end_power_w is not None
                else f"Diarization finished in {elapsed:.2f}s on {target_device}!"
            )
            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                progress_known=True,
                message=complete_msg,
                result={
                    "diarization": result_dict,
                    "diarization_result_id": result.result_id,
                    "diarization_result_path": str(result_path),
                    "elapsed_s": round(elapsed, 2),
                    "audio_id": audio_id,
                    "device": target_device,
                    "power_w": end_power_w,
                    "enrollment_profile": enrollment_profile_name,
                    "sortformer_settings": (
                        sortformer_settings if model_type == "sortformer" else None
                    ),
                },
            )
        except Exception as e:
            task = task_manager.get_task(task_id)
            if task and task["status"] == "cancelled":
                return
            logger.exception("Diarization failed")
            task_manager.update_task(
                task_id,
                status="failed",
                error=str(e),
                message=f"Diarization failed: {e}",
            )

    task_manager.enqueue(task_id, run_diar)
    return web.json_response({"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202)


async def handle_clean_diarization_turns(request: web.Request) -> web.Response:
    """Build a non-persistent clean-turn view from raw diarization turns."""
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise TypeError("request body must be an object")
        settings = _clean_turn_settings(data)
        result_id = str(data.get("result_id") or "").strip()
        if result_id:
            result = _load_diarization_result(result_id)
            raw_turns: Any = [asdict(turn) for turn in result.turns]
        else:
            raw_turns = data.get("turns")
        cleaned = _clean_turn_payloads(raw_turns, settings)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)

    return web.json_response(
        {
            "turns": cleaned,
            "settings": settings,
            "summary": {
                "raw_turn_count": len(raw_turns),
                "clean_turn_count": len(cleaned),
                "removed_turn_count": len(raw_turns) - len(cleaned),
                "raw_duration_s": sum(
                    float(turn.get("end_s", 0)) - float(turn.get("start_s", 0))
                    for turn in raw_turns
                ),
                "clean_duration_s": sum(turn["duration_s"] for turn in cleaned),
            },
        }
    )


async def handle_extract_speaker_audio(request: web.Request) -> web.Response:
    """Extract all turns for a specific speaker and concatenate into a new Audio item."""
    data = await request.json()
    audio_id = data.get("audio_id")
    speaker_id = data.get("speaker_id")
    speaker_name = data.get("speaker_name") or speaker_id
    turns = data.get("turns", [])
    clean_turns = bool(data.get("clean_turns", False))

    mode = data.get("mode", "concatenated")  # "concatenated" or "time_aligned"

    if not audio_id or not speaker_id:
        return web.json_response({"error": "audio_id and speaker_id are required"}, status=400)

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    if clean_turns:
        try:
            turns = _clean_turn_payloads(turns, _clean_turn_settings(data))
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    try:
        extraction_settings = _extraction_settings(data)
    except (TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)

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

        valid_intervals = _padded_audio_intervals(
            spk_turns,
            sample_rate=sr,
            total_frames=total_frames,
            blocker_turns=_request_blocker_turns(
                data, turns, exclude_speaker_id=speaker_id
            ),
            **extraction_settings,
        )

        if not valid_intervals:
            raise ValueError(f"No valid audio samples found for speaker {speaker_id}")

        valid_intervals.sort(key=lambda x: x[0])
        if mode == "time_aligned":
            combined = np.zeros_like(waveform)
            for s, e in valid_intervals:
                combined[s:e] = waveform[s:e]
            mode_suffix = "aligned"
        else:
            segments = [waveform[s:e] for s, e in valid_intervals]
            combined = np.concatenate(segments, axis=0)
            mode_suffix = "concat"

        sanitized_title = _sanitize_filename_component(audio.title or audio.source_id)
        sanitized_spk = _sanitize_filename_component(speaker_name or speaker_id)
        turn_policy = "clean" if clean_turns else "raw"
        out_filename = (
            f"{sanitized_title}_{sanitized_spk}_{turn_policy}_{mode_suffix}.wav"
        )
        out_path = out_dir / out_filename
        sf.write(str(out_path), combined, sr)

        tag_title = (
            f"{audio.title or audio.source_id} [{speaker_name}] "
            f"({turn_policy} {mode_suffix})"
        )
        extracted_audio = Audio.from_file(
            out_path,
            source_id=(
                f"{audio.source_id}_{sanitized_spk}_{turn_policy}_{mode_suffix}"
            ),
            title=tag_title,
            source_url=audio.source_url,
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
            native_sample_rate=audio.native_sample_rate,
            history=(
                *audio.history,
                f"diar_extract_{speaker_id}_{turn_policy}_{mode}",
            ),
        )
        return extracted_audio

    try:
        loop = asyncio.get_running_loop()
        extracted = await loop.run_in_executor(None, do_extract)
        new_id = registry.register(
            extracted,
            source_type="speaker_stem",
            parent_id=audio_id,
            tags=[
                "diarization",
                f"speaker:{speaker_id}",
                f"mode:{mode}",
                f"turns:{'clean' if clean_turns else 'raw'}",
            ],
        )
        return web.json_response({
            "audio_id": new_id,
            "metadata": extracted.metadata(),
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "mode": mode,
            "turn_policy": "clean" if clean_turns else "raw",
            "duration_s": extracted.duration_s,
            "extraction_settings": extraction_settings,
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
    mode = data.get("mode", "concatenated")  # "concatenated" or "time_aligned"
    clean_turns = bool(data.get("clean_turns", False))

    if not audio_id or not turns:
        return web.json_response({"error": "audio_id and non-empty turns are required"}, status=400)

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    if clean_turns:
        try:
            turns = _clean_turn_payloads(turns, _clean_turn_settings(data))
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    try:
        extraction_settings = _extraction_settings(data)
    except (TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)

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

        mode_suffix = "aligned" if mode == "time_aligned" else "concat"
        turn_policy = "clean" if clean_turns else "raw"

        for spk_id in unique_speakers:
            spk_name = speaker_names.get(spk_id, spk_id)
            spk_turns = [t for t in turns if t.get("speaker_id") == spk_id]
            valid_intervals = _padded_audio_intervals(
                spk_turns,
                sample_rate=sr,
                total_frames=total_frames,
                blocker_turns=[
                    turn for turn in turns if turn.get("speaker_id") != spk_id
                ],
                **extraction_settings,
            )

            if not valid_intervals:
                continue

            valid_intervals.sort(key=lambda x: x[0])
            if mode == "time_aligned":
                combined = np.zeros_like(waveform)
                for s, e in valid_intervals:
                    combined[s:e] = waveform[s:e]
            else:
                segments = [waveform[s:e] for s, e in valid_intervals]
                combined = np.concatenate(segments, axis=0)

            sanitized_title = _sanitize_filename_component(audio.title or audio.source_id)
            sanitized_spk = _sanitize_filename_component(spk_name or spk_id)
            out_filename = (
                f"{sanitized_title}_{sanitized_spk}_{turn_policy}_{mode_suffix}.wav"
            )
            out_path = out_dir / out_filename
            sf.write(str(out_path), combined, sr)

            extracted_audio = Audio.from_file(
                out_path,
                source_id=(
                    f"{audio.source_id}_{sanitized_spk}_{turn_policy}_{mode_suffix}"
                ),
                title=(
                    f"{audio.title or audio.source_id} [{spk_name}] "
                    f"({turn_policy} {mode_suffix})"
                ),
                source_url=audio.source_url,
                channel_id=audio.channel_id,
                channel_name=audio.channel_name,
                channel_url=audio.channel_url,
                native_sample_rate=audio.native_sample_rate,
                history=(
                    *audio.history,
                    f"diar_extract_{spk_id}_{turn_policy}_{mode}",
                ),
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
                tags=[
                    "diarization",
                    f"speaker:{spk_id}",
                    f"turns:{'clean' if clean_turns else 'raw'}",
                ],
            )
            registered.append({
                "audio_id": new_id,
                "speaker_id": spk_id,
                "speaker_name": spk_name,
                "metadata": extracted_audio.metadata(),
                "duration_s": extracted_audio.duration_s,
            })
        return web.json_response(
            {
                "extracted": registered,
                "total_speakers": len(registered),
                "turn_policy": "clean" if clean_turns else "raw",
                "extraction_settings": extraction_settings,
            }
        )
    except Exception as e:
        logger.exception("Bulk speaker extraction failed")
        return web.json_response({"error": str(e)}, status=500)


# ==================== KNOWN SPEAKERS ====================


def _profile_summary(verifier: SpeakerVerifier, name: str) -> dict[str, Any]:
    profile = verifier.load_profile(name)
    return {
        "name": profile.name,
        "num_clips": len(profile.clip_paths),
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "clips": [
            {
                "name": path.name,
                "stream_url": (
                    f"/api/speaker-profiles/{quote(profile.name, safe='')}/clips/"
                    f"{quote(path.name, safe='')}"
                ),
            }
            for path in profile.clip_paths
        ],
        "channel_id": profile.channel_id,
        "channel_name": profile.channel_name,
        "channel_url": profile.channel_url,
    }


async def handle_list_speaker_profiles(request: web.Request) -> web.Response:
    """List globally reusable enrolled speaker profiles."""
    verifier = SpeakerVerifier()
    profiles = []
    for name in verifier.list_profiles():
        try:
            profiles.append(_profile_summary(verifier, name))
        except SpeakerVerifierError:
            continue
    return web.json_response({"profiles": profiles})


async def handle_get_speaker_profile(request: web.Request) -> web.Response:
    """Return one speaker profile and its manageable reference clips."""
    verifier = SpeakerVerifier()
    try:
        return web.json_response(_profile_summary(verifier, request.match_info["name"]))
    except SpeakerVerifierError as exc:
        return web.json_response({"error": str(exc)}, status=404)


async def handle_create_speaker_profile(request: web.Request) -> web.Response:
    """Create a global speaker profile from clean reference clips."""
    data = await request.json()
    name = data.get("name")
    clip_audio_ids = data.get("clip_audio_ids", [])
    overwrite = bool(data.get("overwrite", False))
    channel_id = data.get("channel_id")
    channel_name = data.get("channel_name")
    channel_url = data.get("channel_url")

    if not name or not clip_audio_ids:
        return web.json_response(
            {"error": "name and clip_audio_ids are required"}, status=400
        )

    clips = []
    for clip_id in clip_audio_ids:
        clip = registry.get_audio(clip_id)
        if not clip:
            return web.json_response(
                {"error": f"Audio not found: {clip_id}"}, status=404
            )
        clips.append(clip)

    verifier = SpeakerVerifier()
    try:
        profile = verifier.enroll(
            name,
            clips,
            overwrite=overwrite,
            channel_id=channel_id,
            channel_name=channel_name,
            channel_url=channel_url,
        )
    except SpeakerVerifierError as e:
        return web.json_response({"error": str(e)}, status=409)
    return web.json_response(
        _profile_summary(verifier, profile.name),
        status=201,
    )


async def handle_add_speaker_profile_clips(request: web.Request) -> web.Response:
    """Append selected session clips to an existing speaker profile."""
    data = await request.json()
    clip_audio_ids = data.get("clip_audio_ids", [])
    if not clip_audio_ids:
        return web.json_response({"error": "clip_audio_ids is required"}, status=400)

    clips = []
    for clip_id in clip_audio_ids:
        clip = registry.get_audio(clip_id)
        if not clip:
            return web.json_response(
                {"error": f"Audio not found: {clip_id}"}, status=404
            )
        clips.append(clip)

    verifier = SpeakerVerifier()
    try:
        profile = verifier.add_clips(request.match_info["name"], clips)
    except SpeakerVerifierError as exc:
        return web.json_response({"error": str(exc)}, status=409)
    return web.json_response(_profile_summary(verifier, profile.name))


async def handle_stream_speaker_profile_clip(request: web.Request) -> web.StreamResponse:
    """Stream one stored enrollment clip for auditioning."""
    verifier = SpeakerVerifier()
    try:
        profile = verifier.load_profile(request.match_info["name"])
    except SpeakerVerifierError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    clip_name = Path(request.match_info["clip_name"]).name
    clip_path = next(
        (path for path in profile.clip_paths if path.name == clip_name),
        None,
    )
    if clip_path is None:
        return web.json_response({"error": "Profile clip not found"}, status=404)
    return web.FileResponse(clip_path)


async def handle_delete_speaker_profile_clip(request: web.Request) -> web.Response:
    """Remove one bad enrollment clip from a speaker profile."""
    verifier = SpeakerVerifier()
    try:
        profile = verifier.remove_clip(
            request.match_info["name"],
            request.match_info["clip_name"],
        )
    except SpeakerVerifierError as exc:
        return web.json_response({"error": str(exc)}, status=409)
    return web.json_response(_profile_summary(verifier, profile.name))


async def handle_delete_speaker_profile(request: web.Request) -> web.Response:
    """Delete an enrolled target speaker profile."""
    name = request.match_info["name"]
    verifier = SpeakerVerifier()
    try:
        verifier.delete_profile(name)
    except SpeakerVerifierError as e:
        return web.json_response({"error": str(e)}, status=404)
    return web.json_response({"deleted": name})


async def handle_target_speaker_score(request: web.Request) -> web.Response:
    """Score diarization turns against a target speaker profile in background."""
    data = await request.json()
    audio_id = data.get("audio_id")
    profile_name = data.get("profile")
    turns = data.get("turns", [])
    device = data.get("device", "auto")
    token = data.get("token") or os.getenv("HF_TOKEN")

    if not audio_id or not profile_name or not turns:
        return web.json_response(
            {"error": "audio_id, profile, and turns are required"}, status=400
        )

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    try:
        speaker_turns = [
            SpeakerTurn(
                speaker_id=str(t["speaker_id"]),
                start_s=float(t["start_s"]),
                end_s=float(t["end_s"]),
            )
            for t in turns
        ]
    except (KeyError, TypeError, ValueError) as e:
        return web.json_response({"error": f"Invalid turns: {e}"}, status=400)

    diarization = DiarizationResult(
        schema_version="2.0",
        audio_id=audio.source_id,
        speakers=[
            Speaker(speaker_id=spk)
            for spk in sorted({t.speaker_id for t in speaker_turns})
        ],
        turns=speaker_turns,
        source_audio=audio,
        channel_id=audio.channel_id,
        channel_name=audio.channel_name,
        channel_url=audio.channel_url,
    )

    task_id = task_manager.create_task(
        "target_speaker_score",
        {"audio_id": audio_id, "profile": profile_name, "device": device},
    )

    async def run_score():
        target_device = get_default_device() if device == "auto" else device
        task_manager.update_task(
            task_id,
            status="running",
            message=(
                f"Scoring {len(speaker_turns)} segments against profile "
                f"'{profile_name}' on {target_device}..."
            ),
        )
        loop = asyncio.get_running_loop()

        def do_score():
            verifier = SpeakerVerifier(device=target_device, token=token)
            profile = verifier.load_profile(profile_name)
            with verifier:
                return verifier.score(audio, diarization, profile)

        try:
            start_time = time.time()
            scored = await loop.run_in_executor(None, do_score)
            if _task_is_cancelled(task_id):
                return
            elapsed = time.time() - start_time
            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                progress_known=True,
                message=(
                    f"Scored {len(scored.segments)} segments in {elapsed:.2f}s "
                    f"on {target_device}!"
                ),
                result={
                    "scored": asdict(scored),
                    "audio_id": audio_id,
                    "profile": profile_name,
                    "elapsed_s": round(elapsed, 2),
                    "device": target_device,
                },
            )
        except Exception as e:
            task = task_manager.get_task(task_id)
            if task and task["status"] == "cancelled":
                return
            logger.exception("Target speaker scoring failed")
            task_manager.update_task(
                task_id,
                status="failed",
                error=str(e),
                message=f"Target speaker scoring failed: {e}",
            )

    task_manager.enqueue(task_id, run_score)
    return web.json_response(
        {"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202
    )


async def handle_speaker_purity_config(request: web.Request) -> web.Response:
    """Return non-secret speaker-purity defaults for the web workbench."""
    configured_backend = os.getenv("OVERLAP_VERIFIER", "").strip().lower()
    if configured_backend in {"gemma", "gemma4", "gemma-4", "unsloth"}:
        configured_backend = "gemma4"
    elif configured_backend in {"gemini", "gemini-3.1", "gemini-3.1-pro"}:
        configured_backend = "gemini"
    elif configured_backend in {
        "gemini-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-3.1",
    }:
        configured_backend = "gemini-flash-lite"
    elif configured_backend in {"vibevoice", "vibevoice-asr"}:
        configured_backend = "vibevoice"
    else:
        configured_backend = "gemma4"

    host = os.getenv("UNSLOTH_HOST", "localhost").strip() or "localhost"
    port = os.getenv("UNSLOTH_PORT", "8888").strip() or "8888"
    unsloth_endpoint = os.getenv("UNSLOTH_ENDPOINT") or (
        f"http://{host}:{port}/v1/chat/completions"
    )
    return web.json_response(
        {
            "overlap_enabled": True,
            "overlap_backend": configured_backend,
            "overlap_prompt": OVERLAP_PROMPT,
            "overlap_timeout_s": 120.0,
            "overlap_max_output_tokens": DEFAULT_OVERLAP_MAX_OUTPUT_TOKENS,
            "gemma4": {
                "endpoint": unsloth_endpoint or DEFAULT_UNSLOTH_ENDPOINT,
                "model": os.getenv("UNSLOTH_MODEL") or DEFAULT_GEMMA4_MODEL_ID,
                "api_key_configured": bool(os.getenv("UNSLOTH_API_KEY")),
            },
            "gemini": {
                "model": os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL_ID,
                "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
            },
            "gemini-flash-lite": {
                "model": DEFAULT_GEMINI_FLASH_LITE_MODEL_ID,
                "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
            },
            "vibevoice": {
                "model": os.getenv("VIBEVOICE_MODEL") or DEFAULT_VIBEVOICE_MODEL_ID,
                "min_secondary_speech_s": DEFAULT_MIN_SECONDARY_SPEECH_S,
                "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                "batch_size": DEFAULT_VIBEVOICE_BATCH_SIZE,
                "models": vibevoice_studio_models(),
            },
        }
    )


async def handle_purity_verifier_status(request: web.Request) -> web.Response:
    """Probe whether the selected LLM verifier is reachable and ready."""
    backend = str(request.query.get("backend") or "gemma4").strip().lower()
    if backend in {"gemma", "gemma-4", "unsloth"}:
        backend = "gemma4"
    try:
        if backend == "vibevoice":
            verifier = VibeVoicePurityWorkerVerifier(
                model_id=str(
                    request.query.get("model") or DEFAULT_VIBEVOICE_MODEL_ID
                ),
                device=str(request.query.get("device") or "auto"),
            )
            status = await asyncio.to_thread(verifier.check_ready)
            return web.json_response({"backend": "vibevoice", **status})
        config: dict[str, Any] = {"backend": backend}
        model = request.query.get("model")
        if model:
            config["model"] = model
        if backend == "gemma4":
            endpoint = request.query.get("endpoint")
            if endpoint:
                config["endpoint"] = endpoint
        verifier = create_overlap_verifier(config)
        status = verifier.check_ready()
        return web.json_response({"backend": backend, **status})
    except (TypeError, ValueError) as exc:
        return web.json_response(
            {"backend": backend, "ready": False, "message": str(exc), "models": []}
        )
    except OverlapVerifierError as exc:
        return web.json_response(
            {
                "backend": backend,
                "ready": False,
                "message": str(exc),
                "models": [],
            }
        )
    except Exception as exc:
        logger.exception("Purity verifier status probe failed")
        return web.json_response(
            {
                "backend": backend,
                "ready": False,
                "message": f"Verifier probe failed: {exc}",
                "models": [],
            }
        )


async def handle_list_diarization_results(request: web.Request) -> web.Response:
    """List durable diarization results with source/model summaries."""
    verification = _verification_state_index()
    items = []
    for path in DIARIZATION_RESULTS_DIR.glob("*.json"):
        try:
            items.append(_catalog_item_from_path(path, verification))
        except Exception as exc:
            logger.warning("Ignoring invalid diarization result %s: %s", path, exc)
    items.sort(key=lambda item: _created_at_value(item), reverse=True)
    return _json_response({"results": items, "total": len(items)})


async def handle_list_diarization_annotations(request: web.Request) -> web.Response:
    """List durable manual ground-truth annotations."""
    items = []
    for path in DIARIZATION_ANNOTATIONS_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                items.append(_annotation_summary(payload))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid diarization annotation %s: %s", path, exc)
    items.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return _json_response({"annotations": items, "total": len(items)})


async def handle_get_diarization_annotation(request: web.Request) -> web.Response:
    """Return one complete manual annotation and its current session audio ID."""
    try:
        payload = _load_annotation(request.match_info["annotation_id"])
    except ValueError as exc:
        return _json_response({"error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return _json_response({"error": str(exc)}, status=404)
    except (OSError, json.JSONDecodeError) as exc:
        return _json_response({"error": str(exc)}, status=500)
    payload = dict(payload)
    try:
        payload["session_audio_id"] = _ensure_annotation_source_registered(payload)
    except Exception as exc:
        logger.warning("Could not restore annotation source: %s", exc)
        payload["session_audio_id"] = None
    return _json_response(payload)


async def handle_save_diarization_annotation(request: web.Request) -> web.Response:
    """Create or revision-save one validated manual annotation atomically."""
    try:
        data = await request.json()
        annotation_id = str(data.get("annotation_id") or "").strip()
        existing = _load_annotation(annotation_id) if annotation_id else None
        payload = _validated_annotation_payload(data, existing=existing)
        path = _annotation_path(payload["annotation_id"])
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(
                portable_data_payload(payload),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
        return _json_response(payload, status=200 if existing else 201)
    except RuntimeError as exc:
        return _json_response({"error": str(exc)}, status=409)
    except FileNotFoundError as exc:
        return _json_response({"error": str(exc)}, status=404)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _json_response({"error": str(exc)}, status=400)
    except OSError as exc:
        logger.exception("Could not save diarization annotation")
        return _json_response({"error": str(exc)}, status=500)


async def handle_delete_diarization_annotation(request: web.Request) -> web.Response:
    """Delete one persisted manual annotation."""
    try:
        path = _annotation_path(request.match_info["annotation_id"])
    except ValueError as exc:
        return _json_response({"error": str(exc)}, status=400)
    if not path.is_file():
        return _json_response({"error": "Annotation not found"}, status=404)
    try:
        path.unlink()
    except OSError as exc:
        return _json_response({"error": str(exc)}, status=500)
    return _json_response({"deleted": request.match_info["annotation_id"]})


async def handle_evaluate_diarization_results(request: web.Request) -> web.Response:
    """Evaluate compatible model results against one manual annotation."""
    try:
        data = await request.json()
        annotation_id = str(data.get("annotation_id") or "").strip()
        result_ids = data.get("result_ids")
        if not isinstance(result_ids, list) or not result_ids:
            raise ValueError("Select at least one diarization result")
        if len(result_ids) > 50:
            raise ValueError("At most 50 results can be evaluated at once")
        collar_s = _validated_float(
            data.get("collar_s", 0.0), "collar_s", minimum=0.0, maximum=10.0
        )
        skip_overlap = bool(data.get("skip_overlap", False))
        annotation = _load_annotation(annotation_id)
        duration_s = float((annotation.get("source_audio") or {}).get("duration_s"))
        reports = []
        for raw_result_id in result_ids:
            result_id = str(raw_result_id).strip()
            result = _load_diarization_result(result_id)
            matches, match_reason = _annotation_matches_result(annotation, result)
            if not matches:
                raise ValueError(f"{result_id}: {match_reason}")
            metrics = evaluate_diarization(
                annotation.get("turns") or [],
                result.turns,
                duration_s=duration_s,
                collar_s=collar_s,
                skip_overlap=skip_overlap,
            )
            reports.append(
                {
                    "result_id": result.result_id,
                    "model": asdict(result.model) if result.model else None,
                    "created_at": result.created_at,
                    "source_match": match_reason,
                    **metrics,
                }
            )
        reports.sort(key=lambda report: (report["der_pct"], report["jer_pct"]))
        return _json_response(
            {
                "annotation_id": annotation_id,
                "annotation_revision": annotation.get("revision"),
                "settings": {"collar_s": collar_s, "skip_overlap": skip_overlap},
                "results": reports,
            }
        )
    except FileNotFoundError as exc:
        return _json_response({"error": str(exc)}, status=404)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _json_response({"error": str(exc)}, status=400)


async def handle_get_diarization_result(request: web.Request) -> web.Response:
    """Return one complete canonical diarization result."""
    result_id = request.match_info["result_id"]
    try:
        path = _diarization_result_path(result_id)
    except ValueError as exc:
        return _json_response({"error": str(exc)}, status=400)
    if not path.is_file():
        return _json_response({"error": f"Diarization result not found: {result_id}"}, status=404)
    verification = _verification_state_index()
    try:
        payload = _catalog_item_from_path(
            path, verification, complete=True, hydrate_source=True
        )
        return _json_response(payload)
    except ValueError as exc:
        logger.exception("Could not serialize diarization result %s", result_id)
        return _json_response({"error": f"Result is not JSON-serializable: {exc}"}, status=500)
    except (TypeError, OSError, json.JSONDecodeError) as exc:
        return _json_response({"error": str(exc)}, status=400)


async def handle_delete_diarization_result(request: web.Request) -> web.Response:
    """Delete one persisted diarization result JSON file."""
    try:
        path = _diarization_result_path(request.match_info["result_id"])
    except ValueError as exc:
        return _json_response({"error": str(exc)}, status=400)
    if not path.is_file():
        return _json_response(
            {"error": f"Diarization result not found: {request.match_info['result_id']}"},
            status=404,
        )
    path.unlink()
    return _json_response({"deleted": request.match_info["result_id"]})


async def handle_clear_diarization_results(request: web.Request) -> web.Response:
    """Delete every persisted diarization result JSON file."""
    deleted = 0
    for path in list(DIARIZATION_RESULTS_DIR.glob("*.json")):
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            logger.warning("Could not delete diarization result %s: %s", path, exc)
    return _json_response({"cleared": deleted})


async def handle_preview_diarization_turn(request: web.Request) -> web.StreamResponse:
    """Lazily cut and stream one turn without registering a new audio item."""
    try:
        result = _load_diarization_result(request.match_info["result_id"])
        turn_index = int(request.match_info["turn_index"])
        turn = result.turns[turn_index]
        if result.source_audio is None:
            raise ValueError("Diarization result has no source audio reference")
        if not Path(result.source_audio.path).is_file():
            raise FileNotFoundError(f"Source audio is unavailable: {result.source_audio.path}")
    except (IndexError, ValueError, TypeError) as exc:
        return web.Response(text=str(exc), status=400)
    except FileNotFoundError as exc:
        return web.Response(text=str(exc), status=404)

    result_dir = DIARIZATION_PREVIEW_DIR / result.result_id
    output_path = result_dir / f"turn_{turn_index:06d}.wav"
    if not output_path.is_file():
        cutter = AudioCutter(output_dir=result_dir)
        await asyncio.to_thread(
            cutter.cut,
            result.source_audio,
            turn.start_s,
            turn.end_s,
            output_path=output_path,
        )
    return web.FileResponse(output_path)


async def handle_verify_diarization_batch(request: web.Request) -> web.Response:
    """Verify filtered turns from one or more persisted diarization results."""
    data = await request.json()
    result_ids = list(dict.fromkeys(
        str(value) for value in data.get("result_ids", []) if value
    ))
    profile_name = (
        str(data.get("profile") or data.get("profile_name") or "").strip()
        or "unlabeled"
    )
    if not result_ids:
        return web.json_response({"error": "result_ids are required"}, status=400)
    try:
        results = [_load_diarization_result(result_id) for result_id in result_ids]
        min_duration_s = float(data.get("min_duration_s", 1.5))
        max_duration_value = data.get("max_duration_s")
        max_duration_s = float(max_duration_value) if max_duration_value not in (None, "") else None
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)

    speaker_ids = {str(value) for value in data.get("speaker_ids", []) if value}
    overlap_state = str(data.get("overlap_state", "any"))
    verification_filter = str(data.get("verification_state", "all"))
    if min_duration_s <= 0 or (max_duration_s is not None and max_duration_s <= 0):
        return web.json_response({"error": "Turn durations must be greater than zero"}, status=400)
    if max_duration_s is not None and max_duration_s < min_duration_s:
        return web.json_response({"error": "max_duration_s must be at least min_duration_s"}, status=400)
    if overlap_state not in {"any", "exclude", "only"}:
        return web.json_response({"error": "Invalid overlap_state"}, status=400)
    if verification_filter not in {"all", "unverified", "pass", "reject", "error"}:
        return web.json_response({"error": "Invalid verification_state"}, status=400)
    try:
        overlap_config, overlap_failure_policy = _parse_overlap_verifier_request(
            data.get("overlap_verifier")
        )
    except (TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    previous_states = _verification_state_index(profile_name)
    candidates_by_result: dict[str, list[SpeakerTurn]] = {}
    for result in results:
        previous_turns = previous_states.get(result.result_id, {}).get("turns", {})
        candidates = []
        for turn in result.turns:
            duration_s = turn.duration_s
            prior = previous_turns.get(
                _turn_key(turn.speaker_id, turn.start_s, turn.end_s), "unverified"
            )
            if speaker_ids and turn.speaker_id not in speaker_ids:
                continue
            if duration_s < min_duration_s or (max_duration_s is not None and duration_s > max_duration_s):
                continue
            if overlap_state == "exclude" and turn.overlaps_other_speaker:
                continue
            if overlap_state == "only" and not turn.overlaps_other_speaker:
                continue
            if verification_filter != "all" and prior != verification_filter:
                continue
            candidates.append(turn)
        candidates_by_result[result.result_id] = candidates

    total_candidates = sum(len(turns) for turns in candidates_by_result.values())
    if total_candidates == 0:
        return web.json_response({"error": "No turns match the selected filters"}, status=400)

    device = str(data.get("device", "auto"))
    target_device = get_default_device() if device == "auto" else device
    token = data.get("token") or os.getenv("HF_TOKEN")
    overlap_backend = str(overlap_config["backend"])
    task_id = task_manager.create_task(
        "diarization_batch_verify",
        {
            "result_ids": result_ids,
            "profile": profile_name,
            "device": target_device,
            "candidate_count": total_candidates,
            "overlap_verifier": overlap_backend,
        },
    )

    async def run_batch() -> None:
        task_manager.update_task(
            task_id,
            status="running",
            progress=0.0,
            progress_known=True,
            message=(
                f"Checking {total_candidates} candidate(s) directly with "
                f"{overlap_backend}..."
            ),
        )
        collected: list[dict[str, Any]] = []
        started_at = time.time()
        overlap_verifier = None
        verifier_status: dict[str, Any] = {
            "ready": True,
            "message": "",
            "models": [],
        }
        try:
            work_dir = DATA_DIR / "purity" / "work" / task_id
            cutter = AudioCutter(output_dir=work_dir)
            loop = asyncio.get_running_loop()
            completed = 0
            use_vibevoice = overlap_backend == "vibevoice"
            vibe_verifier = None
            overlap_model = str(overlap_config.get("model") or "")
            if use_vibevoice:
                vibe_verifier = _vibevoice_verifier_from_config(
                    overlap_config,
                    device=target_device,
                    token=token,
                )
                await asyncio.to_thread(vibe_verifier.load)
                overlap_verifier = vibe_verifier
                overlap_model = str(
                    overlap_config.get("model") or DEFAULT_VIBEVOICE_MODEL_ID
                )
                verifier_status = {
                    "ready": True,
                    "message": f"Loaded {overlap_model}",
                    "models": [overlap_model],
                }
            else:
                overlap_verifier = create_overlap_verifier(overlap_config)
                overlap_model = str(
                    getattr(
                        overlap_verifier,
                        "model",
                        overlap_config.get("model", ""),
                    )
                )
                task_manager.update_task(
                    task_id,
                    message=f"Probing {overlap_backend} readiness...",
                )
                verifier_status = await loop.run_in_executor(
                    None,
                    _probe_overlap_verifier_or_raise,
                    overlap_verifier,
                    overlap_backend,
                )
                task_manager.update_task(
                    task_id,
                    message=(
                        f"{overlap_backend} is ready. Checking candidates..."
                    ),
                )
            try:
                for result in results:
                    if _task_is_cancelled(task_id):
                        return
                    candidates = candidates_by_result[result.result_id]
                    if not candidates:
                        continue
                    if result.source_audio is None:
                        raise ValueError(
                            f"Result {result.result_id} has no source audio"
                        )
                    audio = result.source_audio
                    if use_vibevoice:
                        batch_items: list[dict[str, Any]] = []
                        batch_segments: list[Audio] = []
                        for turn in candidates:
                            item = _direct_audio_purity_item(
                                audio, result, turn, profile_name
                            )
                            item["result_id"] = result.result_id
                            item["source_title"] = audio.title
                            item["turn_index"] = result.turns.index(turn)
                            output_path = (
                                work_dir
                                / f"candidate_{completed + len(batch_items) + 1:05d}.wav"
                            )
                            batch_segments.append(
                                cutter.cut(
                                    audio,
                                    turn.start_s,
                                    turn.end_s,
                                    output_path=output_path,
                                )
                            )
                            batch_items.append(item)
                        try:
                            batch_results = await loop.run_in_executor(
                                None, vibe_verifier.verify_batch, batch_segments
                            )
                            for item, direct_result in zip(
                                batch_items, batch_results, strict=True
                            ):
                                _apply_vibevoice_purity_item(item, direct_result)
                        except Exception as exc:
                            error_text = f"{type(exc).__name__}: {exc}"
                            for item in batch_items:
                                item["decision"] = (
                                    "error"
                                    if overlap_failure_policy == "fail_closed"
                                    else "pass"
                                )
                                item["reason"] = (
                                    "vibevoice_verification_failed"
                                    if overlap_failure_policy == "fail_closed"
                                    else None
                                )
                                item["error"] = error_text
                                item["passed"] = item["decision"] == "pass"
                        collected.extend(batch_items)
                        completed += len(batch_items)
                        task_manager.update_task(
                            task_id,
                            progress=completed / total_candidates,
                            progress_known=True,
                            message=(
                                f"Direct audio check {completed}/"
                                f"{total_candidates} with {overlap_backend}"
                            ),
                        )
                        continue
                    for turn in candidates:
                        if _task_is_cancelled(task_id):
                            return
                        item = _direct_audio_purity_item(
                            audio, result, turn, profile_name
                        )
                        item["result_id"] = result.result_id
                        item["source_title"] = audio.title
                        item["turn_index"] = result.turns.index(turn)
                        output_path = (
                            work_dir / f"candidate_{completed + 1:05d}.wav"
                        )

                        def verify_candidate(
                            turn=turn, output_path=output_path, audio=audio
                        ):
                            segment = cutter.cut(
                                audio,
                                turn.start_s,
                                turn.end_s,
                                output_path=output_path,
                            )
                            if use_vibevoice:
                                return vibe_verifier.verify(segment)
                            return overlap_verifier.verify(segment)

                        try:
                            direct_result = await loop.run_in_executor(
                                None, verify_candidate
                            )
                            if use_vibevoice:
                                _apply_vibevoice_purity_item(item, direct_result)
                            else:
                                _apply_direct_overlap_decision(
                                    item,
                                    backend=str(overlap_backend),
                                    model=overlap_model,
                                    overlap=direct_result["overlap"],
                                    reason=direct_result["reason"],
                                    error=None,
                                    failure_policy=overlap_failure_policy,
                                )
                        except Exception as exc:
                            if is_overlap_readiness_error(exc):
                                raise OverlapVerifierError(
                                    str(exc),
                                    readiness=True,
                                ) from exc
                            error_text = f"{type(exc).__name__}: {exc}"
                            if use_vibevoice:
                                item["decision"] = (
                                    "error"
                                    if overlap_failure_policy == "fail_closed"
                                    else "pass"
                                )
                                item["reason"] = (
                                    "vibevoice_verification_failed"
                                    if overlap_failure_policy == "fail_closed"
                                    else None
                                )
                                item["error"] = error_text
                                item["passed"] = item["decision"] == "pass"
                            else:
                                _apply_direct_overlap_decision(
                                    item,
                                    backend=str(overlap_backend),
                                    model=overlap_model,
                                    overlap=None,
                                    reason=None,
                                    error=error_text,
                                    failure_policy=overlap_failure_policy,
                                )
                        collected.append(item)
                        completed += 1
                        task_manager.update_task(
                            task_id,
                            progress=completed / total_candidates,
                            progress_known=True,
                            message=(
                                f"Direct audio check {completed}/"
                                f"{total_candidates} with {overlap_backend}"
                            ),
                        )
            finally:
                if vibe_verifier is not None:
                    await asyncio.to_thread(vibe_verifier.unload)
                shutil.rmtree(work_dir, ignore_errors=True)
            verification_id = f"verify_{uuid.uuid4().hex}"
            counts = {
                decision: sum(item["decision"] == decision for item in collected)
                for decision in ("pass", "reject", "uncertain", "error")
            }
            error_summary = _verifier_error_summary(collected)
            report = {
                "kind": "diarization.verification.batch",
                "schema_version": "1.0",
                "verification_id": verification_id,
                "created_at": time.time(),
                "result_ids": result_ids,
                "profile": profile_name,
                "results": collected,
                "counts": counts,
                "verifier_status": {
                    **verifier_status,
                    **error_summary,
                },
                "settings": {
                    "min_duration_s": min_duration_s,
                    "max_duration_s": max_duration_s,
                    "speaker_ids": sorted(speaker_ids),
                    "overlap_state": overlap_state,
                    "verification_state": verification_filter,
                    "overlap_verifier": _overlap_verifier_report_settings(
                        overlap_config,
                        overlap_failure_policy,
                        overlap_verifier,
                    ),
                },
            }
            report_path = DIARIZATION_VERIFICATIONS_DIR / f"{verification_id}.json"
            report_temp_path = report_path.with_suffix(".json.tmp")
            report_temp_path.write_text(
                json.dumps(portable_data_payload(report), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            report_temp_path.replace(report_path)
            for result in results:
                if result.source_audio is None:
                    continue
                active_audio_id = registry.find_id_by_path(result.source_audio.path)
                active_item = registry.get_item(active_audio_id) if active_audio_id else None
                if active_item:
                    result_rows = [
                        row for row in collected if row["result_id"] == result.result_id
                    ]
                    if not result_rows:
                        continue
                    verification_tag = (
                        "verification:passed"
                        if any(row["decision"] == "pass" for row in result_rows)
                        else "verification:rejected"
                    )
                    active_item["system_tags"] = sorted(
                        (set(active_item.get("system_tags", [])) - {"verification:unverified", "verification:passed", "verification:rejected"})
                        | {"stage:verified", f"profile:{profile_name}", verification_tag}
                    )

            # Reflect durable verification metadata in Pipeline when its source
            # item is registered; no segment audio is registered or exported.
            try:
                from src.web_pipeline.dataset_manager import dataset_manager
                for result in results:
                    if result.source_audio is None:
                        continue
                    result_rows = [
                        row for row in collected if row["result_id"] == result.result_id
                    ]
                    if not result_rows:
                        continue
                    item = dataset_manager.find_item_by_path(result.source_audio.path)
                    if item:
                        dataset_manager.attach_target_speaker(
                            item.id,
                            {
                                "profile": profile_name,
                                "passed_candidates": sum(
                                    row["decision"] == "pass" for row in result_rows
                                ),
                                "verification_id": verification_id,
                            },
                        )
            except Exception:
                logger.exception("Could not synchronize verification metadata to Pipeline")

            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                progress_known=True,
                message=(
                    f"Batch complete: {counts['pass']} passed, "
                    f"{counts['reject']} rejected, "
                    f"{counts['uncertain']} uncertain, "
                    f"{counts['error']} errors"
                ),
                result={**report, "elapsed_s": round(time.time() - started_at, 2)},
            )
        except Exception as exc:
            if _task_is_cancelled(task_id):
                return
            logger.exception("Diarization batch verification failed")
            if is_overlap_readiness_error(exc):
                message = f"{overlap_backend} is not ready: {exc}"
            else:
                message = f"Batch verification failed: {exc}"
            task_manager.update_task(
                task_id,
                status="failed",
                error=message,
                message=message,
            )

    task_manager.enqueue(task_id, run_batch)
    return web.json_response(
        {"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202
    )


async def handle_verify_speaker_purity(request: web.Request) -> web.Response:
    """Verify speaker purity of a chosen track with the LLM verifier.

    ``turns`` may be omitted or empty: the whole file is then one candidate.
    """
    data = await request.json()
    audio_id = data.get("audio_id")
    profile_name = str(data.get("profile") or data.get("profile_name") or "").strip() or "unlabeled"
    turns = data.get("turns", [])
    device = data.get("device", "auto")
    token = data.get("token") or os.getenv("HF_TOKEN")

    try:
        min_candidate_duration_s = float(
            data.get("min_candidate_duration_s", 1.5)
        )
    except (TypeError, ValueError) as e:
        return web.json_response(
            {"error": f"Invalid speaker purity numeric setting: {e}"}, status=400
        )
    try:
        overlap_config, overlap_failure_policy = _parse_overlap_verifier_request(
            data.get("overlap_verifier")
        )
    except (TypeError, ValueError) as e:
        return web.json_response({"error": str(e)}, status=400)

    if not audio_id:
        return web.json_response({"error": "audio_id is required"}, status=400)

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    if not turns:
        try:
            duration_s = _audio_duration_s(audio)
            speaker_turns = [
                SpeakerTurn(speaker_id="clip", start_s=0.0, end_s=duration_s)
            ]
        except FileNotFoundError as e:
            return web.json_response({"error": str(e)}, status=404)
        except (TypeError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
    else:
        try:
            speaker_turns = [
                SpeakerTurn(
                    speaker_id=str(t["speaker_id"]),
                    start_s=float(t["start_s"]),
                    end_s=float(t["end_s"]),
                )
                for t in turns
            ]
        except (KeyError, TypeError, ValueError) as e:
            return web.json_response({"error": f"Invalid turns: {e}"}, status=400)

        if min_candidate_duration_s > 0:
            speaker_turns = [
                turn
                for turn in speaker_turns
                if (turn.end_s - turn.start_s) >= min_candidate_duration_s
            ]
        if not speaker_turns:
            return web.json_response(
                {"error": "No turns remain after the duration filter"},
                status=400,
            )

    diarization = DiarizationResult(
        schema_version="2.0",
        audio_id=audio.source_id,
        speakers=[
            Speaker(speaker_id=spk)
            for spk in sorted({t.speaker_id for t in speaker_turns})
        ],
        turns=speaker_turns,
        source_audio=audio,
        channel_id=audio.channel_id,
        channel_name=audio.channel_name,
        channel_url=audio.channel_url,
    )

    task_id = task_manager.create_task(
        "speaker_purity_verify",
        {
            "audio_id": audio_id,
            "profile": profile_name,
            "device": device,
            "turns_count": len(speaker_turns),
            "overlap_verifier": overlap_config["backend"],
            "backend": overlap_config["backend"],
        },
    )

    async def run_verify():
        target_device = get_default_device() if device == "auto" else device
        loop = asyncio.get_running_loop()
        backend = str(overlap_config["backend"])
        use_vibevoice = backend == "vibevoice"
        vibe_verifier = None
        verifier_status: dict[str, Any] = {
            "ready": True,
            "message": "",
            "models": [],
        }

        try:
            start_time = time.time()
            if use_vibevoice:
                verifier = _vibevoice_verifier_from_config(
                    overlap_config,
                    device=target_device,
                    token=token,
                )
                await asyncio.to_thread(verifier.load)
                vibe_verifier = verifier
                model = str(
                    overlap_config.get("model") or DEFAULT_VIBEVOICE_MODEL_ID
                )
                verifier_status = {
                    "ready": True,
                    "message": f"Loaded {model}",
                    "models": [model],
                }
            else:
                verifier = create_overlap_verifier(overlap_config)
                model = str(
                    getattr(verifier, "model", overlap_config.get("model", ""))
                )
                task_manager.update_task(
                    task_id,
                    status="running",
                    progress=0.0,
                    progress_known=True,
                    message=f"Probing {backend} readiness...",
                )
                verifier_status = await loop.run_in_executor(
                    None,
                    _probe_overlap_verifier_or_raise,
                    verifier,
                    backend,
                )
            task_manager.update_task(
                task_id,
                status="running",
                progress=0.0,
                progress_known=True,
                message=(
                    f"Checking {len(speaker_turns)} candidate(s) directly "
                    f"with {backend}..."
                ),
            )
            work_dir = DATA_DIR / "purity" / "work" / task_id
            cutter = AudioCutter(output_dir=work_dir)
            serialized_results = []
            try:
                if use_vibevoice:
                    batch_items = []
                    batch_segments = []
                    for position, turn in enumerate(speaker_turns, start=1):
                        item = _direct_audio_purity_item(
                            audio, diarization, turn, profile_name
                        )
                        output_path = work_dir / f"candidate_{position:05d}.wav"
                        batch_segments.append(
                            cutter.cut(
                                audio,
                                turn.start_s,
                                turn.end_s,
                                output_path=output_path,
                            )
                        )
                        batch_items.append(item)
                    try:
                        batch_results = await loop.run_in_executor(
                            None, verifier.verify_batch, batch_segments
                        )
                        for item, direct_result in zip(
                            batch_items, batch_results, strict=True
                        ):
                            _apply_vibevoice_purity_item(item, direct_result)
                    except Exception as exc:
                        error_text = f"{type(exc).__name__}: {exc}"
                        for item in batch_items:
                            item["error"] = error_text
                            item["decision"] = (
                                "error"
                                if overlap_failure_policy == "fail_closed"
                                else "pass"
                            )
                            item["reason"] = (
                                "vibevoice_verification_failed"
                                if overlap_failure_policy == "fail_closed"
                                else None
                            )
                            item["passed"] = item["decision"] == "pass"
                    serialized_results.extend(batch_items)
                    task_manager.update_task(
                        task_id,
                        progress=1.0,
                        progress_known=True,
                        message=(
                            f"Direct audio check {len(speaker_turns)}/"
                            f"{len(speaker_turns)} with {backend}"
                        ),
                    )
                for position, turn in enumerate(speaker_turns, start=1):
                    if use_vibevoice:
                        break
                    if _task_is_cancelled(task_id):
                        return
                    item = _direct_audio_purity_item(
                        audio, diarization, turn, profile_name
                    )
                    output_path = work_dir / f"candidate_{position:05d}.wav"

                    def verify_candidate(turn=turn, output_path=output_path):
                        segment = cutter.cut(
                            audio,
                            turn.start_s,
                            turn.end_s,
                            output_path=output_path,
                        )
                        return verifier.verify(segment)

                    try:
                        direct_result = await loop.run_in_executor(
                            None, verify_candidate
                        )
                        if use_vibevoice:
                            _apply_vibevoice_purity_item(item, direct_result)
                        else:
                            _apply_direct_overlap_decision(
                                item,
                                backend=backend,
                                model=model,
                                overlap=direct_result["overlap"],
                                reason=direct_result["reason"],
                                error=None,
                                failure_policy=overlap_failure_policy,
                            )
                    except Exception as e:
                        if is_overlap_readiness_error(e):
                            raise OverlapVerifierError(
                                str(e),
                                readiness=True,
                            ) from e
                        error_text = f"{type(e).__name__}: {e}"
                        if use_vibevoice:
                            item["error"] = error_text
                            item["decision"] = (
                                "error"
                                if overlap_failure_policy == "fail_closed"
                                else "pass"
                            )
                            item["reason"] = (
                                "vibevoice_verification_failed"
                                if overlap_failure_policy == "fail_closed"
                                else None
                            )
                            item["passed"] = item["decision"] == "pass"
                        else:
                            _apply_direct_overlap_decision(
                                item,
                                backend=backend,
                                model=model,
                                overlap=None,
                                reason=None,
                                error=error_text,
                                failure_policy=overlap_failure_policy,
                            )
                    serialized_results.append(item)
                    task_manager.update_task(
                        task_id,
                        progress=position / len(speaker_turns),
                        progress_known=True,
                        message=(
                            f"Direct audio check {position}/{len(speaker_turns)} "
                            f"with {backend}"
                        ),
                    )
            finally:
                if vibe_verifier is not None:
                    await asyncio.to_thread(vibe_verifier.unload)
                shutil.rmtree(work_dir, ignore_errors=True)

            for item in serialized_results:
                item["passed"] = item["decision"] == "pass"
            elapsed = time.time() - start_time

            passed_results = [
                item for item in serialized_results if item["decision"] == "pass"
            ]
            total_duration_s = sum(item["duration_s"] for item in serialized_results)
            passed_duration_s = sum(item["duration_s"] for item in passed_results)
            direct_overlap_results = [
                item["direct_overlap"]
                for item in serialized_results
                if item.get("direct_overlap") is not None
            ]
            vibevoice_results = [
                item["vibevoice"]
                for item in serialized_results
                if item.get("vibevoice") is not None
            ]
            error_summary = _verifier_error_summary(serialized_results)

            reasons_count: dict[str, int] = {}
            for item in serialized_results:
                if item["reason"]:
                    reasons_count[item["reason"]] = reasons_count.get(item["reason"], 0) + 1

            task_manager.update_task(
                task_id,
                status="completed",
                progress=1.0,
                progress_known=True,
                message=(
                    f"Purity verification complete: {len(passed_results)}/"
                    f"{len(serialized_results)} passed "
                    f"({passed_duration_s:.1f}s pure speech) in {elapsed:.2f}s"
                    + (
                        f" — {error_summary['error_count']} verifier error(s)"
                        if error_summary["error_count"]
                        else ""
                    )
                ),
                result={
                    "purity_results": serialized_results,
                    "audio_id": audio_id,
                    "profile": profile_name,
                    "elapsed_s": round(elapsed, 2),
                    "device": target_device,
                    "verifier_status": {
                        **verifier_status,
                        **error_summary,
                    },
                    "metrics": {
                        "total_candidates": len(serialized_results),
                        "passed_candidates": len(passed_results),
                        "pass_rate_percent": (
                            (len(passed_results) / len(serialized_results) * 100)
                            if serialized_results
                            else 0.0
                        ),
                        "total_duration_s": round(total_duration_s, 2),
                        "passed_duration_s": round(passed_duration_s, 2),
                        "duration_pass_percent": (
                            (passed_duration_s / total_duration_s * 100)
                            if total_duration_s > 0
                            else 0.0
                        ),
                        "reasons_breakdown": reasons_count,
                        "direct_overlap_checked": len(direct_overlap_results),
                        "direct_overlap_detected": sum(
                            result["overlap"] is True
                            for result in direct_overlap_results
                        ),
                        "direct_overlap_errors": sum(
                            bool(result["error"])
                            for result in direct_overlap_results
                        ),
                        "vibevoice_checked": len(vibevoice_results),
                        "uncertain_candidates": sum(
                            item["decision"] == "uncertain"
                            for item in serialized_results
                        ),
                    },
                    "settings": {
                        "min_candidate_duration_s": min_candidate_duration_s,
                        "overlap_verifier": _overlap_verifier_report_settings(
                            overlap_config,
                            overlap_failure_policy,
                            verifier,
                        ),
                    },
                },
            )
        except Exception as e:
            task = task_manager.get_task(task_id)
            if task and task["status"] == "cancelled":
                return
            logger.exception("Speaker purity verification failed")
            if is_overlap_readiness_error(e):
                message = f"{backend} is not ready: {e}"
            else:
                message = f"Speaker purity verification failed: {e}"
            task_manager.update_task(
                task_id,
                status="failed",
                error=message,
                message=message,
            )

    task_manager.enqueue(task_id, run_verify)
    return web.json_response(
        {"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202
    )


async def handle_export_purity_audio(request: web.Request) -> web.Response:
    """Export passed or selected pure speaker segments into a concatenated or time-aligned audio file."""
    data = await request.json()
    audio_id = data.get("audio_id")
    segments = data.get("segments", [])
    mode = data.get("mode", "concat")  # 'concat' or 'time_aligned'
    profile_name = data.get("profile_name", "pure_speaker")

    if not audio_id:
        return web.json_response({"error": "audio_id is required"}, status=400)
    if not segments:
        return web.json_response({"error": "segments list is empty"}, status=400)

    audio = registry.get_audio(audio_id)
    if not audio:
        return web.json_response({"error": "Audio not found"}, status=404)

    try:
        extraction_settings = _extraction_settings(data)
    except (TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)

    out_dir = DATA_DIR / "purity" / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    def do_extract():
        src_path = Path(audio.path)
        waveform, sr = sf.read(str(src_path), always_2d=True)
        total_frames = waveform.shape[0]

        valid_intervals = _padded_audio_intervals(
            segments,
            sample_rate=sr,
            total_frames=total_frames,
            blocker_turns=_request_blocker_turns(
                data, segments, exclude_speaker_id=None
            ),
            **extraction_settings,
        )

        if not valid_intervals:
            raise ValueError("No valid audio intervals found in segments")

        valid_intervals.sort(key=lambda x: x[0])
        mode_suffix = "aligned" if mode == "time_aligned" else "concat"

        if mode == "time_aligned":
            combined = np.zeros_like(waveform)
            for s_f, e_f in valid_intervals:
                combined[s_f:e_f] = waveform[s_f:e_f]
        else:
            slices = [waveform[s_f:e_f] for s_f, e_f in valid_intervals]
            combined = np.concatenate(slices, axis=0)

        sanitized_title = _sanitize_filename_component(audio.title or audio.source_id)
        sanitized_profile = _sanitize_filename_component(profile_name)
        out_filename = f"{sanitized_title}_{sanitized_profile}_pure_{mode_suffix}.wav"
        out_path = out_dir / out_filename
        sf.write(str(out_path), combined, sr)

        pure_audio = Audio.from_file(
            out_path,
            source_id=f"{audio.source_id}_{sanitized_profile}_pure_{mode_suffix}",
            title=f"{audio.title or audio.source_id} [Pure {profile_name}] ({mode_suffix})",
            source_url=audio.source_url,
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
            native_sample_rate=audio.native_sample_rate,
            history=(*audio.history, f"purity_extract_{profile_name}_{mode}"),
        )
        return pure_audio

    try:
        loop = asyncio.get_running_loop()
        extracted_audio = await loop.run_in_executor(None, do_extract)
        new_id = registry.register(
            extracted_audio,
            source_type="purity_stem",
            parent_id=audio_id,
            tags=["purity", f"profile:{profile_name}"],
        )
        return web.json_response({
            "audio_id": new_id,
            "profile_name": profile_name,
            "metadata": extracted_audio.metadata(),
            "duration_s": extracted_audio.duration_s,
            "segments_count": len(segments),
            "extraction_settings": extraction_settings,
            "status": "success",
        })
    except Exception as e:
        logger.exception("Purity audio export failed")
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

        # Light theme styling for figure
        fig.patch.set_facecolor("#ffffff")
        for ax in fig.axes:
            ax.set_facecolor("#f8fafc")
            ax.tick_params(colors="#334155")
            ax.xaxis.label.set_color("#334155")
            ax.yaxis.label.set_color("#334155")
            ax.title.set_color("#0f172a")
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1")

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
    """Cancel a queued or running Studio task."""
    task_id = request.match_info["id"]
    task = task_manager.get_task(task_id)
    if not task:
        return web.json_response({"error": "Task not found"}, status=404)
    if not task_manager.cancel_task(task_id):
        return web.json_response(
            {"error": "Task could not be cancelled", "task": task},
            status=409,
        )
    return web.json_response(task_manager.get_task(task_id))


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

    gpu = telemetry.get("gpu") or {}
    aggregate = gpu.get("aggregate") or {}
    is_multi_gpu = (gpu.get("device_count") or 0) > 1

    gpu_load_pct = (
        aggregate.get("avg_load_percent")
        if is_multi_gpu
        else gpu.get("load_percent")
    )
    vram_used_mb = (
        aggregate.get("used_vram_mb")
        if is_multi_gpu
        else gpu.get("used_vram_mb")
    )
    vram_total_mb = (
        aggregate.get("total_vram_mb")
        if is_multi_gpu
        else gpu.get("total_vram_mb")
    )
    vram_pct = (
        aggregate.get("vram_percent")
        if is_multi_gpu
        else gpu.get("vram_percent")
    )
    power_w = (
        aggregate.get("total_power_w")
        if is_multi_gpu
        else gpu.get("power_w")
    )
    power_limit_w = (
        aggregate.get("total_power_limit_w")
        if is_multi_gpu
        else gpu.get("power_limit_w")
    )
    power_pct = (
        aggregate.get("power_percent")
        if is_multi_gpu
        else gpu.get("power_percent")
    )

    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    studio_tasks = task_manager.list_tasks()
    studio_status = task_manager.status()

    pipeline_jobs: List[Dict[str, Any]] = []
    pipeline_status = {
        "running": 0,
        "pending": 0,
        "queued": 0,
        "is_paused": False,
        "max_concurrency": 1,
        "workers_per_device": 1,
        "device_queues": {},
    }
    try:
        from src.web_pipeline.queue_manager import queue_manager
        pipeline_jobs = queue_manager.list_jobs(limit=50)
        pipeline_status = queue_manager.status()
    except Exception:
        pass

    unified_items = []

    for t in studio_tasks:
        meta = t.get("metadata", {}) or {}
        unified_items.append({
            "id": t["id"],
            "source": "studio",
            "source_label": "SonicStudio",
            "title": meta.get("title") or meta.get("model") or t.get("type", "Studio Task"),
            "type": t.get("type", "studio_task"),
            "status": t.get("status", "pending"),
            "progress": round(min(100.0, max(0.0, float(t.get("progress") or 0.0) * 100.0)), 1),
            "progress_known": bool(t.get("progress_known")) or t.get("status") == "completed",
            "message": t.get("message", ""),
            "error": t.get("error"),
            "created_at": t.get("created_at"),
            "start_time": t.get("start_time"),
            "end_time": t.get("end_time"),
            "queue_position": t.get("queue_position"),
            "device": t.get("queue_device") or meta.get("queue_device") or meta.get("device"),
            "metadata": meta,
        })

    for j in pipeline_jobs:
        params = j.get("params", {}) or {}
        unified_items.append({
            "id": j["id"],
            "source": "pipeline",
            "source_label": "SonicPipeline",
            "title": j.get("title", "Batch Job"),
            "type": j.get("type", "batch_job"),
            "status": j.get("status", "pending"),
            "progress": j.get("progress", 0.0),
            "progress_known": bool(j.get("progress_known")) or j.get("status") == "completed",
            "message": j.get("current_step", ""),
            "error": j.get("error"),
            "created_at": j.get("created_at"),
            "start_time": j.get("started_at"),
            "end_time": j.get("finished_at"),
            "processed_items": j.get("processed_items", 0),
            "total_items": j.get("total_items", 0),
            "device": j.get("queue_device") or params.get("queue_device") or params.get("device"),
            "metadata": params,
        })

    unified_items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)

    total_running = studio_status["running"] + pipeline_status["running"]
    total_queued = studio_status["queued"] + pipeline_status["pending"]

    device_queues: Dict[str, Dict[str, Any]] = {}
    for source_status in (studio_status, pipeline_status):
        for device, lane in (source_status.get("device_queues") or {}).items():
            bucket = device_queues.setdefault(
                device,
                {"device": device, "running": 0, "queued": 0, "workers": lane.get("workers", 1)},
            )
            bucket["running"] += int(lane.get("running", 0))
            bucket["queued"] += int(lane.get("queued", 0))
            bucket["workers"] = max(int(bucket.get("workers", 1)), int(lane.get("workers", 1)))

    return {
        "device": {
            "name": gpu.get("name") or device_name,
            "cuda_available": torch.cuda.is_available(),
            "gpu_load_pct": gpu_load_pct,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "vram_pct": vram_pct,
            "power_w": power_w,
            "power_limit_w": power_limit_w,
            "power_pct": power_pct,
            "devices": gpu.get("devices", []),
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
        "device_queues": device_queues,
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
            return web.json_response({"error": "Could not cancel studio task"}, status=409)
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
        {
            "audio_id": audio_id,
            "models_count": len(models),
            "clip_title": audio.title,
            "device": device,
        },
    )

    async def run_batch():
        results = []
        target_device = get_default_device() if device == "auto" else device
        total = len(models)
        
        for idx, m_spec in enumerate(models, start=1):
            task = task_manager.get_task(task_id)
            if task and task["status"] == "cancelled":
                break
            m_type = m_spec.get("model_type", "htdemucs")
            m_name = m_spec.get("model_name")
            m_label = m_spec.get("label") or f"{m_type}_{m_name or 'default'}"
            
            task_manager.update_task(
                task_id,
                status="running",
                progress=round((idx - 1) / total, 2),
                progress_known=True,
                message=f"[{idx}/{total}] Separating with {m_label} on {target_device}...",
            )

            try:
                loop = asyncio.get_running_loop()
                def do_sep(curr_type=m_type, curr_name=m_name):
                    if target_device.startswith("cuda:") and torch.cuda.is_available():
                        try:
                            torch.cuda.set_device(int(target_device.split(":")[1]))
                        except Exception:
                            pass

                    if curr_type == "htdemucs":
                        sep = HTDemucs(
                            model=curr_name or "htdemucs",
                            device=target_device,
                            two_stems=two_stems,
                            output_dir=DATA_DIR / "demucs" / "out",
                            work_dir=DATA_DIR / "demucs" / "work",
                            progress_callback=_cli_progress_reporter(task_id, loop, "Demucs: "),
                        )
                        task_manager.set_cancel_callback(task_id, sep.cancel)
                        try:
                            return sep.separate(audio)
                        finally:
                            task_manager.set_cancel_callback(task_id, None)
                            sep.close()
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
                        def report_mvsep_progress(message: str) -> None:
                            def update_progress() -> None:
                                task = task_manager.get_task(task_id)
                                if task and task["status"] == "running":
                                    task_manager.update_task(
                                        task_id,
                                        message=f"[{idx}/{total}] MVSep-MDX23: {message}",
                                    )

                            loop.call_soon_threadsafe(update_progress)

                        sep = MVSepMDX23(
                            device=target_device,
                            output_dir=DATA_DIR / "mvsep_mdx23" / "out",
                            work_dir=DATA_DIR / "mvsep_mdx23" / "work",
                            repo_dir=DATA_DIR / "mvsep_mdx23" / "repo",
                            progress_callback=report_mvsep_progress,
                        )
                        task_manager.set_cancel_callback(task_id, sep.cancel)
                        try:
                            return sep.separate(audio)
                        finally:
                            task_manager.set_cancel_callback(task_id, None)
                            sep.close()
                    else:
                        raise ValueError(f"Unknown model type: {curr_type}")

                t0 = time.time()
                sep_audio = await loop.run_in_executor(None, do_sep)
                elapsed = time.time() - t0
                item_power_w = _get_device_power_w(target_device)

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
                        "device": target_device,
                        "power_w": item_power_w,
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
                    "device": target_device,
                    "power_w": item_power_w,
                })
            except Exception as e:
                task = task_manager.get_task(task_id)
                if task and task["status"] == "cancelled":
                    break
                logger.exception("Failed model separation: %s", m_label)
                results.append({
                    "model_id": f"{m_type}_{m_name or 'default'}",
                    "label": m_label,
                    "error": str(e),
                })

        task = task_manager.get_task(task_id)
        if task and task["status"] == "cancelled":
            return

        last_pwr = _get_device_power_w(target_device)
        complete_msg = (
            f"Completed {len(results)} model separations on {target_device} (⚡ {last_pwr}W) for '{audio.title}'!"
            if last_pwr is not None
            else f"Completed {len(results)} model separations on {target_device} for '{audio.title}'!"
        )
        task_manager.update_task(
            task_id,
            status="completed",
            progress=1.0,
            progress_known=True,
            message=complete_msg,
            result={
                "clip_id": audio_id,
                "clip_title": audio.title,
                "clip_path": str(audio.path),
                "results": results,
                "device": target_device,
                "power_w": last_pwr,
            },
        )

    task_manager.enqueue(task_id, run_batch)
    return web.json_response({"task_id": task_id, "task": task_manager.get_task(task_id)}, status=202)


# ==================== LIVE RELOAD SSE ====================


@web.middleware
async def no_cache_middleware(request: web.Request, handler):
    """Ensure static files and HTML are never served from browser stale cache in development."""
    response = await handler(request)
    if request.path.startswith("/static/") or request.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


async def start_background_tasks(app: web.Application):
    restored = registry.restore()
    if restored:
        logger.info("Restored %d session audio item(s) from disk", restored)
    await task_manager.start()


async def cleanup_background_tasks(app: web.Application):
    await task_manager.stop()


# ==================== STATIC FILE HANDLERS ====================


async def handle_index(request: web.Request) -> web.Response:
    """Serve the SonicStudio frontend index.html."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return web.Response(text="index.html not found", status=404)
    return web.Response(
        text=index_file.read_text(encoding="utf-8"),
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
    app.router.add_get("/api/audio", handle_list_audios)
    app.router.add_get("/api/audio/{id}", handle_get_audio_metadata)
    app.router.add_delete("/api/audio/{id}", handle_delete_audio)
    app.router.add_get("/api/audio/{id}/stream", handle_stream_audio)
    app.router.add_get("/api/audio/{id}/waveform", handle_get_waveform)
    app.router.add_get("/api/audio/{id}/spectrogram", handle_get_spectrogram)
    app.router.add_get("/api/audio/{id}/segment", handle_download_audio_segment)
    app.router.add_post("/api/audio/{id}/segments.zip", handle_download_audio_segments_zip)
    app.router.add_post("/api/audio/{id}/cut", handle_cut_audio)
    app.router.add_post("/api/audio/{id}/quick-save", handle_quick_save)
    app.router.add_post("/api/audio/{id}/save-to", handle_save_to)
    
    app.router.add_post("/api/separation/run", handle_run_separation)
    app.router.add_post("/api/separation/batch-compare", handle_batch_separation_compare)
    app.router.add_get("/api/evaluations", handle_list_evaluations)
    app.router.add_post("/api/evaluations", handle_save_evaluation)
    app.router.add_delete("/api/evaluations/{id}", handle_delete_evaluation)
    app.router.add_post("/api/diarization/run", handle_run_diarization)
    app.router.add_get(
        "/api/diarization/annotations", handle_list_diarization_annotations
    )
    app.router.add_post(
        "/api/diarization/annotations", handle_save_diarization_annotation
    )
    app.router.add_get(
        "/api/diarization/annotations/{annotation_id}",
        handle_get_diarization_annotation,
    )
    app.router.add_delete(
        "/api/diarization/annotations/{annotation_id}",
        handle_delete_diarization_annotation,
    )
    app.router.add_post(
        "/api/diarization/evaluate", handle_evaluate_diarization_results
    )
    app.router.add_post(
        "/api/diarization/clean-turns", handle_clean_diarization_turns
    )
    app.router.add_get("/api/diarization/results", handle_list_diarization_results)
    app.router.add_post(
        "/api/diarization/results/verify", handle_verify_diarization_batch
    )
    app.router.add_post(
        "/api/diarization/results/clear", handle_clear_diarization_results
    )
    app.router.add_get(
        "/api/diarization/results/{result_id}", handle_get_diarization_result
    )
    app.router.add_delete(
        "/api/diarization/results/{result_id}", handle_delete_diarization_result
    )
    app.router.add_get(
        "/api/diarization/results/{result_id}/turns/{turn_index}/audio",
        handle_preview_diarization_turn,
    )
    app.router.add_post("/api/diarization/extract-speaker", handle_extract_speaker_audio)
    app.router.add_post("/api/diarization/extract-all-speakers", handle_extract_all_speakers)
    app.router.add_get("/api/speaker-profiles", handle_list_speaker_profiles)
    app.router.add_post("/api/speaker-profiles", handle_create_speaker_profile)
    app.router.add_get("/api/speaker-profiles/{name}", handle_get_speaker_profile)
    app.router.add_post(
        "/api/speaker-profiles/{name}/clips",
        handle_add_speaker_profile_clips,
    )
    app.router.add_get(
        "/api/speaker-profiles/{name}/clips/{clip_name}",
        handle_stream_speaker_profile_clip,
    )
    app.router.add_delete(
        "/api/speaker-profiles/{name}/clips/{clip_name}",
        handle_delete_speaker_profile_clip,
    )
    app.router.add_delete("/api/speaker-profiles/{name}", handle_delete_speaker_profile)
    app.router.add_post("/api/diarization/target-speaker-score", handle_target_speaker_score)
    app.router.add_get("/api/purity/config", handle_speaker_purity_config)
    app.router.add_get("/api/purity/verifier-status", handle_purity_verifier_status)
    app.router.add_post("/api/purity/verify", handle_verify_speaker_purity)
    app.router.add_post("/api/purity/export-audio", handle_export_purity_audio)
    app.router.add_post("/api/compare/spectrogram", handle_compare_spectrogram)
    app.router.add_get("/api/tasks", handle_list_tasks)
    app.router.add_post("/api/tasks/clear", handle_clear_tasks)
    app.router.add_get("/api/tasks/{id}", handle_get_task)
    app.router.add_delete("/api/tasks/{id}", handle_cancel_task)
    app.router.add_get("/api/queue/shared", handle_shared_queue)
    app.router.add_delete("/api/queue/shared/{id}", handle_shared_queue_cancel)
    app.router.add_get("/api/telemetry", handle_telemetry)

    from src.web_studio.experiment_handler import register_experiment_routes
    register_experiment_routes(app, task_manager, registry)


def register_lifecycle(app: web.Application) -> None:
    """Register SonicStudio background services on an application.

    Args:
        app: Aiohttp application that owns the shared backend.
    """
    app.on_startup.append(start_background_tasks)
    app.on_shutdown.append(cleanup_background_tasks)


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
