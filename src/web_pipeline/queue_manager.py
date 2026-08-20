"""Asynchronous Job Queue Manager and Concurrency Orchestrator."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import torch

logger = logging.getLogger("queue_manager")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / ".data" / "pipeline"
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_queue_device(device: str | None) -> str:
    """Map a requested compute device to a dedicated queue lane key."""
    raw = (device or "").strip().lower()
    if not raw or raw in {"auto", "cuda"}:
        if torch.cuda.is_available():
            best_index = max(
                range(torch.cuda.device_count()),
                key=lambda index: torch.cuda.get_device_properties(index).total_memory,
            )
            return f"cuda:{best_index}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if raw.startswith("cuda:"):
        if torch.cuda.is_available():
            return raw
        return "cpu"
    if raw in {"cpu", "mps"}:
        return raw
    return "cpu"


def discover_queue_devices() -> List[str]:
    """Return the default set of queue lanes for this host."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(f"cuda:{index}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


_PROGRESS_PERCENT_RE = re.compile(r"(?:progress:\s*)?(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE)
_PROGRESS_FRACTION_RE = re.compile(r"(?:^|[^\d])(\d+)\s*/\s*(\d+)(?:\s|$)")


def parse_progress_text(message: str) -> Optional[float]:
    """Return 0-100 if ``message`` contains a detectable progress value.

    Args:
        message: A backend log line, tqdm bar, or ``PROGRESS: N%`` marker.

    Returns:
        A percentage in ``[0, 100]``, or ``None`` when no numeric progress is
        present.
    """
    if not message:
        return None
    match = _PROGRESS_PERCENT_RE.search(message)
    if match:
        value = float(match.group(1))
        if 0.0 <= value <= 100.0:
            return value
    match = _PROGRESS_FRACTION_RE.search(message)
    if match:
        done = float(match.group(1))
        total = float(match.group(2))
        if total > 0:
            return min(100.0, round(100.0 * done / total, 1))
    return None


@dataclass
class JobLog:
    timestamp: float
    level: str
    message: str


@dataclass
class PipelineJob:
    """A batch processing job executed asynchronously."""

    id: str
    type: str
    title: str
    status: str = "pending"  # pending, running, completed, failed, cancelled, paused
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0
    progress_known: bool = False
    current_step: str = "Initialized"
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    item_results: List[Dict[str, Any]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineJob:
        return cls(
            id=data["id"],
            type=data["type"],
            title=data.get("title", "Job"),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            progress=float(data.get("progress", 0.0)),
            progress_known=bool(data.get("progress_known", False)),
            current_step=data.get("current_step", "Initialized"),
            total_items=int(data.get("total_items", 0)),
            processed_items=int(data.get("processed_items", 0)),
            failed_items=int(data.get("failed_items", 0)),
            item_results=list(data.get("item_results", [])),
            params=dict(data.get("params", {})),
            logs=list(data.get("logs", [])),
            error=data.get("error"),
        )

    def add_log(self, message: str, level: str = "info") -> None:
        """Add timestamped log entry."""
        entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
        }
        self.logs.append(entry)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]


class JobQueueManager:
    """Per-device batch job queues with independent dispatch slots.

    Jobs are routed by ``params.device`` onto a dedicated FIFO lane so GPU 0
    and GPU 1 never share a worker slot. ``workers_per_device`` controls how
    many jobs may run concurrently on the same accelerator (default 1).
    """

    def __init__(self, workers_per_device: int = 1) -> None:
        self.workers_per_device = max(1, min(8, workers_per_device))
        self._jobs: Dict[str, PipelineJob] = {}
        self._device_queues: Dict[str, asyncio.Queue[str]] = {}
        self._device_running: Dict[str, Set[str]] = {}
        self._running_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._cancel_flags: Set[str] = set()
        self._is_paused = False
        self._handlers: Dict[str, Callable[..., Coroutine[Any, Any, None]]] = {}
        self._subscribers: Set[asyncio.Queue[Dict[str, Any]]] = set()
        self._worker_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._dispatch_events: Dict[str, asyncio.Event] = {}
        self._shutting_down = False
        self._cancel_callbacks: Dict[str, Callable[[], None]] = {}
        self._started = False
        self._load_persisted_jobs()

    @property
    def max_concurrency(self) -> int:
        """Compatibility alias for workers-per-device."""
        return self.workers_per_device

    def _ensure_device_lane(self, device: str) -> None:
        if device in self._device_queues:
            return
        self._device_queues[device] = asyncio.Queue()
        self._device_running[device] = set()
        self._dispatch_events[device] = asyncio.Event()
        if self._started:
            self._worker_tasks[device] = asyncio.create_task(
                self._worker_loop(device),
                name=f"pipeline-queue-worker-{device}",
            )
        logger.info("Pipeline queue lane ready for %s", device)

    def _load_persisted_jobs(self) -> None:
        """Load job history from disk."""
        if not JOBS_DIR.exists():
            return
        for file_path in JOBS_DIR.glob("job_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    job = PipelineJob.from_dict(data)
                    if job.status in ("pending", "running"):
                        job.status = "cancelled"
                        job.finished_at = time.time()
                        job.current_step = "Cancelled by server shutdown"
                        job.error = "Interrupted by server shutdown"
                        job.add_log(
                            "Job cancelled because the server stopped",
                            "warning",
                        )
                        self._save_job(job)
                    self._jobs[job.id] = job
            except Exception as e:
                logger.warning(f"Failed to load job {file_path}: {e}")

    def register_handler(self, job_type: str, handler: Callable[..., Coroutine[Any, Any, None]]) -> None:
        """Register a coroutine handler for a specific job type."""
        self._handlers[job_type] = handler

    async def start(self) -> None:
        """Start one dispatcher loop per known device lane."""
        if self._started:
            return
        self._started = True
        for device in discover_queue_devices():
            self._ensure_device_lane(device)
            if device not in self._worker_tasks or self._worker_tasks[device].done():
                self._worker_tasks[device] = asyncio.create_task(
                    self._worker_loop(device),
                    name=f"pipeline-queue-worker-{device}",
                )
        logger.info(
            "JobQueueManager started (%d worker(s)/device, lanes=%s)",
            self.workers_per_device,
            ", ".join(sorted(self._device_queues)),
        )

    def set_job_cancel_callback(
        self,
        job_id: str,
        callback: Callable[[], None] | None,
    ) -> None:
        """Register or clear a backend cancel hook for an active job."""
        if callback is None:
            self._cancel_callbacks.pop(job_id, None)
        else:
            self._cancel_callbacks[job_id] = callback

    def _invoke_cancel_callback(self, job_id: str) -> None:
        callback = self._cancel_callbacks.pop(job_id, None)
        if callback is None:
            return
        try:
            callback()
        except Exception:
            logger.warning("Job cancel callback failed for %s", job_id, exc_info=True)

    def _cancel_unfinished_jobs(self, reason: str) -> None:
        """Mark every pending or running job cancelled and persist the result."""
        for job in list(self._jobs.values()):
            if job.status not in ("pending", "running"):
                continue
            self._cancel_flags.add(job.id)
            self._invoke_cancel_callback(job.id)
            job.status = "cancelled"
            job.finished_at = time.time()
            job.current_step = "Cancelled by server shutdown"
            job.add_log(reason, "warning")
            self._save_job(job)
            self.broadcast("job_updated", job.to_dict())
        for job_id in list(self._cancel_callbacks):
            self._invoke_cancel_callback(job_id)

    def _drain_pending_queues(self) -> None:
        """Drop queued job ids so shutdown cannot dispatch more work."""
        for queue in self._device_queues.values():
            while True:
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break

    async def stop(self) -> None:
        """Stop device workers and terminate unfinished jobs immediately."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._is_paused = False
        self._cancel_unfinished_jobs(
            "Job cancelled because the server is shutting down",
        )
        self._drain_pending_queues()

        workers = [task for task in self._worker_tasks.values() if not task.done()]
        for task in workers:
            task.cancel()
        if workers:
            await asyncio.wait(workers, timeout=1.0)
        self._worker_tasks.clear()
        self._started = False

        running_tasks = [task for task in self._running_tasks.values() if not task.done()]
        for task in running_tasks:
            task.cancel()
        if running_tasks:
            await asyncio.wait(running_tasks, timeout=2.0)

    def subscribe(self) -> asyncio.Queue[Dict[str, Any]]:
        """Subscribe to real-time job and telemetry events via SSE."""
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Dict[str, Any]]) -> None:
        """Unsubscribe from event stream."""
        self._subscribers.discard(q)

    def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast event to all connected SSE clients."""
        payload = {"event": event_type, "data": data, "timestamp": time.time()}
        dead_queues = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead_queues.append(q)
            except Exception:
                dead_queues.append(q)
        for q in dead_queues:
            self._subscribers.discard(q)

    def _job_queue_device(self, job: PipelineJob) -> str:
        params = job.params or {}
        requested = params.get("queue_device") or params.get("device")
        # Ingest / upload jobs are host-bound, not GPU-bound.
        if job.type in {"batch_ingest_yt", "batch_ingest_files", "batch_upload"}:
            lane = "cpu"
        else:
            lane = normalize_queue_device(requested if requested is not None else "auto")
        params["device"] = lane
        params["queue_device"] = lane
        job.params = params
        return lane

    def submit_job(
        self,
        job_type: str,
        title: str,
        params: Dict[str, Any],
        total_items: int = 1,
    ) -> PipelineJob:
        """Submit a new job to the processing queue for its target device."""
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job = PipelineJob(
            id=job_id,
            type=job_type,
            title=title,
            status="pending",
            created_at=time.time(),
            total_items=total_items,
            params=dict(params or {}),
        )
        lane = self._job_queue_device(job)
        job.add_log(f"Job enqueued on {lane}: {title} ({job_type})", "info")
        self._jobs[job_id] = job
        self._save_job(job)
        self._ensure_device_lane(lane)
        self._device_queues[lane].put_nowait(job_id)
        self._dispatch_events[lane].set()
        payload = job.to_dict()
        payload["queue_device"] = lane
        self.broadcast("job_created", payload)
        return job

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of jobs sorted by created_at descending."""
        jobs = list(self._jobs.values())
        if status_filter and status_filter != "all":
            jobs = [j for j in jobs if j.status == status_filter]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        result = []
        for job in jobs[:limit]:
            data = job.to_dict()
            data["queue_device"] = (job.params or {}).get("queue_device") or (job.params or {}).get("device")
            result.append(data)
        return result

    def cancel_job(self, job_id: str) -> bool:
        """Signal job cancellation."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return False

        self._cancel_flags.add(job_id)
        if job.status == "pending":
            job.status = "cancelled"
            job.finished_at = time.time()
            job.current_step = "Cancelled while pending"
            job.add_log("Job cancelled by user", "warning")
            self._save_job(job)
            self.broadcast("job_updated", job.to_dict())
            return True

        self._invoke_cancel_callback(job_id)
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()
        job.status = "cancelled"
        job.finished_at = time.time()
        job.current_step = "Cancelled"
        job.add_log("Job cancelled by user", "warning")
        self._save_job(job)
        self.broadcast("job_updated", job.to_dict())
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a finished job record."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status == "running":
            self.cancel_job(job_id)

        self._jobs.pop(job_id, None)
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            try:
                job_file.unlink()
            except Exception:
                pass
        self.broadcast("job_deleted", {"id": job_id})
        return True

    def pause_queue(self) -> None:
        self._is_paused = True
        for event in self._dispatch_events.values():
            event.set()
        self.broadcast(
            "queue_state",
            {
                "is_paused": True,
                "max_concurrency": self.workers_per_device,
                "workers_per_device": self.workers_per_device,
                "device_queues": self.status().get("device_queues", {}),
            },
        )

    def resume_queue(self) -> None:
        self._is_paused = False
        for event in self._dispatch_events.values():
            event.set()
        self.broadcast(
            "queue_state",
            {
                "is_paused": False,
                "max_concurrency": self.workers_per_device,
                "workers_per_device": self.workers_per_device,
                "device_queues": self.status().get("device_queues", {}),
            },
        )

    def set_concurrency(self, concurrency: int) -> None:
        """Set how many jobs may run at once on each device lane."""
        self.workers_per_device = max(1, min(8, concurrency))
        for event in self._dispatch_events.values():
            event.set()
        self.broadcast(
            "queue_state",
            {
                "is_paused": self._is_paused,
                "max_concurrency": self.workers_per_device,
                "workers_per_device": self.workers_per_device,
                "device_queues": self.status().get("device_queues", {}),
            },
        )

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancel_flags

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def running_jobs(self) -> List[PipelineJob]:
        return [job for job in self._jobs.values() if job.status == "running"]

    @property
    def pending_jobs(self) -> List[PipelineJob]:
        return [job for job in self._jobs.values() if job.status == "pending"]

    def status(self) -> Dict[str, Any]:
        """Return aggregate and per-device queue status."""
        device_queues: Dict[str, Dict[str, Any]] = {}
        total_running = 0
        total_queued = 0
        for device, queue in self._device_queues.items():
            running = len(self._device_running.get(device, set()))
            queued = queue.qsize()
            # qsize includes items already claimed but not yet task_done; prefer pending job count
            pending_for_device = sum(
                1
                for job in self._jobs.values()
                if job.status == "pending"
                and ((job.params or {}).get("queue_device") or (job.params or {}).get("device")) == device
            )
            queued = pending_for_device
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
            "pending": total_queued,
            "queued": total_queued,
            "is_paused": self._is_paused,
            "device_queues": device_queues,
        }

    def update_job_progress(
        self,
        job_id: str,
        processed_items: Optional[int] = None,
        failed_items: Optional[int] = None,
        current_step: Optional[str] = None,
        progress: Optional[float] = None,
        log_message: Optional[str] = None,
        log_level: str = "info",
    ) -> None:
        """Update live progress and emit state changes."""
        job = self._jobs.get(job_id)
        if not job:
            return

        if processed_items is not None:
            job.processed_items = processed_items
        if failed_items is not None:
            job.failed_items = failed_items
        if current_step is not None:
            job.current_step = current_step

        if progress is not None:
            job.progress = min(100.0, max(0.0, progress))
            job.progress_known = True
        elif processed_items is not None or failed_items is not None:
            if job.total_items > 1:
                finished = job.processed_items + job.failed_items
                job.progress = min(100.0, max(0.0, round((finished / job.total_items) * 100, 1)))
                job.progress_known = True

        if log_message:
            parsed = parse_progress_text(log_message)
            if parsed is not None and progress is None:
                if job.total_items > 1:
                    finished = job.processed_items + job.failed_items
                    job.progress = min(
                        100.0,
                        round(((finished + parsed / 100.0) / job.total_items) * 100, 1),
                    )
                else:
                    job.progress = parsed
                job.progress_known = True
            job.add_log(log_message, log_level)

        self._save_job(job)
        self.broadcast("job_progress", {
            "id": job.id,
            "progress": job.progress,
            "progress_known": job.progress_known,
            "current_step": job.current_step,
            "processed_items": job.processed_items,
            "failed_items": job.failed_items,
            "total_items": job.total_items,
            "status": job.status,
            "queue_device": (job.params or {}).get("queue_device") or (job.params or {}).get("device"),
            "last_log": job.logs[-1] if job.logs else None,
        })

    def _save_job(self, job: PipelineJob) -> None:
        """Persist single job snapshot to disk."""
        try:
            target = JOBS_DIR / f"{job.id}.json"
            with open(target, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist job {job.id}: {e}")

    async def _worker_loop(self, device: str) -> None:
        """Dispatch jobs for one device up to workers_per_device."""
        queue = self._device_queues[device]
        event = self._dispatch_events[device]
        while True:
            try:
                while True:
                    running_on_device = len(self._device_running.get(device, set()))
                    if self._is_paused or running_on_device >= self.workers_per_device:
                        event.clear()
                        running_on_device = len(self._device_running.get(device, set()))
                        if self._is_paused or running_on_device >= self.workers_per_device:
                            await event.wait()
                        continue

                    try:
                        job_id = queue.get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        event.clear()
                        if not queue.empty():
                            continue
                        await event.wait()

                job = self._jobs.get(job_id)
                if not job or job.status != "pending":
                    queue.task_done()
                    continue

                if self.is_cancelled(job_id):
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    self._save_job(job)
                    queue.task_done()
                    continue

                self._device_running[device].add(job_id)
                task = asyncio.create_task(self._run_job_wrapper(job, device))
                self._running_tasks[job_id] = task

                def _cleanup(t: asyncio.Task[Any], jid: str = job_id, lane: str = device) -> None:
                    self._running_tasks.pop(jid, None)
                    self._device_running.get(lane, set()).discard(jid)
                    self._cancel_flags.discard(jid)
                    queue.task_done()
                    self._dispatch_events[lane].set()

                task.add_done_callback(_cleanup)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue worker loop ({device}): {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _run_job_wrapper(self, job: PipelineJob, device: str) -> None:
        """Execute single job handler with exception handling and VRAM cleanup."""
        try:
            handler = self._handlers.get(job.type)
            if not handler:
                raise ValueError(f"No handler registered for job type '{job.type}'")

            job.status = "running"
            job.started_at = time.time()
            job.add_log(
                f"Started job execution on {device} with handler '{job.type}'",
                "info",
            )
            self._save_job(job)
            payload = job.to_dict()
            payload["queue_device"] = device
            self.broadcast("job_updated", payload)

            await handler(job, self)

            if not self.is_cancelled(job.id) and not self._shutting_down:
                job.status = "completed"
                job.progress = 100.0
                job.progress_known = True
                job.finished_at = time.time()
                job.current_step = f"Completed successfully ({job.processed_items}/{job.total_items} items)"
                job.add_log(f"Job completed successfully in {round(job.finished_at - job.started_at, 2)}s", "info")

        except asyncio.CancelledError:
            job.status = "cancelled"
            job.finished_at = time.time()
            if self._shutting_down:
                job.current_step = "Cancelled by server shutdown"
                job.add_log(
                    "Job execution cancelled because the server is shutting down",
                    "warning",
                )
            else:
                job.current_step = "Cancelled by user"
                job.add_log("Job execution cancelled", "warning")
        except Exception as e:
            if self.is_cancelled(job.id) or self._shutting_down:
                job.status = "cancelled"
                job.finished_at = time.time()
                if self._shutting_down:
                    job.current_step = "Cancelled by server shutdown"
                    job.add_log(
                        "Job cancelled because the server is shutting down",
                        "warning",
                    )
                else:
                    job.current_step = "Cancelled by user"
                    job.add_log("Job cancelled by user", "warning")
            else:
                logger.error(f"Job {job.id} failed: {e}", exc_info=True)
                job.status = "failed"
                job.finished_at = time.time()
                job.error = str(e)
                job.current_step = f"Failed: {e}"
                job.add_log(f"Job failed with error: {e}", "error")
        finally:
            self._cancel_callbacks.pop(job.id, None)
            if self._shutting_down or self.is_cancelled(job.id):
                job.status = "cancelled"
                if not job.finished_at:
                    job.finished_at = time.time()
            self._save_job(job)
            payload = job.to_dict()
            payload["queue_device"] = device
            self.broadcast("job_updated", payload)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Singleton queue manager
queue_manager = JobQueueManager(workers_per_device=1)
