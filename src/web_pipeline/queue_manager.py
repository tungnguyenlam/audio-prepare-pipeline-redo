"""Asynchronous Job Queue Manager and Concurrency Orchestrator."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
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
    """Manages scheduling, concurrent execution, state broadcast, and job persistence."""

    def __init__(self, max_concurrency: int = 2) -> None:
        self.max_concurrency = max_concurrency
        self._jobs: Dict[str, PipelineJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._running_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._cancel_flags: Set[str] = set()
        self._is_paused = False
        self._handlers: Dict[str, Callable[..., Coroutine[Any, Any, None]]] = {}
        self._subscribers: Set[asyncio.Queue[Dict[str, Any]]] = set()
        self._worker_task: Optional[asyncio.Task[Any]] = None
        self._load_persisted_jobs()

    def _load_persisted_jobs(self) -> None:
        """Load job history from disk."""
        if not JOBS_DIR.exists():
            return
        for file_path in JOBS_DIR.glob("job_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    job = PipelineJob.from_dict(data)
                    # If server restarted during running state, mark as failed/interrupted
                    if job.status == "running":
                        job.status = "failed"
                        job.error = "Interrupted by server restart"
                    self._jobs[job.id] = job
            except Exception as e:
                logger.warning(f"Failed to load job {file_path}: {e}")

    def register_handler(self, job_type: str, handler: Callable[..., Coroutine[Any, Any, None]]) -> None:
        """Register a coroutine handler for a specific job type."""
        self._handlers[job_type] = handler

    async def start(self) -> None:
        """Start queue background worker loop."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("JobQueueManager worker loop started")

    async def stop(self) -> None:
        """Stop worker loop and cancel in-flight jobs gracefully."""
        if self._worker_task:
            self._worker_task.cancel()
        for task in self._running_tasks.values():
            task.cancel()

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

    def submit_job(
        self,
        job_type: str,
        title: str,
        params: Dict[str, Any],
        total_items: int = 1,
    ) -> PipelineJob:
        """Submit a new job to the processing queue."""
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job = PipelineJob(
            id=job_id,
            type=job_type,
            title=title,
            status="pending",
            created_at=time.time(),
            total_items=total_items,
            params=params,
        )
        job.add_log(f"Job enqueued: {title} ({job_type})", "info")
        self._jobs[job_id] = job
        self._save_job(job)
        self._queue.put_nowait(job_id)
        self.broadcast("job_created", job.to_dict())
        return job

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of jobs sorted by created_at descending."""
        jobs = list(self._jobs.values())
        if status_filter and status_filter != "all":
            jobs = [j for j in jobs if j.status == status_filter]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

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
        self.broadcast("queue_state", {"is_paused": True, "max_concurrency": self.max_concurrency})

    def resume_queue(self) -> None:
        self._is_paused = False
        self.broadcast("queue_state", {"is_paused": False, "max_concurrency": self.max_concurrency})

    def set_concurrency(self, concurrency: int) -> None:
        self.max_concurrency = max(1, min(8, concurrency))
        self.broadcast("queue_state", {"is_paused": self._is_paused, "max_concurrency": self.max_concurrency})

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancel_flags

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
        elif job.total_items > 0:
            job.progress = min(100.0, max(0.0, round(((job.processed_items + job.failed_items) / job.total_items) * 100, 1)))

        if log_message:
            job.add_log(log_message, log_level)

        self._save_job(job)
        self.broadcast("job_progress", {
            "id": job.id,
            "progress": job.progress,
            "current_step": job.current_step,
            "processed_items": job.processed_items,
            "failed_items": job.failed_items,
            "total_items": job.total_items,
            "status": job.status,
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

    async def _worker_loop(self) -> None:
        """Master dispatch loop allocating workers up to max_concurrency."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        while True:
            try:
                while self._is_paused:
                    await asyncio.sleep(0.5)

                job_id = await self._queue.get()
                job = self._jobs.get(job_id)
                if not job or job.status != "pending":
                    self._queue.task_done()
                    continue

                if self.is_cancelled(job_id):
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    self._save_job(job)
                    self._queue.task_done()
                    continue

                # Adjust semaphore if concurrency setting changed
                await semaphore.acquire()

                task = asyncio.create_task(self._run_job_wrapper(job, semaphore))
                self._running_tasks[job_id] = task

                def _cleanup(t: asyncio.Task[Any], jid: str = job_id) -> None:
                    self._running_tasks.pop(jid, None)
                    self._cancel_flags.discard(jid)
                    self._queue.task_done()

                task.add_done_callback(_cleanup)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue worker loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _run_job_wrapper(self, job: PipelineJob, semaphore: asyncio.Semaphore) -> None:
        """Execute single job handler with exception handling and VRAM cleanup."""
        try:
            handler = self._handlers.get(job.type)
            if not handler:
                raise ValueError(f"No handler registered for job type '{job.type}'")

            job.status = "running"
            job.started_at = time.time()
            job.add_log(f"Started job execution with handler '{job.type}'", "info")
            self._save_job(job)
            self.broadcast("job_updated", job.to_dict())

            await handler(job, self)

            if not self.is_cancelled(job.id):
                job.status = "completed"
                job.progress = 100.0
                job.finished_at = time.time()
                job.current_step = f"Completed successfully ({job.processed_items}/{job.total_items} items)"
                job.add_log(f"Job completed successfully in {round(job.finished_at - job.started_at, 2)}s", "info")

        except asyncio.CancelledError:
            job.status = "cancelled"
            job.finished_at = time.time()
            job.current_step = "Cancelled by user"
            job.add_log("Job execution cancelled", "warning")
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}", exc_info=True)
            job.status = "failed"
            job.finished_at = time.time()
            job.error = str(e)
            job.current_step = f"Failed: {e}"
            job.add_log(f"Job failed with error: {e}", "error")
        finally:
            self._save_job(job)
            self.broadcast("job_updated", job.to_dict())
            # Release worker slot
            semaphore.release()
            # Clean VRAM and RAM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Singleton queue manager
queue_manager = JobQueueManager(max_concurrency=2)
