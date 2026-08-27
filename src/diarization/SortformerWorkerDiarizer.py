"""Main-environment proxy for the isolated NeMo Sortformer worker."""

from __future__ import annotations

from collections import deque
import json
import logging
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

from src.base.model import ManagedModel
from src.diarization.BaseDiarizer import BaseDiarizer
from src.diarization.SortformerDiarizer import (
    DEFAULT_MODEL_FILENAME,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_OFFSET,
    DEFAULT_ONSET,
    DEFAULT_PAD_OFFSET_S,
    DEFAULT_PAD_ONSET_S,
)
from src.diarization.schemas import DiarizationResult
from src.utils.AudioClass import Audio

logger = logging.getLogger(__name__)

_PROTOCOL_PREFIX = "@@SORTFORMER_RPC@@"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class SortformerWorkerDiarizer(BaseDiarizer, ManagedModel):
    """Run Sortformer in a persistent process from its isolated environment.

    The caller remains in the primary application environment. ``load()``
    starts one worker and loads NeMo once; repeated ``diarize()`` calls reuse
    that model until ``close()`` or ``unload()`` is called.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        *,
        model_filename: str = DEFAULT_MODEL_FILENAME,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        token: str | None = None,
        batch_size: int = 1,
        window_duration_s: float = 360.0,
        overlap_duration_s: float = 60.0,
        oom_retry_window_s: float | None = 180.0,
        embedding_model_id: str = "titanet_large",
        enable_speaker_similarity: bool = True,
        embedding_similarity_threshold: float = 0.70,
        overlap_match_threshold: float = 0.35,
        onset: float = DEFAULT_ONSET,
        offset: float = DEFAULT_OFFSET,
        pad_onset_s: float = DEFAULT_PAD_ONSET_S,
        pad_offset_s: float = DEFAULT_PAD_OFFSET_S,
        min_duration_on_s: float = 0.10,
        min_duration_off_s: float = 0.15,
        ffmpeg_bin: str = "ffmpeg",
        worker_python: str | Path | None = None,
    ) -> None:
        ManagedModel.__init__(self)
        configured_python = worker_python or os.getenv("SORTFORMER_PYTHON")
        self.worker_python = Path(
            configured_python
            if configured_python is not None
            else _REPO_ROOT / ".venv-sortformer" / "bin" / "python"
        ).expanduser()
        self._config: dict[str, Any] = {
            "model_id": model_id,
            "revision": revision,
            "model_filename": model_filename,
            "checkpoint_path": (
                str(Path(checkpoint_path).expanduser())
                if checkpoint_path is not None
                else None
            ),
            "device": device,
            "token": token,
            "batch_size": batch_size,
            "window_duration_s": window_duration_s,
            "overlap_duration_s": overlap_duration_s,
            "oom_retry_window_s": oom_retry_window_s,
            "embedding_model_id": embedding_model_id,
            "enable_speaker_similarity": enable_speaker_similarity,
            "embedding_similarity_threshold": embedding_similarity_threshold,
            "overlap_match_threshold": overlap_match_threshold,
            "onset": onset,
            "offset": offset,
            "pad_onset_s": pad_onset_s,
            "pad_offset_s": pad_offset_s,
            "min_duration_on_s": min_duration_on_s,
            "min_duration_off_s": min_duration_off_s,
            "ffmpeg_bin": ffmpeg_bin,
        }
        self._process: subprocess.Popen[str] | None = None
        self._request_lock = threading.Lock()
        self._output_tail: deque[str] = deque(maxlen=100)
        self._cancel_requested = False

    def _load(self) -> None:
        """Start the isolated worker and load Sortformer once."""
        worker_python = (
            self.worker_python
            if self.worker_python.is_absolute()
            else (_REPO_ROOT / self.worker_python)
        ).expanduser()
        if not worker_python.is_file():
            raise RuntimeError(
                f"Sortformer worker Python does not exist: {worker_python}. "
                "Create .venv-sortformer from requirements-sortformer.txt or "
                "set SORTFORMER_PYTHON."
            )
        self._cancel_requested = False
        worker_environment = os.environ.copy()
        worker_environment["VIRTUAL_ENV"] = str(worker_python.parent.parent)
        worker_environment["PATH"] = os.pathsep.join(
            [
                str(worker_python.parent),
                worker_environment.get("PATH", ""),
            ]
        )
        worker_environment["PYTHONNOUSERSITE"] = "1"
        worker_environment.setdefault(
            "HF_HOME",
            os.environ.get("HF_HOME") or str(_REPO_ROOT / ".data" / "huggingface"),
        )
        process = subprocess.Popen(
            [
                str(worker_python),
                "-u",
                "-m",
                "src.diarization.sortformer_worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_REPO_ROOT),
            env=worker_environment,
            start_new_session=True,
        )
        self._process = process
        try:
            self._request({"action": "load", "config": self._config})
        except Exception:
            self._stop_process(process)
            self._process = None
            raise

    def _unload(self) -> None:
        """Unload the worker model and reap the isolated process."""
        process = self._process
        if process is None:
            return
        if process.poll() is None and not self._cancel_requested:
            try:
                self._request({"action": "close"})
            except Exception:
                logger.warning("Sortformer worker did not close cleanly", exc_info=True)
        self._stop_process(process)
        self._process = None

    def diarize(
        self,
        audio: Audio,
        *,
        enrollment_name: str | None = None,
        enrollment_clips: list[str | Path] | None = None,
    ) -> DiarizationResult:
        """Diarize audio with optional pipeline-owned speaker enrollment."""
        if not self.is_loaded or self._process is None:
            raise RuntimeError(
                "Sortformer worker is not loaded. Call load() before diarize(), "
                "or use it as a context manager."
            )
        if not Path(audio.path).is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio.path}")
        request: dict[str, Any] = {
            "action": "diarize",
            "audio": audio.metadata(),
        }
        if enrollment_name or enrollment_clips:
            request["enrollment_name"] = enrollment_name
            request["enrollment_clips"] = [
                str(Path(path)) for path in (enrollment_clips or [])
            ]
        payload = self._request(request)
        return self._result_from_dict(payload)

    def cancel(self) -> None:
        """Request non-blocking cancellation of active worker inference."""
        self._cancel_requested = True
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()

    def close(self) -> None:
        """Compatibility alias that unloads and reaps the worker."""
        self.unload()

    def _request(self, payload: dict[str, Any]) -> Any:
        with self._request_lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError(self._worker_exit_message(process))
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("Sortformer worker pipes are unavailable")
            try:
                process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                process.stdin.flush()
            except BrokenPipeError as exc:
                raise RuntimeError(self._worker_exit_message(process)) from exc

            for line in process.stdout:
                message = line.rstrip()
                if not message:
                    continue
                protocol_index = message.find(_PROTOCOL_PREFIX)
                if protocol_index < 0:
                    self._output_tail.append(message)
                    logger.info("Sortformer worker: %s", message)
                    continue
                preceding_output = message[:protocol_index].strip()
                if preceding_output:
                    self._output_tail.append(preceding_output)
                    logger.info("Sortformer worker: %s", preceding_output)
                response = json.loads(
                    message[protocol_index + len(_PROTOCOL_PREFIX) :]
                )
                if response.get("ok"):
                    return response.get("result")
                detail = response.get("error") or "unknown worker error"
                traceback_text = response.get("traceback") or ""
                raise RuntimeError(
                    f"Sortformer worker failed: {detail}\n{traceback_text[-3000:]}"
                )
            raise RuntimeError(self._worker_exit_message(process))

    def _worker_exit_message(
        self,
        process: subprocess.Popen[str] | None,
    ) -> str:
        returncode = process.poll() if process is not None else None
        detail = "\n".join(self._output_tail).strip()
        if self._cancel_requested:
            return "Sortformer worker was cancelled"
        return (
            f"Sortformer worker exited unexpectedly (exit {returncode}): "
            f"{detail[-3000:] or 'no output'}"
        )

    @staticmethod
    def _stop_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            except OSError:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    return
                except OSError:
                    process.kill()
                process.wait(timeout=5)
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if process.stdout is not None:
            process.stdout.close()

    @staticmethod
    def _result_from_dict(payload: dict[str, Any]) -> DiarizationResult:
        return DiarizationResult.from_dict(payload)
