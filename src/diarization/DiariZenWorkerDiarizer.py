"""Main-environment proxy for the isolated DiariZen worker."""

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
from src.diarization.DiariZenDiarizer import DEFAULT_DIARIZEN_MODEL_ID
from src.diarization.schemas import DiarizationResult
from src.utils.AudioClass import Audio

logger = logging.getLogger(__name__)

_PROTOCOL_PREFIX = "@@DIARIZEN_RPC@@"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class DiariZenWorkerDiarizer(BaseDiarizer, ManagedModel):
    """Run DiariZen in a persistent isolated Python process."""

    def __init__(
        self,
        model_id: str = DEFAULT_DIARIZEN_MODEL_ID,
        *,
        device: str = "auto",
        token: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        batch_size: int = 1,
        ffmpeg_bin: str = "ffmpeg",
        worker_python: str | Path | None = None,
    ) -> None:
        """Initialize the isolated DiariZen proxy.

        Args mirror :class:`DiariZenDiarizer`; ``worker_python`` overrides the
        default ``.venv-diarizen/bin/python`` interpreter.
        """
        ManagedModel.__init__(self)
        configured_python = worker_python or os.getenv("DIARIZEN_PYTHON")
        self.worker_python = Path(
            configured_python
            if configured_python is not None
            else _REPO_ROOT / ".venv-diarizen" / "bin" / "python"
        ).expanduser()
        self._config: dict[str, Any] = {
            "model_id": model_id,
            "device": device,
            "token": token,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "batch_size": batch_size,
            "ffmpeg_bin": ffmpeg_bin,
        }
        self._process: subprocess.Popen[str] | None = None
        self._request_lock = threading.Lock()
        self._output_tail: deque[str] = deque(maxlen=100)
        self._cancel_requested = False

    def _load(self) -> None:
        """Start the isolated worker and load DiariZen once."""
        worker_python = (
            self.worker_python
            if self.worker_python.is_absolute()
            else _REPO_ROOT / self.worker_python
        ).expanduser()
        if not worker_python.is_file():
            raise RuntimeError(
                f"DiariZen worker Python does not exist: {worker_python}. "
                "Create .venv-diarizen from requirements-diarizen.txt or set "
                "DIARIZEN_PYTHON."
            )

        self._cancel_requested = False
        worker_environment = os.environ.copy()
        worker_environment["VIRTUAL_ENV"] = str(worker_python.parent.parent)
        worker_environment["PATH"] = os.pathsep.join(
            [str(worker_python.parent), worker_environment.get("PATH", "")]
        )
        if os.path.isdir("/opt/rocm/bin") and "/opt/rocm/bin" not in worker_environment["PATH"]:
            worker_environment["PATH"] = f"/opt/rocm/bin:{worker_environment['PATH']}"
        worker_environment["PYTHONNOUSERSITE"] = "1"
        worker_environment.setdefault(
            "HF_HOME",
            os.environ.get("HF_HOME") or str(_REPO_ROOT / ".data" / "huggingface"),
        )
        worker_environment.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

        worker_config = dict(self._config)
        requested_device = str(worker_config.get("device", "auto"))
        if requested_device.startswith("cuda:"):
            gpu_idx = requested_device.split(":", 1)[1]
            worker_environment["CUDA_VISIBLE_DEVICES"] = gpu_idx
            worker_environment["HIP_VISIBLE_DEVICES"] = gpu_idx
            worker_environment["ROCR_VISIBLE_DEVICES"] = gpu_idx
            worker_config["device"] = "cuda:0"
        elif requested_device == "cpu":
            worker_environment["CUDA_VISIBLE_DEVICES"] = ""
            worker_environment["HIP_VISIBLE_DEVICES"] = ""
            worker_environment["ROCR_VISIBLE_DEVICES"] = ""

        process = subprocess.Popen(
            [
                str(worker_python),
                "-u",
                "-m",
                "src.diarization.diarizen_worker",
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
            self._request({"action": "load", "config": worker_config})
        except Exception:
            self._stop_process(process)
            self._process = None
            raise

    def _unload(self) -> None:
        """Unload models and reap the worker process."""
        process = self._process
        if process is None:
            return
        if process.poll() is None and not self._cancel_requested:
            try:
                self._request({"action": "close"})
            except Exception:
                logger.warning("DiariZen worker did not close cleanly", exc_info=True)
        self._stop_process(process)
        self._process = None

    def diarize(
        self,
        audio: Audio,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """Diarize ``audio`` through the persistent worker."""
        if not self.is_loaded or self._process is None:
            raise RuntimeError(
                "DiariZen worker is not loaded. Call load() before diarize(), "
                "or use it as a context manager."
            )
        if not Path(audio.path).is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio.path}")

        request: dict[str, Any] = {
            "action": "diarize",
            "audio": audio.metadata(),
        }
        for name, value in (
            ("num_speakers", num_speakers),
            ("min_speakers", min_speakers),
            ("max_speakers", max_speakers),
        ):
            if value is not None:
                request[name] = int(value)
        return self._result_from_dict(self._request(request))

    def cancel(self) -> None:
        """Request non-blocking cancellation of active inference."""
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
                raise RuntimeError("DiariZen worker pipes are unavailable")
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
                    logger.info("DiariZen worker: %s", message)
                    continue
                preceding_output = message[:protocol_index].strip()
                if preceding_output:
                    self._output_tail.append(preceding_output)
                    logger.info("DiariZen worker: %s", preceding_output)
                response = json.loads(
                    message[protocol_index + len(_PROTOCOL_PREFIX) :]
                )
                if response.get("ok"):
                    return response.get("result")
                detail = response.get("error") or "unknown worker error"
                traceback_text = response.get("traceback") or ""
                raise RuntimeError(
                    f"DiariZen worker failed: {detail}\n{traceback_text[-3000:]}"
                )
            raise RuntimeError(self._worker_exit_message(process))

    def _worker_exit_message(
        self,
        process: subprocess.Popen[str] | None,
    ) -> str:
        returncode = process.poll() if process is not None else None
        detail = "\n".join(self._output_tail).strip()
        if self._cancel_requested:
            return "DiariZen worker was cancelled"
        return (
            f"DiariZen worker exited unexpectedly (exit {returncode}): "
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
