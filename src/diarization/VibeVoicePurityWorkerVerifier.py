"""Main-environment proxy for isolated VibeVoice-ASR purity inference."""

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
from src.diarization.schemas import VibeVoicePurityResult
from src.diarization.VibeVoicePurityVerifier import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_SECONDARY_SPEECH_S,
    DEFAULT_VIBEVOICE_BATCH_SIZE,
    DEFAULT_VIBEVOICE_MODEL_ID,
    VibeVoicePurityError,
    vibevoice_model_is_quantized,
)
from src.utils.AudioClass import Audio

logger = logging.getLogger(__name__)

_PROTOCOL_PREFIX = "@@VIBEVOICE_PURITY_RPC@@"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class VibeVoicePurityWorkerVerifier(ManagedModel):
    """Run VibeVoice-ASR purity verification in a persistent isolated process.

    The public ``verify()`` / ``verify_batch()`` contract matches
    :class:`VibeVoicePurityVerifier`. ``load()`` starts
    ``.venv-vibevoice/bin/python -m src.diarization.vibevoice_purity_worker``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VIBEVOICE_MODEL_ID,
        *,
        device: str = "auto",
        token: str | None = None,
        min_secondary_speech_s: float = DEFAULT_MIN_SECONDARY_SPEECH_S,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        batch_size: int = DEFAULT_VIBEVOICE_BATCH_SIZE,
        attn_implementation: str = "eager",
        worker_python: str | Path | None = None,
    ) -> None:
        """Initialize the isolated VibeVoice proxy.

        Args mirror :class:`VibeVoicePurityVerifier`; ``worker_python``
        overrides the default ``.venv-vibevoice/bin/python``.
        """
        ManagedModel.__init__(self)
        configured_python = worker_python or os.getenv("VIBEVOICE_PYTHON")
        self.worker_python = Path(
            configured_python
            if configured_python is not None
            else _REPO_ROOT / ".venv-vibevoice" / "bin" / "python"
        ).expanduser()
        self._config: dict[str, Any] = {
            "model_id": model_id,
            "device": device,
            "token": token,
            "min_secondary_speech_s": min_secondary_speech_s,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "attn_implementation": attn_implementation,
        }
        self._process: subprocess.Popen[str] | None = None
        self._request_lock = threading.Lock()
        self._output_tail: deque[str] = deque(maxlen=100)
        self._cancel_requested = False

    def check_ready(self) -> dict[str, Any]:
        """Check the isolated worker environment without loading model weights.

        Returns:
            Readiness details for the configured interpreter and compute device.
        """
        worker_python = (
            self.worker_python
            if self.worker_python.is_absolute()
            else _REPO_ROOT / self.worker_python
        ).expanduser()
        if not worker_python.is_file():
            return {
                "ready": False,
                "message": (
                    f"VibeVoice worker Python does not exist: {worker_python}. "
                    "Create .venv-vibevoice from requirements-vibevoice.txt or "
                    "set VIBEVOICE_PYTHON."
                ),
                "models": [],
            }

        worker_environment = os.environ.copy()
        worker_environment["VIRTUAL_ENV"] = str(worker_python.parent.parent)
        worker_environment["PATH"] = os.pathsep.join(
            [str(worker_python.parent), worker_environment.get("PATH", "")]
        )
        if os.path.isdir("/opt/rocm/bin") and "/opt/rocm/bin" not in worker_environment["PATH"]:
            worker_environment["PATH"] = f"/opt/rocm/bin:{worker_environment['PATH']}"
        worker_environment["PYTHONNOUSERSITE"] = "1"
        worker_environment["MPLBACKEND"] = "Agg"
        worker_environment.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
        requested_device = str(self._config.get("device", "auto"))
        if requested_device.startswith("cuda:"):
            gpu_idx = requested_device.split(":", 1)[1]
            worker_environment["CUDA_VISIBLE_DEVICES"] = gpu_idx
            worker_environment["HIP_VISIBLE_DEVICES"] = gpu_idx
            worker_environment["ROCR_VISIBLE_DEVICES"] = gpu_idx
        elif requested_device == "cpu":
            worker_environment["CUDA_VISIBLE_DEVICES"] = ""
            worker_environment["HIP_VISIBLE_DEVICES"] = ""
            worker_environment["ROCR_VISIBLE_DEVICES"] = ""
        probe_code = """
import json, torch, transformers
from transformers import VibeVoiceAsrForConditionalGeneration
bnb_ok = True
bnb_error = ""
try:
    import bitsandbytes
except Exception as exc:
    bnb_ok = False
    bnb_error = f"{type(exc).__name__}: {exc}"
print(json.dumps({
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "bitsandbytes": bnb_ok,
    "bitsandbytes_error": bnb_error,
}))
"""
        try:
            completed = subprocess.run(
                [str(worker_python), "-c", probe_code],
                cwd=str(_REPO_ROOT),
                env=worker_environment,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "ready": False,
                "message": f"Could not probe the VibeVoice worker: {exc}",
                "models": [],
            }
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            return {
                "ready": False,
                "message": f"VibeVoice worker probe failed: {detail[-1000:]}",
                "models": [],
            }
        try:
            details = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            details = {}
        model_id = str(self._config["model_id"])
        quantized = vibevoice_model_is_quantized(model_id)
        if quantized and requested_device in {"cpu", "mps"}:
            return {
                "ready": False,
                "message": (
                    f"Quantized VibeVoice checkpoint {model_id} requires CUDA "
                    "(bitsandbytes)."
                ),
                "models": [],
            }
        if requested_device.startswith("cuda") and not details.get("cuda_available"):
            return {
                "ready": False,
                "message": (
                    "VibeVoice worker cannot access requested device "
                    f"{requested_device}."
                ),
                "models": [],
            }
        if quantized and requested_device == "auto" and not details.get(
            "cuda_available"
        ):
            return {
                "ready": False,
                "message": (
                    f"Quantized VibeVoice checkpoint {model_id} requires CUDA, "
                    "and the worker has no GPU."
                ),
                "models": [],
            }
        if quantized and not details.get("bitsandbytes"):
            detail = str(details.get("bitsandbytes_error") or "import failed")
            return {
                "ready": False,
                "message": (
                    f"Quantized VibeVoice checkpoint {model_id} needs "
                    "bitsandbytes>=0.48.1 in .venv-vibevoice "
                    f"({detail[-400:]})."
                ),
                "models": [],
            }
        version = details.get("transformers", "unknown")
        return {
            "ready": True,
            "message": (
                f"Local VibeVoice worker is ready on {requested_device} "
                f"(Transformers {version}); model weights load when verification starts."
            ),
            "models": [model_id],
        }

    def _load(self) -> None:
        """Start the isolated worker and load VibeVoice-ASR once."""
        worker_python = (
            self.worker_python
            if self.worker_python.is_absolute()
            else _REPO_ROOT / self.worker_python
        ).expanduser()
        if not worker_python.is_file():
            raise VibeVoicePurityError(
                f"VibeVoice worker Python does not exist: {worker_python}. "
                "Create .venv-vibevoice from requirements-vibevoice.txt or set "
                "VIBEVOICE_PYTHON."
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
        worker_environment["MPLBACKEND"] = "Agg"
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
        logger.info(
            "Starting VibeVoice worker requested_device=%s worker_device=%s "
            "CUDA_VISIBLE_DEVICES=%s attention=%s",
            requested_device,
            worker_config["device"],
            worker_environment.get("CUDA_VISIBLE_DEVICES", "all"),
            worker_config["attn_implementation"],
        )
        process = subprocess.Popen(
            [
                str(worker_python),
                "-u",
                "-m",
                "src.diarization.vibevoice_purity_worker",
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
        """Unload the worker model and reap the isolated process."""
        process = self._process
        if process is None:
            return
        if process.poll() is None and not self._cancel_requested:
            try:
                self._request({"action": "close"})
            except Exception:
                logger.warning(
                    "VibeVoice purity worker did not close cleanly",
                    exc_info=True,
                )
        self._stop_process(process)
        self._process = None

    def verify(self, audio: Audio) -> VibeVoicePurityResult:
        """Verify ``audio`` through the persistent isolated worker."""
        if not self.is_loaded or self._process is None:
            raise RuntimeError(
                "VibeVoice purity worker is not loaded. Call load() first "
                "or use it as a context manager."
            )
        if not Path(audio.path).is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio.path}")
        payload = self._request({"action": "verify", "audio": self._audio_payload(audio)})
        return VibeVoicePurityResult.from_dict(payload)

    def verify_batch(self, audios: list[Audio]) -> list[VibeVoicePurityResult]:
        """Verify candidates using the worker's configured model batch size."""
        if not self.is_loaded or self._process is None:
            raise RuntimeError(
                "VibeVoice purity worker is not loaded. Call load() first "
                "or use it as a context manager."
            )
        for audio in audios:
            if not Path(audio.path).is_file():
                raise FileNotFoundError(f"Audio file does not exist: {audio.path}")
        payload = self._request(
            {
                "action": "verify_batch",
                "audios": [self._audio_payload(audio) for audio in audios],
            }
        )
        return [VibeVoicePurityResult.from_dict(item) for item in payload]

    @staticmethod
    def _audio_payload(audio: Audio) -> dict[str, Any]:
        return {
            "path": str(Path(audio.path).resolve()),
            "source_id": audio.source_id,
            "title": audio.title,
            "source_url": audio.source_url,
            "channel_id": audio.channel_id,
            "channel_name": audio.channel_name,
            "channel_url": audio.channel_url,
            "sample_rate": audio.sample_rate,
            "duration_s": audio.duration_s,
            "channels": audio.channels,
            "format": audio.format,
            "native_sample_rate": audio.native_sample_rate,
            "history": list(audio.history),
        }

    def close(self) -> None:
        """Compatibility alias that unloads and reaps the worker."""
        self.unload()

    def _request(self, payload: dict[str, Any]) -> Any:
        with self._request_lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError(self._worker_exit_message(process))
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("VibeVoice purity worker pipes are unavailable")
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
                    logger.info("VibeVoice purity worker: %s", message)
                    continue
                preceding_output = message[:protocol_index].strip()
                if preceding_output:
                    self._output_tail.append(preceding_output)
                    logger.info("VibeVoice purity worker: %s", preceding_output)
                response = json.loads(
                    message[protocol_index + len(_PROTOCOL_PREFIX) :]
                )
                if response.get("ok"):
                    return response.get("result")
                detail = response.get("error") or "unknown worker error"
                traceback_text = response.get("traceback") or ""
                raise RuntimeError(
                    f"VibeVoice purity worker failed: {detail}\n"
                    f"{traceback_text[-3000:]}"
                )
            raise RuntimeError(self._worker_exit_message(process))

    def _worker_exit_message(
        self,
        process: subprocess.Popen[str] | None,
    ) -> str:
        returncode = process.poll() if process is not None else None
        detail = "\n".join(self._output_tail).strip()
        if self._cancel_requested:
            return "VibeVoice purity worker was cancelled"
        return (
            f"VibeVoice purity worker exited unexpectedly (exit {returncode}): "
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
