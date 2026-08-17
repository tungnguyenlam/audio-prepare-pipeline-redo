"""MVSEP-MDX23 separation backend via ZFTurbo's ``inference.py`` CLI.

Wraps https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model for
vocals cleanup (``--only_vocals`` by default). The upstream repo is cloned
on first use; checkpoints download themselves on the first inference run.
"""

from __future__ import annotations

from collections import deque
import logging
import os
import signal
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from src.separation.audio_utils import normalize_wav, probe_wav
from src.separation.BaseSeparator import BaseSeparator
from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio

logger = logging.getLogger(__name__)

DEFAULT_REPO_URL = (
    "https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model.git"
)

_UPSTREAM_DEVICE_OVERRIDE = """if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
"""
_UPSTREAM_DEVICE_PATCH = """if __name__ == '__main__':
    import os

    # Device visibility is supplied by the MVSepMDX23 wrapper.
    print('GPU visibility: {}'.format(os.environ.get('CUDA_VISIBLE_DEVICES', 'default')))
"""

# Upstream writes ``{stem}_instrum.wav`` for the residual instrumental.
_STEM_OUTPUT_IDS = {
    "vocals": "vocals",
    "instrumental": "instrum",
    "instrum": "instrum",
    "bass": "bass",
    "drums": "drums",
    "other": "other",
}


class MVSepMDX23Error(RuntimeError):
    """Raised when MVSEP-MDX23 separation cannot run or the output is unusable."""


class MVSepMDX23(BaseSeparator):
    """MVSEP-MDX23 ensemble backend (Demucs + Kim MDX ONNX).

    Usage:
        separator = MVSepMDX23(device="cpu")
        cleaned = separator.separate(audio)

    Notes:
        - CUDA + ``onnxruntime-gpu`` is the intended fast path.
        - On Apple Silicon / machines without CUDA ONNX, ``auto`` device
          falls back to ``--cpu`` (slow, but functional). Explicit ``device="cuda"``
          raises :exc:`MVSepMDX23Error` if CUDA or ONNX CUDA provider is missing.
        - Inputs are split into bounded 10-minute segments by default to limit
          host RAM. Pass ``max_segment_seconds=None`` to disable segmentation.
        - First run clones the upstream repo and downloads multi-GB weights.
    """

    def __init__(
        self,
        model: str = "mvsep-mdx23",
        device: str = "auto",
        two_stems: str = "vocals",
        output_dir: str | Path = ".data/mvsep_mdx23/out",
        work_dir: str | Path = ".data/mvsep_mdx23/work",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        ffmpeg_bin: Optional[str] = None,
        repo_dir: str | Path | None = None,
        repo_url: str = DEFAULT_REPO_URL,
        only_vocals: bool | None = None,
        single_onnx: bool = True,
        large_gpu: bool = False,
        use_kim_model_1: bool = False,
        overlap_large: float = 0.25,
        overlap_small: float = 0.25,
        chunk_size: int | None = None,
        max_segment_seconds: float | None = 600.0,
        python_bin: Optional[str] = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            two_stems=two_stems,
            output_dir=output_dir,
            work_dir=work_dir,
            sample_rate=sample_rate,
            channels=channels,
            ffmpeg_bin=ffmpeg_bin,
        )
        self.repo_dir = Path(repo_dir) if repo_dir else Path(".data/mvsep_mdx23/repo")
        self.repo_url = repo_url
        self.single_onnx = single_onnx
        self.large_gpu = large_gpu
        self.use_kim_model_1 = use_kim_model_1
        self.overlap_large = overlap_large
        self.overlap_small = overlap_small
        self.chunk_size = chunk_size
        if max_segment_seconds is not None and max_segment_seconds <= 0:
            raise MVSepMDX23Error("max_segment_seconds must be positive or None")
        self.max_segment_seconds = max_segment_seconds
        self.python_bin = python_bin or sys.executable
        self.progress_callback = progress_callback
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._cancel_requested = False

        if only_vocals is None:
            stem_id = _STEM_OUTPUT_IDS.get(self.two_stems.lower())
            only_vocals = stem_id in ("vocals", "instrum")
        self.only_vocals = only_vocals

    def _output_stem_id(self) -> str:
        stem = self.two_stems.lower()
        if stem not in _STEM_OUTPUT_IDS:
            raise MVSepMDX23Error(
                f"unsupported stem {self.two_stems!r}; "
                f"expected one of {sorted(_STEM_OUTPUT_IDS)}"
            )
        return _STEM_OUTPUT_IDS[stem]

    def _should_use_cpu(self) -> bool:
        """Determine whether to append ``--cpu`` or run on CUDA.

        Raises:
            MVSepMDX23Error: If a CUDA device was explicitly requested but
                either PyTorch CUDA or ONNX Runtime ``CUDAExecutionProvider``
                is unavailable.
        """
        device = str(self.device).lower()
        if device in ("cpu", "mps") or device.startswith("mps"):
            return True
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    raise MVSepMDX23Error(
                        f"Device {self.device!r} requested but PyTorch CUDA is not available."
                    )
                if ":" in device:
                    try:
                        device_index = int(device.split(":", 1)[1])
                    except ValueError as exc:
                        raise MVSepMDX23Error(
                            f"Invalid CUDA device {self.device!r}; expected cuda:<index>."
                        ) from exc
                    if not 0 <= device_index < torch.cuda.device_count():
                        raise MVSepMDX23Error(
                            f"CUDA device index {device_index} is unavailable; "
                            f"found {torch.cuda.device_count()} device(s)."
                        )
            except ImportError as exc:
                raise MVSepMDX23Error(
                    f"Device {self.device!r} requested but torch is not installed."
                ) from exc

            if not self._onnx_cuda_available():
                raise MVSepMDX23Error(
                    f"Device {self.device!r} requested but onnxruntime "
                    "CUDAExecutionProvider is not available; ensure onnxruntime-gpu is installed."
                )
            return False

        # auto / unknown: only skip --cpu when both stacks can use CUDA.
        try:
            import torch

            if not torch.cuda.is_available():
                return True
        except ImportError:
            return True
        return not self._onnx_cuda_available()

    @staticmethod
    def _onnx_cuda_available() -> bool:
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def _ensure_repo(self) -> Path:
        """Clone the upstream repo if needed; return path to ``inference.py``."""
        inference_py = self.repo_dir / "inference.py"
        if inference_py.is_file():
            (self.repo_dir / "models").mkdir(parents=True, exist_ok=True)
            self._patch_upstream_device_override(inference_py)
            return inference_py

        if self.repo_dir.exists() and any(self.repo_dir.iterdir()):
            raise MVSepMDX23Error(
                f"MVSEP repo dir exists but inference.py is missing: {self.repo_dir}"
            )

        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning MVSEP-MDX23 repo into %s", self.repo_dir)
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                self.repo_url,
                str(self.repo_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise MVSepMDX23Error(
                f"failed to clone MVSEP-MDX23 repo: {detail[:2000] or 'no output'}"
            )
        if not inference_py.is_file():
            raise MVSepMDX23Error(
                f"clone finished but inference.py not found at {inference_py}"
            )
        (self.repo_dir / "models").mkdir(parents=True, exist_ok=True)
        self._patch_upstream_device_override(inference_py)
        return inference_py

    @staticmethod
    def _patch_upstream_device_override(inference_py: Path) -> None:
        """Make the cloned CLI preserve the wrapper's CUDA device isolation."""
        source = inference_py.read_text(encoding="utf-8")
        if _UPSTREAM_DEVICE_PATCH in source:
            return
        if _UPSTREAM_DEVICE_OVERRIDE not in source:
            if 'gpu_use = "0"' in source or "gpu_use = '0'" in source:
                raise MVSepMDX23Error(
                    "unsupported upstream GPU-selection block in "
                    f"{inference_py}; refusing to run on an ambiguous device"
                )
            return
        patched_source = source.replace(
            _UPSTREAM_DEVICE_OVERRIDE,
            _UPSTREAM_DEVICE_PATCH,
            1,
        )
        temporary_path = inference_py.with_name(
            f".{inference_py.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary_path.write_text(patched_source, encoding="utf-8")
            temporary_path.replace(inference_py)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _emit_progress(self, message: str) -> None:
        """Log an upstream status line and forward it to the optional caller."""
        logger.info("MVSEP-MDX23: %s", message)
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(message)
        except Exception:
            logger.warning("MVSEP-MDX23 progress callback failed", exc_info=True)

    def _prepare_inputs(self, src: Path) -> list[Path]:
        """Convert source to bounded 44.1 kHz stereo WAV segments."""
        self._emit_progress("Preparing 44.1 kHz stereo input...")
        input_dir = self.work_dir / "input"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True)

        if self.max_segment_seconds is None:
            output_path = input_dir / f"{src.stem}.wav"
        else:
            output_path = input_dir / f"{src.stem}_part_%04d.wav"

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(src),
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_f32le",
        ]
        if self.max_segment_seconds is not None:
            cmd.extend(
                [
                    "-f",
                    "segment",
                    "-segment_time",
                    str(self.max_segment_seconds),
                    "-reset_timestamps",
                    "1",
                ]
            )
        cmd.append(str(output_path))

        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise MVSepMDX23Error(
                f"ffmpeg input conversion failed: {detail[-2000:] or 'no output'}"
            )
        if self.max_segment_seconds is None:
            working_wavs = [output_path]
        else:
            working_wavs = sorted(input_dir.glob("*.wav"))
        if not working_wavs:
            raise MVSepMDX23Error("ffmpeg input conversion produced no WAV segments")
        self._emit_progress(f"Prepared {len(working_wavs)} input segment(s).")
        return working_wavs

    def _build_cmd(
        self,
        inference_py: Path,
        working_wavs: list[Path],
        output_dir: Path,
    ) -> list[str]:
        cmd = [
            self.python_bin,
            "-u",
            str(inference_py),
            "--input_audio",
            *(str(path) for path in working_wavs),
            "--output_folder",
            str(output_dir),
            "--overlap_large",
            str(self.overlap_large),
            "--overlap_small",
            str(self.overlap_small),
        ]
        if self.only_vocals:
            cmd.append("--only_vocals")
        if self._should_use_cpu():
            cmd.append("--cpu")
        if self.single_onnx:
            cmd.append("--single_onnx")
        if self.large_gpu:
            cmd.append("--large_gpu")
        if self.use_kim_model_1:
            cmd.append("--use_kim_model_1")
        if self.chunk_size is not None:
            cmd.extend(["--chunk_size", str(self.chunk_size)])
        return cmd

    def _concat_stem_outputs(self, inputs: list[Path], destination: Path) -> Path:
        """Concatenate separated segments without loading them into host memory."""
        self._emit_progress(f"Joining {len(inputs)} separated segments...")
        cmd = [self.ffmpeg_bin, "-y"]
        for input_path in inputs:
            cmd.extend(["-i", str(input_path)])
        input_labels = "".join(f"[{index}:a:0]" for index in range(len(inputs)))
        cmd.extend(
            [
                "-filter_complex",
                f"{input_labels}concat=n={len(inputs)}:v=0:a=1[out]",
                "-map",
                "[out]",
                "-c:a",
                "pcm_f32le",
                str(destination),
            ]
        )
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise MVSepMDX23Error(
                f"ffmpeg output concatenation failed: {detail[-2000:] or 'no output'}"
            )
        return destination

    def _subprocess_env(self) -> dict[str, str]:
        """Build an environment that exposes only the requested CUDA device."""
        env = os.environ.copy()
        device = str(self.device).lower()
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        elif device.isdigit():
            env["CUDA_VISIBLE_DEVICES"] = device
        return env

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        """Terminate an inference process group, escalating if it does not stop."""
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
        process.wait(timeout=5)

    def _run_inference(self, cmd: list[str]) -> None:
        """Run upstream inference while streaming logs and supporting cancellation."""
        with self._process_lock:
            if self._cancel_requested:
                raise MVSepMDX23Error("MVSEP-MDX23 separation was cancelled")
            if self._process is not None:
                raise MVSepMDX23Error(
                    "MVSepMDX23 does not support concurrent calls on one instance"
                )
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.repo_dir),
                env=self._subprocess_env(),
                start_new_session=True,
            )
            self._process = process

        output_tail: deque[str] = deque(maxlen=80)
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    message = line.rstrip()
                    if not message:
                        continue
                    output_tail.append(message)
                    self._emit_progress(message)
            returncode = process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            with self._process_lock:
                if self._process is process:
                    self._process = None

        if returncode != 0:
            with self._process_lock:
                cancelled = self._cancel_requested
            if cancelled:
                raise MVSepMDX23Error("MVSEP-MDX23 separation was cancelled")
            detail = "\n".join(output_tail).strip()
            raise MVSepMDX23Error(
                f"MVSEP-MDX23 failed (exit {returncode}): "
                f"{detail[-3000:] or 'no output'}"
            )

    def _separate_stem(self, src: Path) -> Path:
        with self._process_lock:
            if self._process is not None:
                raise MVSepMDX23Error(
                    "MVSepMDX23 does not support concurrent calls on one instance"
                )
            if self._cancel_requested:
                raise MVSepMDX23Error("MVSEP-MDX23 separation was cancelled")

        inference_py = self._ensure_repo()
        working_wavs = self._prepare_inputs(src)

        output_dir = self.work_dir / "mvsep_out"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        cmd = self._build_cmd(inference_py, working_wavs, output_dir)
        onnx_models = 1 if self.single_onnx else 2
        self._emit_progress(
            f"Starting on {self.device} with {onnx_models} ONNX model(s) "
            f"and overlap {self.overlap_large:.2f}..."
        )
        logger.info("Running MVSEP-MDX23: %s", " ".join(cmd))
        self._run_inference(cmd)

        stem_id = self._output_stem_id()
        separated_parts = [
            output_dir / f"{working_wav.stem}_{stem_id}.wav"
            for working_wav in working_wavs
        ]
        missing = [path.name for path in separated_parts if not path.is_file()]
        if missing:
            available = sorted(path.name for path in output_dir.glob("*.wav"))
            raise MVSepMDX23Error(
                f"MVSEP-MDX23 finished but stem {stem_id!r} is missing for "
                f"segments {missing}. Available: {available}"
            )
        if len(separated_parts) == 1:
            return separated_parts[0]
        return self._concat_stem_outputs(
            separated_parts,
            output_dir / f"combined_{stem_id}.wav",
        )

    def separate(self, audio: Audio) -> Audio:
        """Separate ``audio`` and return a cleaned ``Audio`` object."""
        src_path = Path(audio.path)
        if not src_path.is_file():
            raise MVSepMDX23Error(f"audio not found: {src_path}")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            separated = self._separate_stem(src_path)
        except FileNotFoundError as exc:  # pragma: no cover
            raise MVSepMDX23Error(
                "MVSEP-MDX23 requires git, ffmpeg, demucs, and onnxruntime-gpu; "
                "install onnxruntime-gpu with: uv pip install onnxruntime-gpu"
            ) from exc

        dest = self.output_dir / f"{src_path.stem}.wav"
        normalize_wav(
            separated,
            dest,
            sample_rate=self.sample_rate,
            channels=self.channels,
            ffmpeg_bin=self.ffmpeg_bin,
        )

        sample_rate, duration_s, channels = probe_wav(dest)
        return audio.with_file(
            dest,
            sample_rate=sample_rate,
            duration_s=duration_s,
            channels=channels,
            step=f"mvsep_mdx23_{self.two_stems}",
        )

    def close(self) -> None:
        """Cancel any active upstream inference process and release its resources."""
        self.cancel()
        with self._process_lock:
            process = self._process
        if process is not None:
            self._terminate_process(process)

    def cancel(self) -> None:
        """Request non-blocking cancellation of active upstream inference."""
        with self._process_lock:
            self._cancel_requested = True
            process = self._process
        if process is None or process.poll() is not None:
            return
        self._emit_progress("Cancelling active inference process...")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()
