"""MVSEP-MDX23 separation backend via ZFTurbo's ``inference.py`` CLI.

Wraps https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model for
vocals cleanup (``--only_vocals`` by default). The upstream repo is cloned
on first use; checkpoints download themselves on the first inference run.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.separation.audio_utils import normalize_wav, probe_wav
from src.separation.BaseSeparator import BaseSeparator
from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio

logger = logging.getLogger(__name__)

DEFAULT_REPO_URL = (
    "https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model.git"
)

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
        single_onnx: bool = False,
        large_gpu: bool = False,
        use_kim_model_1: bool = False,
        overlap_large: float = 0.6,
        overlap_small: float = 0.5,
        chunk_size: int | None = None,
        python_bin: Optional[str] = None,
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
        self.python_bin = python_bin or sys.executable

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
        return inference_py

    def _prepare_input(self, src: Path) -> Path:
        """Convert source to a working stereo WAV for upstream ``librosa`` load."""
        input_dir = self.work_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        working_wav = input_dir / f"{src.stem}.wav"
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(src),
            "-ac",
            "2",
            "-c:a",
            "pcm_f32le",
            str(working_wav),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise MVSepMDX23Error(
                f"ffmpeg input conversion failed: {detail[-2000:] or 'no output'}"
            )
        return working_wav

    def _build_cmd(self, inference_py: Path, working_wav: Path, output_dir: Path) -> list[str]:
        cmd = [
            self.python_bin,
            str(inference_py),
            "--input_audio",
            str(working_wav),
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

    def _separate_stem(self, src: Path) -> Path:
        inference_py = self._ensure_repo()
        working_wav = self._prepare_input(src)

        output_dir = self.work_dir / "mvsep_out"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        cmd = self._build_cmd(inference_py, working_wav, output_dir)
        logger.info("Running MVSEP-MDX23: %s", " ".join(cmd))
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(self.repo_dir),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise MVSepMDX23Error(
                f"MVSEP-MDX23 failed (exit {completed.returncode}): "
                f"{detail[-3000:] or 'no output'}"
            )

        stem_id = self._output_stem_id()
        separated = output_dir / f"{working_wav.stem}_{stem_id}.wav"
        if not separated.is_file():
            matches = list(output_dir.glob(f"*_{stem_id}.wav"))
            if not matches:
                available = sorted(p.name for p in output_dir.glob("*.wav"))
                raise MVSepMDX23Error(
                    f"MVSEP-MDX23 finished but stem {stem_id!r} not found under "
                    f"{output_dir}. Available: {available}"
                )
            separated = matches[0]
        return separated

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
