"""HT Demucs (v4) Vocal & Music Source Separation Wrapper."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from src.separation.base import BaseSeparator, SeparationError, SeparationResult

logger = logging.getLogger(__name__)


class HTDemucsSeparator(BaseSeparator):
    """HT Demucs source separator (Facebook Research).

    Separates audio into Vocals and Accompaniment stems (or 4 stems).
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cuda",
        shifts: int = 1,
        overlap: float = 0.25,
        two_stems: str = "vocals",
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.shifts = shifts
        self.overlap = overlap
        self.two_stems = two_stems

    def check_status(self) -> dict[str, Union[bool, str]]:
        """Check if demucs is installed and ready."""
        installed = importlib.util.find_spec("demucs") is not None
        return {
            "available": installed,
            "message": "Sẵn sàng (HT Demucs v4)" if installed else "Chưa cài Demucs: uv pip install demucs",
            "model": self.model_name,
        }

    def separate(self, input_path: Path, output_dir: Path) -> SeparationResult:
        """Run HT Demucs vocal separation on input audio.

        Args:
            input_path: Path to input audio file
            output_dir: Destination folder for output stems

        Returns:
            SeparationResult containing mapping of stems (e.g. {'vocals': Path, 'accompaniment': Path})
        """
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Input audio not found: {input_path}")

        status = self.check_status()
        if not status["available"]:
            raise SeparationError(str(status["message"]))

        output_dir.mkdir(parents=True, exist_ok=True)

        device_name = self.device
        if device_name.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    device_name = "cpu"
            except Exception:
                device_name = "cpu"

        command = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            self.model_name,
            "--two-stems",
            self.two_stems,
            "--device",
            device_name,
            "--shifts",
            str(self.shifts),
            "--overlap",
            str(self.overlap),
            "--out",
            str(output_dir),
            str(input_path),
        ]

        logger.info("Executing Demucs command: %s", " ".join(command))
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise SeparationError(f"Không thể chạy HT Demucs: {exc}") from exc

        if completed.returncode != 0:
            error_msg = (completed.stderr or completed.stdout or "Unknown Demucs error").strip()
            raise SeparationError(f"HT Demucs gặp lỗi: {error_msg[-2000:]}")

        stems = self._collect_and_organize_stems(output_dir, input_path.stem)
        return SeparationResult(
            model="htdemucs",
            input_file=input_path,
            output_dir=output_dir,
            stems=stems,
        )

    def _collect_and_organize_stems(self, output_dir: Path, stem_prefix: str) -> Dict[str, Path]:
        """Find and standardize only the vocals.wav stem, discarding non-vocal audio."""
        found_wavs = sorted(output_dir.rglob("*.wav"))
        if not found_wavs:
            raise SeparationError("HT Demucs chạy xong nhưng không tìm thấy file WAV đầu ra nào.")

        vocals_path: Optional[Path] = None
        for wav in found_wavs:
            stem_name = wav.stem.lower()
            if stem_name == "vocals":
                target_path = output_dir / "vocals.wav"
                if wav != target_path:
                    shutil.move(str(wav), str(target_path))
                vocals_path = target_path
            else:
                # Remove accompaniment / no_vocals to keep output clean and save disk space
                wav.unlink(missing_ok=True)

        # Clean up empty subdirectories created by Demucs
        for child in list(output_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

        if not vocals_path or not vocals_path.is_file():
            # If named differently, take the first available WAV as vocals
            remaining_wavs = list(output_dir.glob("*.wav"))
            if remaining_wavs:
                target_path = output_dir / "vocals.wav"
                if remaining_wavs[0] != target_path:
                    shutil.move(str(remaining_wavs[0]), str(target_path))
                vocals_path = target_path
            else:
                raise SeparationError("Không tìm thấy stem Vocal được tạo từ HT Demucs.")

        return {"vocals": vocals_path}
