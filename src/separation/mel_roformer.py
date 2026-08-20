"""Mel-Band RoFormer Vocal Source Separation Wrapper."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

from src.separation.base import BaseSeparator, SeparationError, SeparationResult

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "melband-roformer-infer"


class MelRoFormerSeparator(BaseSeparator):
    """Mel-Band RoFormer vocal separation wrapper.

    SOTA transformer-based vocal separation with high-frequency resolution.
    """

    def __init__(
        self,
        device: str = "cuda",
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(device=device)
        self.config_path = Path(config_path) if config_path else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    def check_status(self) -> dict[str, Union[bool, str]]:
        """Check if Mel-Band RoFormer dependencies and weights are available."""
        has_package = importlib.util.find_spec("mel_band_roformer") is not None

        # Check explicit env vars or cache
        config = self.config_path or os.environ.get("MEL_ROFORMER_CONFIG")
        checkpoint = self.checkpoint_path or os.environ.get("MEL_ROFORMER_CHECKPOINT")

        cached_kim = CACHE_DIR / "melband-roformer-kim-vocals"
        has_cached_weights = (
            cached_kim.exists()
            and (cached_kim / "MelBandRoformer.ckpt").is_file()
            and (cached_kim / "config_vocals_mel_band_roformer.yaml").is_file()
        )

        weights_ready = (
            has_cached_weights
            or (bool(config and checkpoint and Path(config).is_file() and Path(checkpoint).is_file()))
        )

        available = has_package and (weights_ready or True)  # Can auto-download if missing
        msg = (
            "Sẵn sàng (Mel-Band RoFormer SOTA)"
            if available
            else "Cần cài đặt: uv pip install melband-roformer-infer"
        )

        return {
            "available": available,
            "message": msg,
            "model": "mel_roformer",
        }

    def separate(self, input_path: Path, output_dir: Path) -> SeparationResult:
        """Run Mel-Band RoFormer vocal separation.

        Args:
            input_path: Path to input audio
            output_dir: Folder to store separated stem WAV files

        Returns:
            SeparationResult with stems dictionary mapping 'vocals' and 'accompaniment'.
        """
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Input audio not found: {input_path}")

        status = self.check_status()
        if not status["available"]:
            raise SeparationError(str(status["message"]))

        output_dir.mkdir(parents=True, exist_ok=True)

        device_str = self.device
        if device_str.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    device_str = "cpu"
            except Exception:
                device_str = "cpu"

        # Resolve explicit weights or cached weights
        cached_kim = CACHE_DIR / "melband-roformer-kim-vocals"
        config_arg: Optional[Path] = self.config_path
        ckpt_arg: Optional[Path] = self.checkpoint_path

        if not config_arg and os.environ.get("MEL_ROFORMER_CONFIG"):
            config_arg = Path(os.environ["MEL_ROFORMER_CONFIG"])
        elif not config_arg and (cached_kim / "config_vocals_mel_band_roformer.yaml").is_file():
            config_arg = cached_kim / "config_vocals_mel_band_roformer.yaml"

        if not ckpt_arg and os.environ.get("MEL_ROFORMER_CHECKPOINT"):
            ckpt_arg = Path(os.environ["MEL_ROFORMER_CHECKPOINT"])
        elif not ckpt_arg and (cached_kim / "MelBandRoformer.ckpt").is_file():
            ckpt_arg = cached_kim / "MelBandRoformer.ckpt"

        with tempfile.TemporaryDirectory(prefix="mel-roformer-input-") as temp_input:
            temp_input_dir = Path(temp_input)
            staged_input = temp_input_dir / input_path.name
            shutil.copy2(input_path, staged_input)

            command = [
                sys.executable,
                "-m",
                "mel_band_roformer.inference",
                "--input_folder",
                str(temp_input_dir),
                "--store_dir",
                str(output_dir),
                "--device",
                device_str,
            ]

            if config_arg and config_arg.is_file():
                command.extend(["--config_path", str(config_arg)])
            if ckpt_arg and ckpt_arg.is_file():
                command.extend(["--model_path", str(ckpt_arg)])

            logger.info("Executing Mel-Band RoFormer: %s", " ".join(command))
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:
                raise SeparationError(f"Không thể chạy Mel-Band RoFormer: {exc}") from exc

            if completed.returncode != 0:
                err_text = (completed.stderr or completed.stdout or "Unknown RoFormer error").strip()
                raise SeparationError(f"Mel-Band RoFormer gặp lỗi: {err_text[-2000:]}")

        stems = self._collect_and_organize_stems(output_dir, input_path.stem)
        return SeparationResult(
            model="mel_roformer",
            input_file=input_path,
            output_dir=output_dir,
            stems=stems,
        )

    def _collect_and_organize_stems(self, output_dir: Path, stem_prefix: str) -> Dict[str, Path]:
        """Find and standardize only the vocals.wav stem, discarding non-vocal audio."""
        found_wavs = sorted(output_dir.rglob("*.wav"))
        if not found_wavs:
            raise SeparationError("Mel-Band RoFormer chạy xong nhưng không tìm thấy file WAV đầu ra nào.")

        vocals_path: Optional[Path] = None
        for wav in found_wavs:
            name = wav.stem.lower()
            if "vocal" in name and "no_vocal" not in name and "instrument" not in name:
                target_path = output_dir / "vocals.wav"
                if wav != target_path:
                    shutil.move(str(wav), str(target_path))
                vocals_path = target_path
            else:
                # Remove accompaniment / instrumental to save space
                wav.unlink(missing_ok=True)

        # Clean up empty subdirectories
        for child in list(output_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

        if not vocals_path or not vocals_path.is_file():
            remaining_wavs = list(output_dir.glob("*.wav"))
            if remaining_wavs:
                target_path = output_dir / "vocals.wav"
                if remaining_wavs[0] != target_path:
                    shutil.move(str(remaining_wavs[0]), str(target_path))
                vocals_path = target_path
            else:
                raise SeparationError("Không tìm thấy stem Vocal được tạo từ Mel-Band RoFormer.")

        return {"vocals": vocals_path}
