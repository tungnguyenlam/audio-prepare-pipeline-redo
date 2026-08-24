"""DeepFilterNet (v3) Speech Enhancement & Denoising Separator Wrapper."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import soundfile as sf
import torch

from src.separation.base import BaseSeparator, SeparationError, SeparationResult

logger = logging.getLogger(__name__)


def _ensure_torchaudio_compat():
    """Shim torchaudio.backend for torchaudio 2.x+ compatibility with DeepFilterNet."""
    import types
    try:
        import torchaudio
        if not hasattr(torchaudio, "backend") or "torchaudio.backend" not in sys.modules:
            backend_mod = types.ModuleType("torchaudio.backend")
            common_mod = types.ModuleType("torchaudio.backend.common")
            common_mod.AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
            backend_mod.common = common_mod
            sys.modules["torchaudio.backend"] = backend_mod
            sys.modules["torchaudio.backend.common"] = common_mod
    except Exception:
        pass


class DeepFilterNetSeparator(BaseSeparator):
    """DeepFilterNet3 speech enhancement and noise suppression wrapper (Rikorose).

    Enhances speech, eliminates background acoustic noise, environment hiss and reverberation.
    """

    def __init__(
        self,
        model_name: str = "DeepFilterNet3",
        device: str = "cuda",
        post_filter: bool = False,
        atten_lim_db: Optional[float] = None,
        chunk_sec: float = 30.0,
        overlap_sec: float = 1.0,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.post_filter = post_filter
        self.atten_lim_db = atten_lim_db
        self.chunk_sec = chunk_sec
        self.overlap_sec = overlap_sec
        self._model = None
        self._df_state = None

    def _init_model(self):
        """Lazy loader for DeepFilterNet model."""
        if self._model is not None and self._df_state is not None:
            return self._model, self._df_state

        _ensure_torchaudio_compat()
        try:
            from df.enhance import init_df
            logger.info("Initializing DeepFilterNet model (%s)...", self.model_name)
            model, df_state, _ = init_df(
                default_model=self.model_name,
                post_filter=self.post_filter,
                log_level="ERROR",
                log_file=None,
                config_allow_defaults=True,
            )
            self._model = model
            self._df_state = df_state
            return self._model, self._df_state
        except Exception as e:
            raise SeparationError(f"Không thể khởi tạo mô hình DeepFilterNet: {e}") from e

    def check_status(self) -> dict[str, Union[bool, str]]:
        """Check if DeepFilterNet is installed and ready."""
        _ensure_torchaudio_compat()
        try:
            import df
            from df.enhance import init_df
            return {
                "available": True,
                "message": "Sẵn sàng (DeepFilterNet3 SOTA Denoising & Enhancement)",
                "model": self.model_name,
            }
        except Exception as e:
            return {
                "available": False,
                "message": f"Chưa cài DeepFilterNet: {e}",
                "model": self.model_name,
            }

    def _enhance_chunked(
        self,
        model,
        df_state,
        audio_tensor: torch.Tensor,
        atten_lim_db: Optional[float] = None,
    ) -> np.ndarray:
        """Process audio in overlapping chunks for smooth continuous output and optimal GPU memory."""
        from df.enhance import enhance

        target_sr = df_state.sr()
        total_samples = audio_tensor.shape[-1]
        chunk_samples = int(self.chunk_sec * target_sr)
        overlap_samples = int(self.overlap_sec * target_sr)

        # Attenuation limit in dB (None or >= 100 means full attenuation / max noise suppression)
        eff_atten = None if (atten_lim_db is None or atten_lim_db >= 100) else float(atten_lim_db)

        # If audio is shorter than chunk length, process in a single pass
        if total_samples <= chunk_samples:
            with torch.no_grad():
                out = enhance(model, df_state, audio_tensor, pad=True, atten_lim_db=eff_atten)
            return out.squeeze().cpu().detach().numpy()

        hop_samples = chunk_samples - overlap_samples
        output_audio = np.zeros(total_samples, dtype=np.float32)
        weight = np.zeros(total_samples, dtype=np.float32)

        # Hann / linear crossfade window for overlapping regions
        window = np.ones(chunk_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
        window[:overlap_samples] = fade_in
        window[-overlap_samples:] = fade_out

        for start in range(0, total_samples, hop_samples):
            end = min(start + chunk_samples, total_samples)
            cur_len = end - start
            chunk = audio_tensor[:, start:end]

            if cur_len < chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - cur_len))

            with torch.no_grad():
                out = enhance(model, df_state, chunk, pad=True, atten_lim_db=eff_atten)

            out_np = out.squeeze().cpu().detach().numpy()[:cur_len]

            w = window[:cur_len].copy()
            if start == 0:
                w[:overlap_samples] = 1.0
            if end == total_samples:
                tail_overlap = min(overlap_samples, cur_len)
                w[-tail_overlap:] = 1.0

            output_audio[start:end] += out_np * w
            weight[start:end] += w

        weight[weight == 0] = 1.0
        output_audio /= weight
        return output_audio

    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        atten_lim_db: Optional[float] = None,
        post_filter: Optional[bool] = None,
    ) -> SeparationResult:
        """Enhance and isolate clean voice from input audio using DeepFilterNet3.

        Args:
            input_path: Path to input audio file
            output_dir: Destination folder for output vocal stem
            atten_lim_db: Noise attenuation limit in dB (e.g. 100 for max, 12-40 for custom)
            post_filter: Whether to enable DeepFilterNet post filter

        Returns:
            SeparationResult containing mapping of stems ({'vocals': Path})
        """
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()

        if not input_path.is_file():
            raise FileNotFoundError(f"Input audio not found: {input_path}")

        status = self.check_status()
        if not status["available"]:
            raise SeparationError(str(status["message"]))

        # Override post_filter if explicitly passed
        if post_filter is not None and post_filter != self.post_filter:
            self.post_filter = post_filter
            self._model = None
            self._df_state = None

        eff_atten = atten_lim_db if atten_lim_db is not None else self.atten_lim_db

        output_dir.mkdir(parents=True, exist_ok=True)
        output_vocal_path = output_dir / "vocals.wav"

        t0 = time.time()
        model, df_state = self._init_model()
        target_sr = df_state.sr()

        try:
            import torchaudio

            # Read original audio
            info = sf.info(str(input_path))
            orig_sr = info.samplerate
            audio_data, _ = sf.read(str(input_path), dtype="float32")

            # Ensure shape is [Channels, Samples]
            if audio_data.ndim == 1:
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            elif audio_data.ndim == 2:
                # If multichannel, take mean to mono or preserve channels
                audio_tensor = torch.from_numpy(audio_data.T)
                if audio_tensor.shape[0] > 1:
                    audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            else:
                raise SeparationError(f"Unsupported audio dimension: {audio_data.ndim}")

            # Resample to DeepFilterNet target sample rate (48000 Hz)
            if orig_sr != target_sr:
                audio_tensor = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)

            logger.info("Running DeepFilterNet enhancement on audio tensor: %s", audio_tensor.shape)

            enhanced_np = self._enhance_chunked(model, df_state, audio_tensor, atten_lim_db=eff_atten)

            # Saving as 48kHz PCM_16 WAV gives high resolution
            sf.write(str(output_vocal_path), enhanced_np, target_sr, subtype="PCM_16")

            elapsed = time.time() - t0
            logger.info("DeepFilterNet enhancement finished in %.2fs: %s", elapsed, output_vocal_path)

            return SeparationResult(
                model="deepfilternet",
                input_file=input_path,
                output_dir=output_dir,
                stems={"vocals": output_vocal_path},
            )

        except Exception as exc:
            logger.error("DeepFilterNet enhancement failed: %s", exc, exc_info=True)
            raise SeparationError(f"Lỗi khi xử lý qua DeepFilterNet: {exc}") from exc
