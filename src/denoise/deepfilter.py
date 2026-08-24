"""DeepFilterNet3 Speech Enhancement & Denoising Implementation.

Stage 5 / Stage 7 in SOTA Audio Processing Pipeline:
Suppresses background physical environmental noise, room reverberation, air conditioner hum,
and residual sound effects/meme audio while strictly preserving natural vocal formants and timbre.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.utils.AudioClass import Audio, _probe_wav, _write_sidecar

logger = logging.getLogger(__name__)


def _ensure_torchaudio_compat():
    """Shim torchaudio.backend and AudioMetaData for torchaudio 2.x+ compatibility with DeepFilterNet."""
    try:
        if not hasattr(torchaudio, "backend") or "torchaudio.backend" not in sys.modules:
            backend_mod = types.ModuleType("torchaudio.backend")
            common_mod = types.ModuleType("torchaudio.backend.common")
            
            @dataclass
            class AudioMetaData:
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int = 16
                encoding: str = "PCM_S"

            common_mod.AudioMetaData = AudioMetaData
            backend_mod.common = common_mod
            sys.modules["torchaudio.backend"] = backend_mod
            sys.modules["torchaudio.backend.common"] = common_mod
    except Exception:
        pass


class DeepFilterEnhancer:
    """DeepFilterNet3 Speech Enhancement wrapper with chunked streaming to prevent GPU OOM."""

    def __init__(self, post_filter: bool = True):
        self.post_filter = post_filter
        self._model = None
        self._df_state = None

    def _lazy_init(self):
        if self._model is None:
            _ensure_torchaudio_compat()
            try:
                from df.enhance import init_df
                logger.info("Initializing DeepFilterNet3 speech enhancer model...")
                self._model, self._df_state, _ = init_df(post_filter=self.post_filter)
            except ImportError:
                raise ImportError(
                    "DeepFilterNet is not installed. Install via `pip install deepfilternet`."
                )

    def enhance_tensor(self, audio_tensor: torch.Tensor, orig_sr: int) -> Tuple[torch.Tensor, int]:
        """Enhance an in-memory 1D/2D audio tensor with DeepFilterNet3."""
        self._lazy_init()
        from df.enhance import enhance

        target_sr = self._df_state.sr()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim > 2:
            audio_tensor = audio_tensor.squeeze()

        if orig_sr != target_sr:
            t_audio = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
        else:
            t_audio = audio_tensor

        # For long audio (> 30s), process in 20s chunks to maintain < 50MB VRAM
        chunk_len = 20 * target_sr
        total_len = t_audio.shape[-1]

        if total_len <= chunk_len:
            enhanced = enhance(self._model, self._df_state, t_audio)
            return enhanced, target_sr

        enhanced_list = []
        for start_idx in range(0, total_len, chunk_len):
            chunk = t_audio[:, start_idx : min(total_len, start_idx + chunk_len)]
            if chunk.shape[-1] < 1024:
                break
            enh_chunk = enhance(self._model, self._df_state, chunk)
            enhanced_list.append(enh_chunk)

        full_enhanced = torch.cat(enhanced_list, dim=-1)
        return full_enhanced, target_sr

    def enhance(
        self,
        input_audio: Union[str, Path, Audio],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Audio:
        """Enhance audio file by suppressing environmental noise and acoustic artifacts."""
        src_path = Path(input_audio.path if isinstance(input_audio, Audio) else input_audio)
        if not src_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {src_path}")

        dest_path = Path(output_path) if output_path else src_path.with_name(f"{src_path.stem}_denoised.wav")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        audio_data, orig_sr = sf.read(str(src_path), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        t_audio = torch.from_numpy(audio_data).float().unsqueeze(0)
        enhanced_tensor, target_sr = self.enhance_tensor(t_audio, orig_sr)
        out_np = enhanced_tensor.squeeze().cpu().numpy()

        # Save enhanced audio
        sf.write(str(dest_path), out_np, target_sr, subtype="PCM_16")

        rate, dur, ch = _probe_wav(dest_path)

        if isinstance(input_audio, Audio):
            out_audio = input_audio.derive(
                path=dest_path,
                sample_rate=rate,
                duration_s=dur,
                channels=ch,
                step="denoised_dfnet3"
            )
        else:
            out_audio = Audio(
                path=dest_path.resolve(),
                source_id=src_path.stem,
                title=src_path.stem,
                sample_rate=rate,
                duration_s=dur,
                channels=ch,
                format="wav"
            )
            _write_sidecar(out_audio)

        return out_audio


def enhance_audio_file(
    input_path: Union[str, Path], 
    output_path: Optional[Union[str, Path]] = None
) -> Audio:
    """Helper function to quickly enhance a single audio file."""
    enhancer = DeepFilterEnhancer()
    return enhancer.enhance(input_path, output_path)
