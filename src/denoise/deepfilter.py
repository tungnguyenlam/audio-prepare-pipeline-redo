"""DeepFilterNet3 Speech Enhancement & Denoising Implementation.

Stage 5 in SOTA Audio Processing Pipeline:
Suppresses background physical environmental noise (fans, traffic, office) 
while strictly preserving natural vocal formants, timbre, and non-verbal human vocalizations (crying, laughter, sighs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from src.utils.AudioClass import Audio, _probe_wav, _write_sidecar

logger = logging.getLogger(__name__)


class DeepFilterEnhancer:
    """DeepFilterNet3 Speech Enhancement wrapper."""

    def __init__(self, post_filter: bool = True):
        self.post_filter = post_filter
        self._model = None
        self._df_state = None

    def _lazy_init(self):
        if self._model is None:
            try:
                from df.enhance import init_df
                logger.info("Initializing DeepFilterNet3 model...")
                self._model, self._df_state, _ = init_df(post_filter=self.post_filter)
            except ImportError:
                raise ImportError(
                    "DeepFilterNet is not installed. Install via `pip install deepfilternet`."
                )

    def enhance(
        self,
        input_audio: Union[str, Path, Audio],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Audio:
        """Enhance audio by suppressing environmental noise."""
        self._lazy_init()
        from df.enhance import enhance, load_audio, save_audio

        src_path = Path(input_audio.path if isinstance(input_audio, Audio) else input_audio)
        if not src_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {src_path}")

        dest_path = Path(output_path) if output_path else src_path.with_name(f"{src_path.stem}_denoised.wav")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        audio_tensor, sr = load_audio(str(src_path), sr=self._df_state.sr())
        enhanced_tensor = enhance(self._model, self._df_state, audio_tensor)
        save_audio(str(dest_path), enhanced_tensor, sr)

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
