"""Mel-Band RoFormer separation backend via ``melband-roformer-infer``.

Uses the lifecycle ``MelBandRoformerSession`` API so the Kimberley Jensen
(or other registry) checkpoint is loaded once and reused across many
``Audio`` objects.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

from mel_band_roformer import MelBandRoformerSession

from src.base.model import ManagedModel
from src.utils.audio_utils import (
    AudioConvertError,
    normalize_wav,
    prepare_separator_wav,
    probe_wav,
)
from src.separation.BaseSeparator import BaseSeparator
from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio

logger = logging.getLogger(__name__)


class MelRoFormerError(RuntimeError):
    """Raised when Mel-Band RoFormer separation cannot run or the output is unusable."""


def _iter_manifest_outputs(manifest: Any):
    """Yield ``(output_id, output_path)`` from either package manifest shape."""
    outputs = getattr(manifest, "outputs", None)
    if outputs is not None:
        for output in outputs:
            yield output.output_id, Path(output.output_path)
        return

    for output in manifest:
        yield output["output_id"], Path(output["output_path"])


class MelRoFormer(BaseSeparator, ManagedModel):
    """Mel-Band RoFormer backend for vocal/music separation.

    Default model is Kimberley Jensen's MelBand Roformer vocals checkpoint
    (``melband-roformer-kim-vocals``), matching
    https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model.

    Usage:
        separator = MelRoFormer(device="mps")
        separator.load()
        cleaned = separator.separate(audio)
        separator.unload()
    """

    def __init__(
        self,
        model: str = "melband-roformer-kim-vocals",
        device: str = "auto",
        two_stems: str = "vocals",
        output_dir: str | Path = ".data/mel_roformer/out",
        work_dir: str | Path = ".data/mel_roformer/work",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        ffmpeg_bin: Optional[str] = None,
        backend: Optional[str] = None,
        model_sample_rate: int = 44100,
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
        ManagedModel.__init__(self)
        self.backend = backend
        self.model_sample_rate = model_sample_rate
        self._session: MelBandRoformerSession | None = None

    def _load(self) -> None:
        """Create and load the Mel-Band RoFormer session."""
        # ``auto`` means CUDA-else-CPU; on Apple Silicon pass device="mps".
        session_device = None if self.device in ("auto", "None") else self.device
        session = MelBandRoformerSession(
            model_name=self.model,
            device=session_device,
            backend=self.backend,
        )
        session.load()
        self._session = session

    def _unload(self) -> None:
        """Close the Mel-Band RoFormer session and clear its reference."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _prepare_input(self, src: Path) -> tuple[Path, Path]:
        """Convert source to 44.1 kHz stereo WAV for the checkpoint STFT.

        ``MelBandRoformerSession.infer`` processes a folder, so the input
        directory is wiped and rebuilt to contain only this sample.
        """
        input_dir = self.work_dir / "input"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True)

        working_wav = input_dir / f"{src.stem}.wav"
        try:
            prepare_separator_wav(
                src,
                working_wav,
                sample_rate=self.model_sample_rate,
                channels=2,
                ffmpeg_bin=self.ffmpeg_bin,
            )
        except AudioConvertError as exc:
            raise MelRoFormerError(f"ffmpeg input conversion failed: {exc}") from exc
        return working_wav, input_dir

    def _separate_stem(self, src: Path) -> Path:
        _, input_dir = self._prepare_input(src)

        output_dir = self.work_dir / "roformer_out"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        session = self._session
        if session is None:  # pragma: no cover - guarded by separate()
            raise MelRoFormerError(
                "Mel-Band RoFormer is not loaded. Call load() before separate(), "
                "or use it as a context manager."
            )
        logger.info(
            "Running Mel-Band RoFormer model=%s device=%s stem=%s",
            self.model,
            self.device,
            self.two_stems,
        )
        manifest = session.infer(str(input_dir), store_dir=str(output_dir))

        available: list[str] = []
        for output_id, output_path in _iter_manifest_outputs(manifest):
            available.append(output_id)
            if output_id == self.two_stems:
                return output_path

        raise MelRoFormerError(
            f"Mel-Band RoFormer did not produce stem {self.two_stems!r}. "
            f"Available: {sorted(set(available))}"
        )

    def separate(self, audio: Audio) -> Audio:
        """Separate ``audio`` and return a cleaned ``Audio`` object."""
        if not self.is_loaded:
            raise MelRoFormerError(
                "Mel-Band RoFormer is not loaded. Call load() before separate(), "
                "or use it as a context manager."
            )

        src_path = Path(audio.path)
        if not src_path.is_file():
            raise MelRoFormerError(f"audio not found: {src_path}")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        separated = self._separate_stem(src_path)

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
            step=f"mel_roformer_{self.two_stems}",
        )

    def close(self) -> None:
        """Compatibility alias for :meth:`unload`."""
        self.unload()
