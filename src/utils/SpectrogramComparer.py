"""Compare two file-backed ``Audio`` objects as aligned mel spectrograms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.utils.AudioClass import Audio

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class SpectrogramCompareError(ValueError):
    """Raised when two clips cannot be aligned or plotted."""


class SpectrogramComparer:
    """Plot mixture, estimate, and residual mel spectrograms.

    Both files are resampled to ``sample_rate``, truncated to the shared
    length, and converted with one dB reference so the color scales match.

    Example::

        comparer = SpectrogramComparer(sample_rate=16000)
        comparer.compare(mixture, vocals)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        hop_length: int = 512,
        fmax: Optional[float] = None,
        top_db: float = 80.0,
        ref: float = 1.0,
    ) -> None:
        if sample_rate <= 0:
            raise SpectrogramCompareError(
                f"sample_rate must be positive, got {sample_rate}"
            )
        if n_mels <= 0:
            raise SpectrogramCompareError(f"n_mels must be positive, got {n_mels}")
        if hop_length <= 0:
            raise SpectrogramCompareError(
                f"hop_length must be positive, got {hop_length}"
            )
        if top_db <= 0:
            raise SpectrogramCompareError(f"top_db must be positive, got {top_db}")
        if ref <= 0:
            raise SpectrogramCompareError(f"ref must be positive, got {ref}")

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fmax = fmax
        self.top_db = top_db
        self.ref = ref

    def compare(
        self,
        before: Audio,
        after: Audio,
        *,
        before_title: str = "Before separation (mixture)",
        after_title: str = "After separation (vocals)",
        residual_title: str = "Residual (removed)",
        show: bool = True,
    ) -> Figure | None:
        """Display aligned spectrograms for ``before`` vs ``after``.

        Panels are mixture, estimate, and time-domain residual
        (``before - after``).

        Args:
            before: Mixture (or pre-separation) ``Audio``.
            after: Estimate (or post-separation) ``Audio``.
            before_title: Label for the first panel.
            after_title: Label for the second panel.
            residual_title: Label for the residual panel.
            show: Whether to render the figure. When true the figure is
                displayed and not returned, so Jupyter does not draw it twice.

        Returns:
            ``None`` when ``show`` is true, otherwise the Matplotlib figure.

        Raises:
            FileNotFoundError: If either audio path is missing.
            SpectrogramCompareError: If either clip is empty after alignment.
        """
        import librosa.display
        import matplotlib.pyplot as plt

        y_before, y_after = self._load_aligned(before, after)
        y_residual = y_before - y_after

        s_before = self._mel_db(y_before)
        s_after = self._mel_db(y_after)
        s_residual = self._mel_db(y_residual)

        fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True)
        vmin, vmax = -self.top_db, 0.0
        panels = [
            (s_before, before_title),
            (s_after, after_title),
            (s_residual, residual_title),
        ]

        for ax, (spec, title) in zip(axes, panels):
            img = librosa.display.specshow(
                spec,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="mel",
                ax=ax,
                vmin=vmin,
                vmax=vmax,
            )
            colorbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
            colorbar.set_label("dB")
            ax.set_title(title)
            ax.set_ylabel("Mel frequency")

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        if show:
            # Show without returning the figure, or Jupyter renders it twice.
            plt.show()
            plt.close(fig)
            return None
        return fig

    def _load_aligned(self, before: Audio, after: Audio) -> tuple[NDArray, NDArray]:
        import librosa

        y_before = self._load_mono(before)
        y_after = self._load_mono(after)
        n = min(len(y_before), len(y_after))
        if n == 0:
            raise SpectrogramCompareError(
                "aligned audio is empty; both clips must contain samples"
            )
        return y_before[:n], y_after[:n]

    def _load_mono(self, audio: Audio) -> NDArray:
        import librosa

        path = Path(audio.path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        y, _sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return y

    def _mel_db(self, y: NDArray) -> NDArray:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            fmax=self.fmax,
        )
        return librosa.power_to_db(mel, ref=self.ref, top_db=self.top_db)
