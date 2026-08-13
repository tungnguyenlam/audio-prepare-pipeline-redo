"""Compare two file-backed ``Audio`` objects as aligned waveforms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.AudioClass import Audio

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class WaveformCompareError(ValueError):
    """Raised when two clips cannot be aligned or plotted."""


class WaveformComparer:
    """Plot mixture, estimate, and residual waveforms.

    Both files are resampled to ``sample_rate`` and truncated to the shared
    length so the time axes match. Amplitude limits are shared across panels.

    Example::

        comparer = WaveformComparer(sample_rate=16000)
        comparer.compare(mixture, vocals)
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        if sample_rate <= 0:
            raise WaveformCompareError(
                f"sample_rate must be positive, got {sample_rate}"
            )
        self.sample_rate = sample_rate

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
        """Display aligned waveforms for ``before`` vs ``after``.

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
            WaveformCompareError: If either clip is empty after alignment.
        """
        import librosa.display
        import matplotlib.pyplot as plt
        import numpy as np

        y_before, y_after = self._load_aligned(before, after)
        y_residual = y_before - y_after

        peak = float(
            max(
                np.max(np.abs(y_before)),
                np.max(np.abs(y_after)),
                np.max(np.abs(y_residual)),
            )
        )
        if peak == 0.0:
            peak = 1.0

        fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True)
        panels = [
            (y_before, before_title, "C0"),
            (y_after, after_title, "C0"),
            (y_residual, residual_title, "C3"),
        ]

        for ax, (y, title, color) in zip(axes, panels):
            librosa.display.waveshow(y, sr=self.sample_rate, ax=ax, color=color)
            ax.set_ylim(-peak, peak)
            ax.set_title(title)
            ax.set_ylabel("Amplitude")

        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        if show:
            # Show without returning the figure, or Jupyter renders it twice.
            plt.show()
            plt.close(fig)
            return None
        return fig

    def _load_aligned(self, before: Audio, after: Audio) -> tuple[NDArray, NDArray]:
        y_before = self._load_mono(before)
        y_after = self._load_mono(after)
        n = min(len(y_before), len(y_after))
        if n == 0:
            raise WaveformCompareError(
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
