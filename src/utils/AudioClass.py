"""Audio dataclass representation module."""

from __future__ import annotations

import re
import shutil
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Optional

DEFAULT_SAMPLE_RATE = 44100
ResampleAction = Literal["upscale", "downscale", "keep"]


def _sanitize_filename_component(name: str) -> str:
    """Sanitize a string for safe inclusion in filenames across filesystems."""
    cleaned = re.sub(r'[\\/*?:"<>|\s]+', "_", name)
    return cleaned.strip("._")



def _probe_wav(path: Path) -> tuple[int, float, int]:
    """Return ``(sample_rate, duration_s, channels)`` for a WAV file."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.getnframes()
        channels = wf.getnchannels()
    duration = frames / float(rate) if rate else 0.0
    return rate, duration, channels


@dataclass
class Audio:
    """Represents a downloaded and standardized audio file."""

    path: Path
    source_id: str
    title: Optional[str] = None
    sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE
    duration_s: Optional[float] = None
    channels: Optional[int] = 1
    format: str = "wav"
    native_sample_rate: Optional[int] = None
    history: tuple[str, ...] = ()

    def __repr__(self) -> str:
        duration = (
            f"{self.duration_s:.2f}s" if self.duration_s is not None else "None"
        )
        history_repr = f", history={list(self.history)!r}" if self.history else ""
        return (
            f"Audio(source_id={self.source_id!r}, title={self.title!r}, "
            f"path={str(self.path)!r}, sample_rate={self.sample_rate}, "
            f"native_sample_rate={self.native_sample_rate}, "
            f"duration_s={duration}, "
            f"channels={self.channels}, format={self.format!r}"
            f"{history_repr})"
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        source_id: Optional[str] = None,
        title: Optional[str] = None,
        native_sample_rate: Optional[int] = None,
        history: Optional[tuple[str, ...] | list[str]] = None,
    ) -> Audio:
        """Load an audio file from disk and return an ``Audio`` instance.

        For WAV files, sample rate, duration, and channel count are probed from
        the file. Other formats keep the class defaults for those fields.

        Args:
            path: Path to an existing audio file.
            source_id: Optional identifier; defaults to the file stem.
            title: Optional display title; defaults to the file stem.
            native_sample_rate: Original capture/source rate before pipeline
                resampling. Defaults to the probed file rate.
            history: Optional list or tuple of step fingerprint strings.

        Returns:
            Audio: Instance pointing at the resolved file path.

        Raises:
            FileNotFoundError: If ``path`` does not exist or is not a file.
        """
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        fmt = file_path.suffix.lstrip(".").lower() or "wav"
        sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE
        duration_s: Optional[float] = None
        channels: Optional[int] = 1

        if fmt == "wav":
            try:
                sample_rate, duration_s, channels = _probe_wav(file_path)
            except wave.Error:
                pass

        return cls(
            path=file_path,
            source_id=source_id if source_id is not None else file_path.stem,
            title=title if title is not None else file_path.stem,
            sample_rate=sample_rate,
            duration_s=duration_s,
            channels=channels,
            format=fmt,
            native_sample_rate=(
                native_sample_rate if native_sample_rate is not None else sample_rate
            ),
            history=tuple(history) if history is not None else (),
        )

    def metadata(self, *, target_sample_rate: Optional[int] = None) -> dict[str, Any]:
        """Return a serializable snapshot of identity and rate metadata.

        When ``target_sample_rate`` is given, include ``resample_action`` so a
        downstream model can decide whether to upscale, downscale, or keep the
        current file rate.
        """
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["history"] = list(self.history)
        if target_sample_rate is not None:
            payload["target_sample_rate"] = target_sample_rate
            payload["resample_action"] = self.resample_action(target_sample_rate)
        return payload

    def resample_action(self, target_sample_rate: int) -> ResampleAction:
        """Return how this file's rate compares to a model's expected rate."""
        if target_sample_rate <= 0:
            raise ValueError(
                f"target_sample_rate must be positive, got {target_sample_rate}"
            )
        if self.sample_rate is None:
            raise ValueError("sample_rate is unknown; cannot choose a resample action")
        if target_sample_rate > self.sample_rate:
            return "upscale"
        if target_sample_rate < self.sample_rate:
            return "downscale"
        return "keep"

    def add_step(self, step_tag: str) -> Audio:
        """Return a new ``Audio`` instance with ``step_tag`` appended to its history.

        Args:
            step_tag: Short descriptor of the transformation (e.g. ``'htdemucs_vocals'``).

        Returns:
            Audio: A new ``Audio`` instance with updated history.
        """
        clean_tag = _sanitize_filename_component(step_tag)
        if not clean_tag:
            return self
        return replace(self, history=(*self.history, clean_tag))

    @property
    def fingerprint(self) -> str:
        """Return a formatted string representing the audio's processing fingerprint.

        Combines the source identifier, step history, and audio specs (sample rate,
        channels) using double underscores ``__`` as segment separators.
        """
        parts: list[str] = []
        base_id = _sanitize_filename_component(self.source_id or self.path.stem or "audio")
        parts.append(base_id or "audio")

        if self.history:
            parts.extend(self.history)

        specs: list[str] = []
        if self.duration_s is not None:
            specs.append(f"{self.duration_s:.2f}s")
        if self.sample_rate is not None:
            specs.append(f"{self.sample_rate}Hz")
        if self.channels is not None:
            specs.append(f"{self.channels}ch")
        if specs:
            parts.append("_".join(specs))

        return "__".join(p for p in parts if p)

    def with_file(
        self,
        path: str | Path,
        *,
        sample_rate: Optional[int],
        duration_s: Optional[float],
        channels: Optional[int],
        format: str = "wav",
        source_id: Optional[str] = None,
        title: Optional[str] = None,
        step: Optional[str] = None,
        history: Optional[tuple[str, ...] | list[str]] = None,
    ) -> Audio:
        """Return a new ``Audio`` at ``path``, keeping ``native_sample_rate`` and history."""
        if history is not None:
            new_history = tuple(history)
        else:
            new_history = self.history

        if step is not None:
            clean_step = _sanitize_filename_component(step)
            if clean_step:
                new_history = (*new_history, clean_step)

        return replace(
            self,
            path=Path(path).resolve(),
            sample_rate=sample_rate,
            duration_s=duration_s,
            channels=channels,
            format=format,
            source_id=self.source_id if source_id is None else source_id,
            title=self.title if title is None else title,
            history=new_history,
        )

    def show_mel_spectrogram(
        self,
        *,
        n_mels: int = 128,
        hop_length: int = 512,
        fmax: Optional[float] = None,
        title: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Print (display) the mel-spectrogram of the audio file.

        ``librosa`` is imported lazily so the dependency is only required when
        this method is actually called.

        Args:
            n_mels: Number of mel bands.
            hop_length: Number of samples between successive frames.
            fmax: Highest frequency (in Hz) for the mel bands; defaults to
                ``sr / 2``.
            title: Optional title for the plot; defaults to the source title.
            show: Whether to call ``pyplot.show()`` to render the figure.
        """
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt

        if not Path(self.path).exists():
            raise FileNotFoundError(f"Audio file does not exist: {self.path}")

        y, sr = librosa.load(str(self.path), sr=self.sample_rate, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax
        )
        mel_db = librosa.power_to_db(mel, ref=mel.max())

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(title or self.title or self.source_id)
        plt.tight_layout()
        if show:
            plt.show()

    def notebook_display(
        self,
        dest: Optional[str | Path] = None,
    ) -> None:
        """Display this audio file in an interactive Jupyter / IPython player.

        Optionally saves (copies) the audio to ``dest`` before playback.

        Args:
            dest: Optional destination file or directory path. When provided, the
                audio is copied there via ``save_to`` before playback.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        from IPython.display import Audio as IPythonAudio
        from IPython.display import display

        if dest is not None:
            self.save_to(dest)

        path = Path(self.path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        # display() only — do not return the player, or Jupyter renders it twice
        display(IPythonAudio(filename=str(path), rate=self.sample_rate))

    display = notebook_display

    def save_to(self, dest: str | Path) -> Audio:
        """Save (copy) the audio file to a destination file path or directory.

        Args:
            dest: Target file path or directory path.

        Returns:
            Audio: This Audio instance with path updated to the destination file location.
        """
        dest_path = Path(dest)
        src_path = Path(self.path)

        if not src_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {src_path}")

        if dest_path.is_dir() or str(dest).endswith(("/", "\\")):
            dest_path.mkdir(parents=True, exist_ok=True)
            target = dest_path / src_path.name
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            target = dest_path

        if src_path.resolve() != target.resolve():
            shutil.copy2(src_path, target)

        self.path = target.resolve()
        return self

    def quick_save(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Audio:
        """Quickly save (copy) the audio file to a temp directory with fingerprint in the filename.

        By default, saves into ``<project_root>/temp/`` and generates a filename
        incorporating the step fingerprint (``source_id``, history steps, duration,
        sample rate, channels) so it can be immediately recognized. Prints the
        destination path.

        Args:
            output_dir: Target directory. Defaults to ``<project_root>/temp``.
            name: Optional explicit filename to override metadata/fingerprint-based naming.
            prefix: Optional prefix prepended to the generated filename.
            suffix: Optional suffix appended to the generated filename before extension.
            tag: Optional tag appended to the fingerprint components.

        Returns:
            Audio: This Audio instance with path updated to the destination file location.

        Raises:
            FileNotFoundError: If the source audio file does not exist.
        """
        src_path = Path(self.path)
        if not src_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {src_path}")

        fmt = (self.format or src_path.suffix.lstrip(".") or "wav").lower()

        if name is not None:
            filename = name if "." in name else f"{name}.{fmt}"
        else:
            parts: list[str] = []
            if prefix:
                parts.append(_sanitize_filename_component(str(prefix)))

            parts.append(self.fingerprint)

            if tag:
                parts.append(_sanitize_filename_component(str(tag)))
            if suffix:
                parts.append(_sanitize_filename_component(str(suffix)))

            filename = f"{'__'.join(p for p in parts if p)}.{fmt}"

            # If filename is excessively long (> 200 chars), add a short hash and truncate safely
            if len(filename) > 200:
                import hashlib

                h = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8]
                ext_len = len(fmt) + 1
                cutoff = 190 - ext_len - 10
                filename = f"{filename[:cutoff]}__h{h}.{fmt}"

        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(__file__).resolve().parents[2] / "temp"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = (target_dir / filename).resolve()

        if src_path.resolve() != target_file:
            shutil.copy2(src_path, target_file)

        self.path = target_file
        print(f"Quick saved to: {self.path}")
        return self

