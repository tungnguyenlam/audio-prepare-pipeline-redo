"""Cut file-backed ``Audio`` objects into smaller segments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import soundfile as sf

from src.utils.AudioClass import Audio, _sanitize_filename_component

TimeUnit = Literal["seconds", "minutes", "hours", "percent", "timestamp"]
TimeValue = Union[int, float, str]


class AudioCutterError(ValueError):
    """Raised when cut bounds are invalid or the source cannot be read."""


class AudioCutter:
    """Extract a contiguous slice from an ``Audio`` file.

    Bounds may be percentages (``0``–``100``) or time values. Time values
    support several units:

    - ``seconds`` / ``minutes`` / ``hours``: magnitude only (seconds may be
      decimal, e.g. ``12.5``).
    - ``timestamp``: packed digits or ``H:MM:SS`` / ``M:SS`` strings.

      Packed-digit rules (parsed right-to-left as ``…HH MM SS``):

      - 4 digits → ``MM:SS`` (``1234`` → ``12:34``)
      - 5 digits → ``H:MM:SS`` (``12345`` → ``1:23:45``)
      - 6+ digits → ``H+:MM:SS`` (``1213421`` → ``121:34:21``)
      - 1–2 digits → seconds; 3 digits → ``M:SS``

    Example::

        cutter = AudioCutter(output_dir=".data/audio_cutter/out")
        clip = cutter.cut(audio, 10.5, 45.0)  # seconds
        clip = cutter.cut(audio, 1, 3, unit="minutes")
        clip = cutter.cut(audio, 0, 25, unit="percent")
        clip = cutter.cut(audio, 1234, 1213421, unit="timestamp")  # 12:34 → 121:34:21
        clip = cutter.cut(audio, "1:02", "2:15:30", unit="timestamp")
    """

    def __init__(
        self,
        output_dir: str | Path = ".data/audio_cutter/out",
    ) -> None:
        self.output_dir = Path(output_dir)

    def cut(
        self,
        audio: Audio,
        start: TimeValue,
        end: TimeValue,
        *,
        unit: TimeUnit = "seconds",
        output_path: Optional[str | Path] = None,
    ) -> Audio:
        """Write ``[start, end)`` of ``audio`` to a new file and return it.

        Args:
            audio: Source ``Audio`` instance (file-backed).
            start: Inclusive start bound.
            end: Exclusive end bound.
            unit: How to interpret ``start`` / ``end``. One of ``seconds``,
                ``minutes``, ``hours``, ``percent``, or ``timestamp``.
            output_path: Optional destination file. When omitted, a file is
                written under ``output_dir``.

        Returns:
            A new ``Audio`` pointing at the cut segment.
        """
        src_path = Path(audio.path)
        if not src_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {src_path}")

        waveform, sample_rate = sf.read(str(src_path), always_2d=True)
        if waveform.shape[0] == 0:
            raise AudioCutterError(f"Audio source is empty: {src_path}")

        duration_s = waveform.shape[0] / float(sample_rate)
        start_s, end_s = self._resolve_bounds(start, end, duration_s, unit)

        start_frame = int(round(start_s * sample_rate))
        end_frame = int(round(end_s * sample_rate))
        start_frame = max(0, min(start_frame, waveform.shape[0]))
        end_frame = max(start_frame, min(end_frame, waveform.shape[0]))
        if end_frame <= start_frame:
            raise AudioCutterError(
                f"resolved empty cut: start={start_s:.6f}s end={end_s:.6f}s "
                f"(duration={duration_s:.6f}s)"
            )

        segment = waveform[start_frame:end_frame]
        dest = self._resolve_output_path(
            audio=audio,
            start_s=start_s,
            end_s=end_s,
            output_path=output_path,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), segment, sample_rate)

        return Audio.from_file(
            dest,
            source_id=f"{audio.source_id}_{start_s:.3f}-{end_s:.3f}",
            title=f"{audio.title or audio.source_id} [{start_s:.3f}s-{end_s:.3f}s]",
            source_url=audio.source_url,
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
            native_sample_rate=audio.native_sample_rate,
            history=(*audio.history, f"cut_{start_s:.2f}s-{end_s:.2f}s"),
        )

    def _resolve_bounds(
        self,
        start: TimeValue,
        end: TimeValue,
        duration_s: float,
        unit: TimeUnit,
    ) -> tuple[float, float]:
        if unit == "percent":
            start_s = self._percent_to_seconds(start, duration_s, "start")
            end_s = self._percent_to_seconds(end, duration_s, "end")
        elif unit in ("seconds", "minutes", "hours", "timestamp"):
            start_s = self._to_seconds(start, unit)
            end_s = self._to_seconds(end, unit)
        else:
            raise AudioCutterError(
                "unit must be one of 'seconds', 'minutes', 'hours', "
                f"'percent', 'timestamp'; got {unit!r}"
            )

        if start_s < 0 or end_s < 0:
            raise AudioCutterError(
                f"bounds must be non-negative (got start={start_s}, end={end_s})"
            )
        if start_s >= end_s:
            raise AudioCutterError(
                f"start must be < end (got start={start_s}, end={end_s})"
            )
        if start_s >= duration_s:
            raise AudioCutterError(
                f"start ({start_s:.6f}s) is past audio duration ({duration_s:.6f}s)"
            )

        return start_s, min(end_s, duration_s)

    def _percent_to_seconds(
        self,
        value: TimeValue,
        duration_s: float,
        name: str,
    ) -> float:
        try:
            pct = float(value)
        except (TypeError, ValueError) as exc:
            raise AudioCutterError(
                f"{name} percent bound must be numeric, got {value!r}"
            ) from exc
        if not 0.0 <= pct <= 100.0:
            raise AudioCutterError(
                f"{name} percent bound must be in [0, 100] (got {value!r})"
            )
        return duration_s * (pct / 100.0)

    def _to_seconds(self, value: TimeValue, unit: TimeUnit) -> float:
        if unit == "timestamp":
            return self.parse_timestamp(value)
        try:
            magnitude = float(value)
        except (TypeError, ValueError) as exc:
            raise AudioCutterError(
                f"{unit} bound must be numeric, got {value!r}"
            ) from exc
        if magnitude < 0:
            raise AudioCutterError(f"{unit} bound must be non-negative, got {value!r}")
        if unit == "seconds":
            return magnitude
        if unit == "minutes":
            return magnitude * 60.0
        if unit == "hours":
            return magnitude * 3600.0
        raise AudioCutterError(f"unsupported time unit: {unit!r}")

    @staticmethod
    def parse_timestamp(value: TimeValue) -> float:
        """Convert a timestamp value to seconds.

        Accepts:

        - colon form: ``"MM:SS"``, ``"HH:MM:SS"`` (seconds may be decimal)
        - packed digits: ``1234`` → ``12:34``, ``1213421`` → ``121:34:21``
        """
        if isinstance(value, bool):
            raise AudioCutterError(f"invalid timestamp: {value!r}")

        if isinstance(value, float):
            if not value.is_integer():
                raise AudioCutterError(
                    "timestamp floats must be whole packed digits "
                    f"(got {value!r}); use unit='seconds' for decimal seconds"
                )
            value = int(value)

        if isinstance(value, int):
            if value < 0:
                raise AudioCutterError(f"timestamp must be non-negative, got {value!r}")
            return AudioCutter._parse_packed_digits(str(value))

        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise AudioCutterError("timestamp string is empty")
            if ":" in text:
                return AudioCutter._parse_colon_timestamp(text)
            if text.isdigit():
                return AudioCutter._parse_packed_digits(text)
            raise AudioCutterError(
                f"timestamp string must be packed digits or H:MM:SS, got {value!r}"
            )

        raise AudioCutterError(f"unsupported timestamp type: {type(value).__name__}")

    @staticmethod
    def _parse_colon_timestamp(text: str) -> float:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise AudioCutterError(
                f"colon timestamp must be MM:SS or HH:MM:SS, got {text!r}"
            )
        try:
            numbers = [float(part) for part in parts]
        except ValueError as exc:
            raise AudioCutterError(f"invalid colon timestamp: {text!r}") from exc

        if len(numbers) == 2:
            minutes, seconds = numbers
            hours = 0.0
        else:
            hours, minutes, seconds = numbers

        if hours < 0 or minutes < 0 or seconds < 0:
            raise AudioCutterError(f"timestamp fields must be non-negative: {text!r}")
        if minutes >= 60 or seconds >= 60:
            raise AudioCutterError(
                f"minutes/seconds must be < 60 in colon timestamp: {text!r}"
            )
        return hours * 3600.0 + minutes * 60.0 + seconds

    @staticmethod
    def _parse_packed_digits(digits: str) -> float:
        if not digits.isdigit():
            raise AudioCutterError(f"packed timestamp must be digits, got {digits!r}")
        if digits != "0" and digits.startswith("0") and len(digits) > 1:
            # Keep leading zeros meaningful only for fixed-width forms like "0123".
            pass

        n = len(digits)
        if n <= 2:
            hours, minutes, seconds = 0, 0, int(digits)
        elif n == 3:
            hours, minutes, seconds = 0, int(digits[0]), int(digits[1:])
        elif n == 4:
            hours, minutes, seconds = 0, int(digits[:2]), int(digits[2:])
        elif n == 5:
            hours, minutes, seconds = int(digits[0]), int(digits[1:3]), int(digits[3:])
        else:
            # 6+: …HHMMSS with hours taking all remaining leading digits.
            seconds = int(digits[-2:])
            minutes = int(digits[-4:-2])
            hours = int(digits[:-4])

        if minutes >= 60 or seconds >= 60:
            raise AudioCutterError(
                "packed timestamp minutes/seconds must be < 60 "
                f"(got {digits!r} → {hours}:{minutes:02d}:{seconds:02d})"
            )
        return float(hours * 3600 + minutes * 60 + seconds)

    def _resolve_output_path(
        self,
        *,
        audio: Audio,
        start_s: float,
        end_s: float,
        output_path: Optional[str | Path],
    ) -> Path:
        sanitized_title = _sanitize_filename_component(
            audio.title or audio.source_id or audio.path.stem
        )
        if len(sanitized_title) > 100:
            sanitized_title = sanitized_title[:100].rstrip("._")
        file_stem = sanitized_title or "audio"
        filename = (
            f"{file_stem}_{start_s:.3f}-{end_s:.3f}."
            f"{audio.format or 'wav'}"
        )

        if output_path is not None:
            path = Path(output_path)
            if path.suffix:
                return path
            path.mkdir(parents=True, exist_ok=True)
            return path / filename

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / filename
