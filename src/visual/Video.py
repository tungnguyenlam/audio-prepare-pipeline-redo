"""File-backed video identity, parallel to ``Audio``."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from src.data_paths import DATA_DIR
from src.utils.AudioClass import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_SAMPLE_RATE,
    Audio,
    _sanitize_filename_component,
    _write_sidecar,
)


class VideoError(RuntimeError):
    """Raised when a video file cannot be probed or converted."""


_SIDECAR_KIND = "video.sidecar"
_SIDECAR_FIELDS = (
    "source_id",
    "title",
    "source_url",
    "channel_id",
    "channel_name",
    "channel_url",
    "duration_s",
    "fps",
    "width",
    "height",
    "format",
)


def _sidecar_path(video_path: Path) -> Path:
    return Path(video_path).with_suffix(".json")


def _write_video_sidecar(video: Video) -> None:
    payload: dict[str, Any] = {"kind": _SIDECAR_KIND}
    for field in _SIDECAR_FIELDS:
        payload[field] = getattr(video, field)
    _sidecar_path(video.path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_video_sidecar(video_path: Path) -> dict[str, Any] | None:
    path = _sidecar_path(video_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("kind") != _SIDECAR_KIND:
        return None
    return data


def _parse_frame_rate(value: Any) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return None
        return rate if rate > 0 else None


def probe_video(path: Path, *, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Return duration, fps, and frame size for a video file.

    Args:
        path: Video file to probe.
        ffprobe_bin: ``ffprobe`` executable.

    Returns:
        Dict with ``duration_s``, ``fps``, ``width``, ``height``, and ``format``.

    Raises:
        VideoError: If ffprobe fails or the file has no video stream.
    """
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,codec_name",
        "-show_entries",
        "format=duration,format_name",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise VideoError(
            f"ffprobe failed for {path}: {detail[:1000] or 'No error output'}"
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoError(f"ffprobe returned invalid JSON for {path}") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise VideoError(f"No video stream found in {path}")
    stream = streams[0]
    fmt = payload.get("format") or {}
    duration = stream.get("duration") or fmt.get("duration")
    fps = _parse_frame_rate(stream.get("avg_frame_rate")) or _parse_frame_rate(
        stream.get("r_frame_rate")
    )
    format_name = path.suffix.lstrip(".").lower() or str(
        (fmt.get("format_name") or "mp4").split(",")[0]
    )
    return {
        "duration_s": float(duration) if duration not in (None, "N/A") else None,
        "fps": fps,
        "width": int(stream["width"]) if stream.get("width") else None,
        "height": int(stream["height"]) if stream.get("height") else None,
        "format": format_name,
    }


@dataclass
class Video:
    """Represents a file-backed video source used for visual identity."""

    path: Path
    source_id: str
    title: Optional[str] = None
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: str = "mp4"
    source_url: Optional[str] = None
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    channel_url: Optional[str] = None

    def metadata(self) -> dict[str, Any]:
        """Return a serializable snapshot of identity and media fields."""
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        source_id: str | None = None,
        title: str | None = None,
        source_url: str | None = None,
        channel_id: str | None = None,
        channel_name: str | None = None,
        channel_url: str | None = None,
        ffprobe_bin: str = "ffprobe",
    ) -> Video:
        """Create a ``Video`` object for an existing file.

        If ``{stem}.json`` exists next to the video, identity fields are restored
        from it. Explicit keyword arguments override the sidecar. Probed media
        properties win over sidecar snapshots.

        Args:
            path: Video file path.
            source_id: Optional stable source identity. Defaults to the file stem
                or sidecar value.
            title: Optional display title.
            source_url: Optional originating URL.
            channel_id: Optional channel identity.
            channel_name: Optional channel display name.
            channel_url: Optional canonical channel URL.
            ffprobe_bin: ``ffprobe`` executable used to read media properties.

        Returns:
            A file-backed ``Video``.

        Raises:
            FileNotFoundError: If ``path`` is not a file.
            VideoError: If the file cannot be probed.
        """
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {file_path}")

        sidecar = _read_video_sidecar(file_path)
        probed = probe_video(file_path, ffprobe_bin=ffprobe_bin)

        if source_id is not None:
            resolved_source_id = source_id
        elif sidecar is not None and sidecar.get("source_id"):
            resolved_source_id = str(sidecar["source_id"])
        else:
            resolved_source_id = file_path.stem

        if title is not None:
            resolved_title = title
        elif sidecar is not None and sidecar.get("title") is not None:
            resolved_title = str(sidecar["title"])
        else:
            resolved_title = file_path.stem

        def resolve_optional(explicit: Optional[str], field: str) -> Optional[str]:
            if explicit is not None:
                return explicit
            if sidecar is not None and sidecar.get(field) is not None:
                return str(sidecar[field])
            return None

        video = cls(
            path=file_path,
            source_id=resolved_source_id,
            title=resolved_title,
            duration_s=probed["duration_s"],
            fps=probed["fps"],
            width=probed["width"],
            height=probed["height"],
            format=probed["format"] or file_path.suffix.lstrip(".").lower() or "mp4",
            source_url=resolve_optional(source_url, "source_url"),
            channel_id=resolve_optional(channel_id, "channel_id"),
            channel_name=resolve_optional(channel_name, "channel_name"),
            channel_url=resolve_optional(channel_url, "channel_url"),
        )
        return video

    def write_sidecar(self) -> Video:
        """Write identity metadata next to ``self.path`` as ``{stem}.json``."""
        _write_video_sidecar(self)
        return self

    def extract_audio(
        self,
        dest: str | Path | None = None,
        *,
        sample_rate: int | None = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        audio_format: str = DEFAULT_AUDIO_FORMAT,
        ffmpeg_bin: str | None = None,
    ) -> Audio:
        """Extract a normalized audio file from this video.

        Args:
            dest: Optional destination audio path. Defaults to
                ``.data/visual/audio/<source_id>.<format>``.
            sample_rate: Target rate. ``None`` keeps the source rate.
            channels: Target channel count.
            audio_format: Output container/codec format (default WAV).
            ffmpeg_bin: ``ffmpeg`` executable.

        Returns:
            A new file-backed ``Audio`` sharing this video's identity.

        Raises:
            FileNotFoundError: If the video file is missing.
            VideoError: If ffmpeg fails.
        """
        src_path = Path(self.path)
        if not src_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {src_path}")

        if dest is None:
            output_dir = DATA_DIR / "visual" / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = _sanitize_filename_component(self.source_id) or "video"
            dest_path = output_dir / f"{stem}.{audio_format}"
        else:
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src_path),
            "-vn",
            "-ac",
            str(channels),
        ]
        if sample_rate is not None:
            cmd.extend(["-ar", str(sample_rate)])
        if audio_format.lower() == "wav":
            cmd.extend(["-c:a", "pcm_s16le"])
        cmd.append(str(dest_path))
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise VideoError(
                f"ffmpeg audio extraction failed (exit code {completed.returncode}): "
                f"{detail[:1000] or 'No error output'}"
            )

        duration_s = self.duration_s
        probed_rate = sample_rate
        probed_channels = channels
        if dest_path.suffix.lower() == ".wav" and dest_path.is_file():
            try:
                import wave

                with wave.open(str(dest_path), "rb") as wf:
                    probed_rate = wf.getframerate()
                    probed_channels = wf.getnchannels()
                    frames = wf.getnframes()
                if probed_rate:
                    duration_s = frames / float(probed_rate)
            except Exception:
                pass

        audio = Audio(
            path=dest_path.resolve(),
            source_id=self.source_id,
            title=self.title,
            source_url=self.source_url,
            channel_id=self.channel_id,
            channel_name=self.channel_name,
            channel_url=self.channel_url,
            sample_rate=probed_rate,
            duration_s=duration_s,
            channels=probed_channels,
            format=audio_format.lower(),
            native_sample_rate=probed_rate,
        )
        _write_sidecar(audio)
        return audio
