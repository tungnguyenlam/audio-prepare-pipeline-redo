"""YouTube Audio Crawler and Extraction Module.

Extracted from audio-prepare-pipeline ingestion logic with robust anti-bot & 403 fallback handling.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
import wave
from pathlib import Path
from typing import Optional, List

from src.utils.AudioClass import (
    DEFAULT_SAMPLE_RATE,
    Audio,
    _sanitize_filename_component,
    _write_sidecar,
)

logger = logging.getLogger(__name__)

# Extensions that should never be saved as final audio artifacts
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v"}


class DownloadError(RuntimeError):
    """Raised when yt-dlp or ffmpeg fails during download or post-processing."""


def probe_wav(path: Path) -> tuple[int, float, int]:
    """Extract sample rate, duration (seconds), and channels count from a WAV file."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.getnframes()
        channels = wf.getnchannels()
    duration = frames / float(rate) if rate else 0.0
    return rate, duration, channels


class YtCrawler:
    """Download and normalize audio from YouTube links into standardized WAV."""

    def __init__(
        self,
        output_dir: str | Path = "audio_crawl",
        work_dir: str | Path = "temp",
        audio_format: str = "wav",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 2,
        retries: int = 3,
        yt_dlp_bin: Optional[str] = None,
        ffmpeg_bin: Optional[str] = None,
        cookies_file: Optional[str] = None,
        cookies_from_browser: Optional[str] = None,
        proxy: Optional[str] = None,
        extractor_args: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.work_dir = Path(work_dir)
        self.audio_format = audio_format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.retries = retries
        self.yt_dlp_bin = yt_dlp_bin
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        self.cookies_file = cookies_file
        self.cookies_from_browser = cookies_from_browser
        self.proxy = proxy
        self.extractor_args = extractor_args

    @classmethod
    def ingest(
        cls,
        link: str,
        output_dir: str | Path = "audio_crawl",
        work_dir: str | Path = "temp",
        audio_format: str = "wav",
        sample_rate: int = 44100,
        channels: int = 2,
        **kwargs,
    ) -> Audio:
        """Class method to directly ingest a YouTube link and return an Audio instance."""
        crawler = cls(
            output_dir=output_dir,
            work_dir=work_dir,
            audio_format=audio_format,
            sample_rate=sample_rate,
            channels=channels,
            **kwargs,
        )
        return crawler.download(link)

    def _yt_dlp_prefix(self) -> list[str]:
        """Determine binary prefix for yt-dlp invocation."""
        if self.yt_dlp_bin:
            return [self.yt_dlp_bin]
        return [sys.executable, "-m", "yt_dlp"]

    def build_command(
        self, 
        url: str, 
        target_work_dir: Path,
        override_extractor_args: Optional[str] = None,
        use_browser_cookies: Optional[str] = None
    ) -> list[str]:
        """Construct the yt-dlp command-line arguments."""
        out_template = str(target_work_dir / "%(id)s.%(ext)s")
        cmd = self._yt_dlp_prefix() + [
            "--no-playlist",
            "--no-simulate",
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            self.audio_format,
            "-o",
            out_template,
            "--write-info-json",
            "--no-write-playlist-metafiles",
            "--retries",
            str(self.retries),
            "--fragment-retries",
            str(self.retries),
            "--postprocessor-args",
            f"ExtractAudio:-ac {self.channels}",
        ]
        
        # Add JS runtimes if node exists
        node_path = shutil.which("node") or "/home/vsf/.nvm/versions/node/v18.20.8/bin/node"
        if os.path.exists(node_path):
            cmd.extend(["--js-runtimes", f"node:{node_path}"])
            
        ext_args = override_extractor_args or self.extractor_args
        if ext_args:
            cmd.extend(["--extractor-args", ext_args])
            
        if self.proxy:
            cmd.extend(["--proxy", self.proxy])
            
        if self.cookies_file and os.path.exists(self.cookies_file):
            cmd.extend(["--cookies", self.cookies_file])
        else:
            browser = use_browser_cookies or self.cookies_from_browser
            if browser:
                cmd.extend(["--cookies-from-browser", browser])
                
        cmd.append(url)
        return cmd

    def download(self, url: str) -> Audio:
        """Download audio from url, convert to target specs with multi-strategy fallback."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        session_dir = self.work_dir / f"job-{uuid.uuid4().hex}"
        session_dir.mkdir(parents=True, exist_ok=False)

        # Multi-strategy retry list to handle 403 / anti-bot restrictions
        strategies = [
            {"extractor_args": "youtube:player_client=android,web", "browser": self.cookies_from_browser},
            {"extractor_args": "youtube:player_client=web,android,ios", "browser": self.cookies_from_browser},
            {"extractor_args": "youtube:player_client=android_vr,web", "browser": self.cookies_from_browser},
        ]
        
        # If firefox or chrome available, add as fallback
        if shutil.which("firefox"):
            strategies.append({"extractor_args": "youtube:player_client=web,android", "browser": "firefox"})

        last_error = ""
        success = False

        try:
            for i, strat in enumerate(strategies):
                cmd = self.build_command(
                    url, 
                    session_dir, 
                    override_extractor_args=strat["extractor_args"],
                    use_browser_cookies=strat["browser"]
                )
                logger.info(f"Running yt-dlp strategy {i+1}: {' '.join(cmd)}")
                
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if completed.returncode == 0:
                    success = True
                    break
                else:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    last_error = detail
                    logger.warning(f"Strategy {i+1} failed: {detail[:200]}")
                    
            if not success:
                raise DownloadError(f"Tải video thất bại: {last_error[:800] or 'Unknown error'}")

            info_paths = sorted(
                session_dir.glob("*.info.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if not info_paths:
                raise DownloadError("yt-dlp hoàn thành nhưng không tìm thấy metadata *.info.json.")

            info_file = info_paths[-1]
            try:
                info = json.loads(info_file.read_text(encoding="utf-8"))
            except Exception as err:
                raise DownloadError(f"Lỗi parse JSON metadata: {err}") from err

            source_id = str(info.get("id") or info_file.name.removesuffix(".info.json"))
            title = info.get("title")
            if title is not None:
                title = str(title)

            audio_candidates = [
                p
                for p in session_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() not in {".json"}
                and source_id in p.stem
                and p.suffix.lower() not in _VIDEO_SUFFIXES
            ]

            if not audio_candidates:
                raise DownloadError(
                    f"yt-dlp hoàn thành nhưng không tìm thấy file audio cho source_id={source_id}"
                )

            preferred = [
                p for p in audio_candidates if p.suffix.lower() == f".{self.audio_format}"
            ]
            src_audio = preferred[0] if preferred else audio_candidates[0]
            native_sample_rate = _native_sample_rate(src_audio, info)

            # Construct output filename: <sanitized_title>__<source_id>.<format>
            sanitized_title = _sanitize_filename_component(title) if title else ""
            if sanitized_title:
                if len(sanitized_title) > 80:
                    sanitized_title = sanitized_title[:80].rstrip("._")
                file_stem = f"{sanitized_title}__{source_id}"
            else:
                file_stem = source_id

            final_dest = self.output_dir / f"{file_stem}.{self.audio_format}"

            self._ensure_standard_audio(src_audio, final_dest)

            duration_s = info.get("duration")
            sample_rate = self.sample_rate
            channels = self.channels

            if final_dest.suffix.lower() == ".wav" and final_dest.exists():
                try:
                    sample_rate, probed_dur, channels = probe_wav(final_dest)
                    if probed_dur > 0:
                        duration_s = probed_dur
                except Exception:
                    pass

            if native_sample_rate is None:
                native_sample_rate = sample_rate

            for path in self.output_dir.glob(f"*{source_id}.*"):
                if path.suffix.lower() in _VIDEO_SUFFIXES:
                    path.unlink(missing_ok=True)

            audio_obj = Audio(
                path=final_dest.resolve(),
                source_id=source_id,
                title=title,
                sample_rate=sample_rate,
                duration_s=float(duration_s) if duration_s is not None else None,
                channels=channels,
                format=self.audio_format,
                native_sample_rate=native_sample_rate,
            )

            # Write standard identity sidecar JSON next to the audio file
            _write_sidecar(audio_obj)

            return audio_obj
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)

    def _ensure_standard_audio(self, src: Path, dest: Path) -> None:
        """Convert audio to required format, sample rate, and channel count using ffmpeg."""
        needs_ffmpeg = True
        if src.suffix.lower() == f".{self.audio_format}" and self.audio_format == "wav":
            try:
                rate, _, ch = probe_wav(src)
                if rate == self.sample_rate and ch == self.channels:
                    needs_ffmpeg = False
            except wave.Error:
                needs_ffmpeg = True

        if not needs_ffmpeg:
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            return

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(src),
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
        ]
        if self.audio_format == "wav":
            cmd.extend(["-c:a", "pcm_s16le"])

        cmd.append(str(dest))

        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise DownloadError(
                f"ffmpeg conversion failed (exit code {completed.returncode}): {detail[:1000] or 'No error output'}"
            )


def _native_sample_rate(src_audio: Path, info: dict) -> Optional[int]:
    """Best-effort original rate: pre-normalize WAV, else yt-dlp asr."""
    if src_audio.suffix.lower() == ".wav":
        try:
            rate, _, _ = probe_wav(src_audio)
            if rate:
                return rate
        except Exception:
            pass

    asr = info.get("asr")
    if asr:
        return int(asr)

    downloads = info.get("requested_downloads") or []
    if downloads:
        download_asr = downloads[0].get("asr")
        if download_asr:
            return int(download_asr)
    return None
