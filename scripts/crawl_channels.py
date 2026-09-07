#!/usr/bin/env python3
"""Crawl audio from YouTube channels for pipeline benchmarking and fine-tuning dataset generation.

Uses YtCrawler from src.yt_crawler.YtCrawlerClass to download and normalize
audio to 16 kHz mono WAV.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_paths import DATA_DIR
from src.utils.AudioClass import Audio
from src.yt_crawler.YtCrawlerClass import YtCrawler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("channel_crawler")

DEFAULT_CHANNELS = [
    "https://www.youtube.com/@TRANTHANHTOWN",
    "https://www.youtube.com/@KhánhVyOFFICIAL",
]


def sanitize_channel_name(url_or_handle: str) -> str:
    """Extract a clean folder name from a channel URL or handle."""
    clean = url_or_handle.strip().rstrip("/")
    if "/@" in clean:
        clean = clean.split("/@")[-1]
    elif "@" in clean:
        clean = clean.split("@")[-1]
    clean = re.sub(r"[^\w\-_]", "_", clean).lower()
    return clean or "unknown_channel"


def list_channel_videos(
    channel_url: str,
    max_candidates: int = 15,
) -> list[dict[str, Any]]:
    """Query recent videos from a channel using yt-dlp flat-playlist extraction."""
    clean_url = channel_url.rstrip("/")
    if not clean_url.endswith("/videos"):
        videos_url = f"{clean_url}/videos"
    else:
        videos_url = clean_url

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--print",
        "%(id)s\t%(duration)s\t%(title)s",
        "--playlist-end",
        str(max_candidates),
        videos_url,
    ]
    logger.info("Listing recent videos for %s...", channel_url)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Failed to list videos from %s: %s", channel_url, res.stderr)
        return []

    videos = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            v_id = parts[0].strip()
            try:
                v_dur = float(parts[1].strip())
            except (ValueError, TypeError):
                v_dur = 0.0
            v_title = parts[2].strip()
            videos.append({"id": v_id, "duration": v_dur, "title": v_title})
    return videos


def crawl_channel(
    channel_url: str,
    max_videos: int = 2,
    min_dur: float = 180.0,
    max_dur: float = 1800.0,
    sample_rate: int = 16000,
    out_base_dir: Path = DATA_DIR / "crawled",
) -> list[dict[str, Any]]:
    """Crawl selected videos from a channel."""
    chan_slug = sanitize_channel_name(channel_url)
    target_dir = out_base_dir / chan_slug
    work_dir = target_dir / "work"
    target_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    crawler = YtCrawler(
        output_dir=target_dir,
        work_dir=work_dir,
        audio_format="wav",
        sample_rate=sample_rate,
        channels=1,
    )

    all_vids = list_channel_videos(channel_url, max_candidates=20)
    # Filter by duration
    suitable = [
        v for v in all_vids
        if min_dur <= v["duration"] <= max_dur
    ]
    logger.info("Found %d suitable videos (between %.0fs and %.0fs) for %s", len(suitable), min_dur, max_dur, chan_slug)

    selected = suitable[:max_videos]
    crawled_manifest = []

    for v in selected:
        vid_id = v["id"]
        vid_title = v["title"]
        vid_url = f"https://www.youtube.com/watch?v={vid_id}"
        expected_wav = target_dir / f"{vid_id}.wav"

        logger.info("--- Downloading: %s (%s, %.1fs) ---", vid_title, vid_id, v["duration"])
        try:
            if expected_wav.is_file():
                logger.info("Audio already exists: %s", expected_wav)
                audio = Audio.from_file(expected_wav)
            else:
                audio = crawler.download(vid_url)

            entry = {
                "id": audio.source_id or vid_id,
                "title": audio.title or vid_title,
                "url": vid_url,
                "channel": chan_slug,
                "path": str(audio.path.resolve()),
                "duration_s": audio.duration_s,
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
            }
            crawled_manifest.append(entry)
            logger.info("Successfully ingested: %s (%.2fs, %dHz)", audio.title, audio.duration_s, audio.sample_rate)
        except Exception as exc:
            logger.error("Failed to download video %s: %s", vid_url, exc)

    return crawled_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl audio from challenging YouTube channels.")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="List of YouTube channel URLs or handles.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=2,
        help="Maximum videos to download per channel.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=180.0,
        help="Minimum duration in seconds to accept.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=1800.0,
        help="Maximum duration in seconds to accept.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output sample rate in Hz (default 16000).",
    )
    args = parser.parse_args()

    out_base = DATA_DIR / "crawled"
    out_base.mkdir(parents=True, exist_ok=True)
    manifest_path = out_base / "crawled_manifest.json"

    existing_manifest = []
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except Exception:
            existing_manifest = []

    all_crawled = list(existing_manifest)
    known_ids = {item["id"] for item in all_crawled}

    for chan_url in args.channels:
        logger.info("=== Processing Channel: %s ===", chan_url)
        items = crawl_channel(
            chan_url,
            max_videos=args.max_videos,
            min_dur=args.min_duration,
            max_dur=args.max_duration,
            sample_rate=args.sample_rate,
            out_base_dir=out_base,
        )
        for it in items:
            if it["id"] not in known_ids:
                all_crawled.append(it)
                known_ids.add(it["id"])

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_crawled, f, indent=2, ensure_ascii=False)

    logger.info("=== Crawling Completed ===")
    logger.info("Total crawled videos in manifest: %d", len(all_crawled))
    logger.info("Manifest saved to: %s", manifest_path)


if __name__ == "__main__":
    main()
