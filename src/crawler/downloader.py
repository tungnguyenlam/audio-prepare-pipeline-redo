import re
import os
import json
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.crawler.storage import (
    STORAGE_DIR,
    format_duration,
    format_filesize,
    save_audio_record,
    init_storage,
    PROJECT_ROOT
)
from src.yt_crawler.YtCrawlerClass import YtCrawler, DownloadError

TEMP_DIR = PROJECT_ROOT / "temp"


def _download_and_convert_sync(
    url: str,
    sample_rate: int = 44100,
    mono: bool = False,
    cookies_from_browser: Optional[str] = None
) -> Dict[str, Any]:
    """Download YouTube audio using robust YtCrawler from main branch."""
    init_storage()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    channels = 1 if mono else 2
    
    crawler = YtCrawler(
        output_dir=STORAGE_DIR,
        work_dir=TEMP_DIR,
        audio_format="wav",
        sample_rate=sample_rate,
        channels=channels,
        cookies_from_browser=cookies_from_browser,
    )
    
    try:
        audio_obj = crawler.download(url)
    except DownloadError as de:
        raise RuntimeError(str(de))
    except Exception as e:
        raise RuntimeError(f"Lỗi khi crawl video: {str(e)}")

    # Extract metadata from the sidecar JSON if available
    sidecar_path = Path(audio_obj.path).with_suffix(".json")
    sidecar_data = {}
    if sidecar_path.exists():
        try:
            sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    final_file = Path(audio_obj.path)
    filesize = final_file.stat().st_size if final_file.exists() else 0
    duration = audio_obj.duration_s or 0.0

    # Read info.json if preserved or construct thumbnail
    video_id = audio_obj.source_id or "audio"
    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if len(video_id) == 11 else ""

    record = {
        "id": video_id,
        "filename": final_file.name,
        "title": audio_obj.title or final_file.stem,
        "url": url,
        "uploader": sidecar_data.get("uploader", "YouTube"),
        "thumbnail": thumbnail_url,
        "duration": duration,
        "duration_formatted": format_duration(duration),
        "filesize": filesize,
        "filesize_formatted": format_filesize(filesize),
        "sample_rate": audio_obj.sample_rate,
        "channels": "mono" if audio_obj.channels == 1 else "stereo",
        "format": "wav",
        "created_at": datetime.now().isoformat(),
    }
    
    return record


async def crawl_youtube_audio(
    url: str,
    sample_rate: int = 44100,
    mono: bool = False,
    cookies_from_browser: Optional[str] = None
) -> Dict[str, Any]:
    """Asynchronous wrapper for downloading and saving YouTube audio."""
    loop = asyncio.get_running_loop()
    record = await loop.run_in_executor(
        None,
        _download_and_convert_sync,
        url,
        sample_rate,
        mono,
        cookies_from_browser
    )
    # Save to metadata storage
    saved_record = await save_audio_record(record)
    return saved_record
