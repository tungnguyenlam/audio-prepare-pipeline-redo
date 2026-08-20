import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Resolve base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "audio_crawl"
METADATA_FILE = STORAGE_DIR / "metadata.json"
PROCESSED_DIR = PROJECT_ROOT / "processed_audio"

_lock = asyncio.Lock()


def init_storage():
    """Ensure input and derived-audio storage directories exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not METADATA_FILE.exists():
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def format_duration(seconds: float) -> str:
    """Format duration into mm:ss or hh:mm:ss."""
    if not seconds or seconds < 0:
        return "00:00"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_filesize(size_bytes: int) -> str:
    """Format bytes into readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _read_metadata_sync() -> List[Dict[str, Any]]:
    init_storage()
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def _write_metadata_sync(data: List[Dict[str, Any]]) -> None:
    init_storage()
    temp_file = METADATA_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, METADATA_FILE)


async def get_all_audio() -> List[Dict[str, Any]]:
    """Retrieve all audio items with metadata, verifying file existence on disk."""
    async with _lock:
        data = _read_metadata_sync()
        
        # Verify physical files exist and update metadata if needed
        valid_items = []
        modified = False
        for item in data:
            filename = item.get("filename")
            if not filename:
                continue
            file_path = STORAGE_DIR / filename
            if file_path.exists() and file_path.is_file():
                # Update filesize if missing or changed
                actual_size = file_path.stat().st_size
                if item.get("filesize") != actual_size:
                    item["filesize"] = actual_size
                    item["filesize_formatted"] = format_filesize(actual_size)
                    modified = True
                valid_items.append(item)
            else:
                modified = True
                
        if modified:
            _write_metadata_sync(valid_items)
            
        # Return sorted by created_at descending
        return sorted(valid_items, key=lambda x: x.get("created_at", ""), reverse=True)


async def save_audio_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Save or update an audio metadata record."""
    async with _lock:
        data = _read_metadata_sync()
        # Remove any existing record with same filename or id
        data = [
            item for item in data 
            if item.get("filename") != record.get("filename") and item.get("id") != record.get("id")
        ]
        data.insert(0, record)
        _write_metadata_sync(data)
        return record


async def get_audio_record(filename: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific audio filename."""
    async with _lock:
        data = _read_metadata_sync()
        for item in data:
            if item.get("filename") == filename:
                return item
        return None


async def delete_audio_record(filename: str) -> bool:
    """Delete audio file, sidecar JSON, and its metadata record from local disk."""
    async with _lock:
        data = _read_metadata_sync()
        file_path = STORAGE_DIR / Path(filename).name
        stem = file_path.stem
        
        deleted = False
        # Delete the main audio file and sidecar .json files sharing the same stem
        for matched_file in STORAGE_DIR.glob(f"{stem}.*"):
            if matched_file.name == "metadata.json" or matched_file.name == ".gitkeep":
                continue
            try:
                if matched_file.exists() and matched_file.is_file():
                    matched_file.unlink()
                    deleted = True
            except Exception as e:
                print(f"Error deleting file {matched_file}: {e}")

        # If file_path specifically still exists
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                deleted = True
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        # Update metadata list
        new_data = [item for item in data if item.get("filename") != filename and item.get("filename") != file_path.name]
        if len(new_data) != len(data) or deleted:
            _write_metadata_sync(new_data)
            return True
        return False


def get_audio_path(filename: str) -> Optional[Path]:
    """Securely resolve audio file path preventing directory traversal."""
    init_storage()
    clean_name = Path(filename).name
    target_path = (STORAGE_DIR / clean_name).resolve()
    if target_path.parent != STORAGE_DIR.resolve():
        return None
    if target_path.exists() and target_path.is_file():
        return target_path
    return None
