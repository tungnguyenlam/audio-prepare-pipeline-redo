#!/usr/bin/env python3
"""Official High-Speed Model Downloader for Whisper & Audio Pipeline.

Uses standard Git LFS (the official Hugging Face repository backend) to clone
and download exact model weights with auto-resume, checksum verification,
and full bandwidth utilization.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTER_WHISPER_CACHE = PROJECT_ROOT / ".cache" / "faster_whisper"
FASTER_WHISPER_CACHE.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("base", "https://huggingface.co/Systran/faster-whisper-base"),
    ("medium", "https://huggingface.co/Systran/faster-whisper-medium"),
    ("large-v3", "https://huggingface.co/Systran/faster-whisper-large-v3"),
]


def download_repo_lfs(name: str, repo_url: str):
    target_dir = FASTER_WHISPER_CACHE / name
    print("=" * 65)
    print(f"🚀 BẮT ĐẦU TẢI MODEL: '{name.upper()}' ({repo_url})")
    print("=" * 65)

    # 1. Clone repository structure (without large files first)
    if target_dir.exists() and not (target_dir / ".git").exists():
        import shutil
        shutil.rmtree(target_dir)

    if not (target_dir / ".git").exists():
        print(f"  📥 1. Khởi tạo repository '{name}'...")
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            env=env,
            check=True,
        )
    else:
        print(f"  ⚡ Repository '{name}' đã có sẵn. Đang đồng bộ...")

    # Configure multi-threaded concurrent transfers for Git LFS
    subprocess.run(["git", "-C", str(target_dir), "config", "lfs.concurrenttransfers", "8"], check=True)

    # 2. Pull / fetch large LFS objects (model.bin) with progress
    print(f"  📦 2. Đang tải trọng số LFS 'model.bin' cho '{name}'...")
    t0 = time.time()
    res = subprocess.run(
        ["git", "-C", str(target_dir), "lfs", "pull"],
        check=True,
    )

    elapsed = max(time.time() - t0, 0.1)
    sz_mb = sum(f.stat().st_size for f in target_dir.glob("*") if f.is_file()) / (1024 * 1024)
    print(f"  ✅ Đã tải xong '{name}' ({sz_mb:.1f} MB) trong {elapsed:.1f}s!\n")


def main():
    print("=" * 65)
    print("⚡ BỘ TẢI CHÍNH THỨC TOÀN BỘ CÁC MODEL WHISPER (Git LFS Official)")
    print("=" * 65)
    t_start = time.time()

    for name, url in MODELS:
        download_repo_lfs(name, url)

    total_time = max(time.time() - t_start, 0.1)
    print("=" * 65)
    print(f"🎉 TẤT CẢ MODEL ĐÃ HOÀN TẤT VÀ SẴN SÀNG OFFLINE TRONG {total_time:.1f}s!")
    print("=" * 65)


if __name__ == "__main__":
    main()
