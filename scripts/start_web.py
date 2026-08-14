#!/usr/bin/env python3
"""Start SonicStudio Web Application."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def free_port(port: int) -> None:
    """Kill any existing process occupying the target port."""
    current_pid = os.getpid()
    try:
        if shutil.which("lsof"):
            out = subprocess.check_output(
                ["lsof", "-ti", f":{port}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                for pid_str in out.split():
                    try:
                        pid = int(pid_str)
                        if pid != current_pid:
                            print(f"⚠️  Port {port} is occupied by PID {pid}. Releasing...")
                            os.kill(pid, 9)
                    except (ValueError, ProcessLookupError, PermissionError):
                        pass
                time.sleep(0.5)
        elif shutil.which("fuser"):
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True,
                check=False,
            )
            time.sleep(0.5)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio Prepare Pipeline Web Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    args = parser.parse_args()

    free_port(args.port)

    from src.web.server import create_app
    from aiohttp import web
    import torch

    def get_device():
        if torch.cuda.is_available():
            return "CUDA"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "MPS"
        return "CPU"

    app = create_app()
    print("=" * 60)
    print("   🎙️  SONICSTUDIO - Audio Prepare & Separation Suite")
    print(f"   🚀 Running at: http://{args.host}:{args.port}")
    print(f"   ⚡ Compute Device: {get_device()}")
    print("=" * 60)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
