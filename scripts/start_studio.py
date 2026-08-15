#!/usr/bin/env python3
"""Start SonicStudio (Interactive Audio Exploration Studio) Web Application."""

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


def free_port(port: int, host: str = "127.0.0.1") -> None:
    """Kill any existing process occupying the target port across Linux/macOS."""
    import re
    import signal
    import socket

    current_pid = os.getpid()
    pids: set[int] = set()

    # 1. Try lsof
    if shutil.which("lsof"):
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{port}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            for pid_str in out.split():
                if pid_str.isdigit():
                    pids.add(int(pid_str))
        except Exception:
            pass

    # 2. Try ss (standard on all modern Linux installations)
    if not pids and shutil.which("ss"):
        try:
            out = subprocess.check_output(
                ["ss", "-lptn", f"sport = :{port}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            for match in re.finditer(r"pid=(\d+)", out):
                pids.add(int(match.group(1)))
        except Exception:
            pass

    # 3. Try fuser
    if not pids and shutil.which("fuser"):
        try:
            out = subprocess.check_output(
                ["fuser", f"{port}/tcp"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            for pid_str in out.split():
                if pid_str.isdigit():
                    pids.add(int(pid_str))
        except Exception:
            pass

    pids.discard(current_pid)

    if pids:
        pid_list = ", ".join(str(p) for p in sorted(pids))
        print(f"⚠️  Port {port} is occupied by PID(s): {pid_list}. Releasing...")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.3)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.3)

    # Validate if port was freed
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            if s.connect_ex((host, port)) == 0:
                print(f"⚠️  Notice: Port {port} appears to still be busy. If binding fails, pass a different port with --port <PORT>.")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="SonicStudio - Interactive Audio Exploration Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port number (default: 8765)")
    args = parser.parse_args()

    free_port(args.port, host=args.host)

    from src.web_backend.server import create_app
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
    print("   🎙️  SONIC WEB — Shared Backend")
    print(f"   🚀 SonicStudio: http://{args.host}:{args.port}/studio/")
    print(f"   ⚡ SonicPipeline: http://{args.host}:{args.port}/pipeline/")
    print(f"   ⚡ Compute Device: {get_device()}")
    print("=" * 60)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
