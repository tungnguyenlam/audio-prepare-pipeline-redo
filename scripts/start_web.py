#!/usr/bin/env python3
"""Start the shared backend for SonicStudio and SonicPipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from start_studio import free_port


def main() -> None:
    """Run the unified Sonic web backend."""
    parser = argparse.ArgumentParser(
        description="Shared backend for SonicStudio and SonicPipeline"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8765, help="Backend port")
    args = parser.parse_args()

    free_port(args.port, host=args.host)

    from aiohttp import web

    from src.web_backend.server import create_app
    from src.web_studio.server import get_system_device_info

    device = get_system_device_info()["device_name"]
    base_url = f"http://{args.host}:{args.port}"
    print("=" * 65)
    print("   SONIC WEB — Shared Backend")
    print(f"   SonicStudio:   {base_url}/studio/")
    print(f"   SonicPipeline: {base_url}/pipeline/")
    print(f"   Compute Device: {device}")
    print("=" * 65)
    web.run_app(create_app(), host=args.host, port=args.port, shutdown_timeout=5)


if __name__ == "__main__":
    main()
