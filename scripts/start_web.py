#!/usr/bin/env python3
"""Start Audio Web Applications (SonicStudio or SonicPipeline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Audio Pipeline Web Application")
    parser.add_argument(
        "--mode",
        choices=["studio", "pipeline"],
        default="pipeline",
        help="Web app to run: 'studio' (SonicStudio Interactive) or 'pipeline' (SonicPipeline Batch / Large-Scale) (default: pipeline)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port number (default: 8080 for studio, 8081 for pipeline)")
    args = parser.parse_args()

    if args.mode == "studio":
        from scripts.start_studio import free_port
        from src.web_studio.server import create_app
        from aiohttp import web

        port = args.port or 8080
        free_port(port)
        app = create_app()
        print(f"🎙️ Starting SonicStudio (Interactive) on http://{args.host}:{port}")
        web.run_app(app, host=args.host, port=port)
    else:
        from scripts.start_pipeline import free_port
        from src.web_pipeline.server import create_app
        from aiohttp import web

        port = args.port or 8081
        free_port(port)
        app = create_app()
        print(f"⚡ Starting SonicPipeline (Large-Scale Batch) on http://{args.host}:{port}")
        web.run_app(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
