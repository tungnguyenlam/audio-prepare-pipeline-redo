"""Unified REST backend and frontend host for both Sonic web applications."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable

from aiohttp import web
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".data" / "huggingface"))

from src.web_pipeline import server as pipeline_server
from src.web_studio import server as studio_server

logger = logging.getLogger("web_backend")

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]
SHUTDOWN_WATCHDOG_S = 6.0


def _ensure_own_process_group() -> None:
    """Put the backend in its own process group so restart can stop job children."""
    if os.name == "nt":
        return
    try:
        os.setpgrp()
    except OSError:
        pass


def terminate_descendant_processes(timeout_s: float = 1.5) -> None:
    """Stop child processes spawned by running jobs (yt-dlp, ffmpeg, demucs, MVSEP)."""
    try:
        import psutil
    except ImportError:
        return

    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return

    if not children:
        return

    logger.info("Terminating %d descendant process(es) from running jobs", len(children))
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(children, timeout=timeout_s)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass


def _start_shutdown_watchdog(timeout_s: float = SHUTDOWN_WATCHDOG_S) -> None:
    """Force-exit if shutdown is blocked by in-process model threads."""

    def _force_exit() -> None:
        time.sleep(timeout_s)
        logger.warning("Shutdown watchdog expired; forcing process exit")
        terminate_descendant_processes(timeout_s=0.5)
        os._exit(0)

    threading.Thread(
        target=_force_exit,
        name="sonic-shutdown-watchdog",
        daemon=True,
    ).start()


@web.middleware
async def no_cache_frontend_middleware(
    request: web.Request,
    handler: Handler,
) -> web.StreamResponse:
    """Disable caching for the independently mounted frontend assets."""
    response = await handler(request)
    if request.path.startswith(("/studio/", "/pipeline/")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


async def handle_root(request: web.Request) -> web.Response:
    """Redirect the backend root to the interactive frontend."""
    raise web.HTTPFound("/studio/")


async def handle_health(request: web.Request) -> web.Response:
    """Report the unified backend and its frontend mount points."""
    return web.json_response(
        {
            "status": "ok",
            "backend": "sonic",
            "frontends": {
                "studio": "/studio/",
                "pipeline": "/pipeline/",
            },
        }
    )


def create_app() -> web.Application:
    """Create the single backend used by both Sonic frontends.

    Returns:
        Configured aiohttp application with both API surfaces and frontend
        mounts.
    """
    _ensure_own_process_group()
    app = web.Application(
        client_max_size=2048 * 1024 * 1024,
        middlewares=[no_cache_frontend_middleware],
    )

    async def on_shutdown_begin(app: web.Application) -> None:
        logger.info("Backend shutting down; cancelling all running jobs")
        _start_shutdown_watchdog()

    async def on_shutdown_end(app: web.Application) -> None:
        terminate_descendant_processes()

    app.on_shutdown.append(on_shutdown_begin)
    studio_server.register_lifecycle(app)
    pipeline_server.register_lifecycle(app)
    app.on_shutdown.append(on_shutdown_end)
    studio_server.register_api_routes(app)
    pipeline_server.register_api_routes(app)

    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/", handle_root)
    app.router.add_get("/studio", studio_server.handle_index)
    app.router.add_get("/studio/", studio_server.handle_index)
    app.router.add_get("/pipeline", pipeline_server.handle_index)
    app.router.add_get("/pipeline/", pipeline_server.handle_index)
    app.router.add_static(
        "/studio/static/",
        path=studio_server.STATIC_DIR,
        name="studio-static",
    )
    app.router.add_static(
        "/pipeline/static/",
        path=pipeline_server.STATIC_DIR,
        name="pipeline-static",
    )

    return app
