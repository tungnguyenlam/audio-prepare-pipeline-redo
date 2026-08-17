"""Unified REST backend and frontend host for both Sonic web applications."""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from aiohttp import web
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

from src.web_pipeline import server as pipeline_server
from src.web_studio import server as studio_server


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


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
    app = web.Application(
        client_max_size=2048 * 1024 * 1024,
        middlewares=[no_cache_frontend_middleware],
    )

    studio_server.register_lifecycle(app)
    pipeline_server.register_lifecycle(app)
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
