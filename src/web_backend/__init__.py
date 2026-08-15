"""Shared backend for the SonicStudio and SonicPipeline frontends."""

from src.web_backend.server import create_app

__all__ = ["create_app"]
