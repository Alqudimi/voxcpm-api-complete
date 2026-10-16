"""Standalone ASGI entrypoint shared with the integrated router."""

from .api import app

__all__ = ["app"]
