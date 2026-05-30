"""Session services."""

from __future__ import annotations

from typing import Any

from .base import Session, SessionService
from .memory import InMemorySessionService
from .sqlite import SQLiteSessionService

__all__ = [
    "Session",
    "SessionService",
    "InMemorySessionService",
    "SQLiteSessionService",
    "PostgresSessionService",
    "RedisSessionService",
]


def __getattr__(name: str) -> Any:
    # Lazy imports so psycopg / redis are only needed when their backend is used.
    if name == "PostgresSessionService":
        from .postgres import PostgresSessionService

        return PostgresSessionService
    if name == "RedisSessionService":
        from .redis import RedisSessionService

        return RedisSessionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_backends() -> None:
    """Register session backends as ``session`` components (discoverable by name)."""
    from ..extensions import register

    register("session", "memory", lambda **kw: InMemorySessionService())
    register("session", "sqlite", lambda **kw: SQLiteSessionService(**kw))

    def _pg(**kw: Any) -> Any:
        from .postgres import PostgresSessionService

        return PostgresSessionService(**kw)

    def _redis(**kw: Any) -> Any:
        from .redis import RedisSessionService

        return RedisSessionService(**kw)

    register("session", "postgres", _pg)
    register("session", "aurora", _pg)  # Aurora PostgreSQL via the same driver
    register("session", "redis", _redis)


_register_backends()
