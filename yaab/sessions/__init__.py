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
]


def __getattr__(name: str) -> Any:
    # Lazy import so psycopg is only needed when the Postgres backend is used.
    if name == "PostgresSessionService":
        from .postgres import PostgresSessionService

        return PostgresSessionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
