"""Artifact storage: binary/file blobs produced or consumed by agents.

Kept separate from session state (which is structured KV) and memory (which is
semantic). The default backend is in-process; swap in object storage for prod.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: f"art_{uuid.uuid4().hex[:12]}")
    name: str
    mime_type: str = "application/octet-stream"
    size: int = 0
    metadata: dict = Field(default_factory=dict)


@runtime_checkable
class ArtifactService(Protocol):
    async def put(self, name: str, data: bytes, *, mime_type: str = ...) -> Artifact: ...

    async def get(self, artifact_id: str) -> bytes | None: ...

    async def info(self, artifact_id: str) -> Artifact | None: ...


class InMemoryArtifactService:
    """Store artifact bytes in a process-local dict."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._meta: dict[str, Artifact] = {}

    async def put(
        self, name: str, data: bytes, *, mime_type: str = "application/octet-stream"
    ) -> Artifact:
        art = Artifact(name=name, mime_type=mime_type, size=len(data))
        self._data[art.id] = data
        self._meta[art.id] = art
        return art

    async def get(self, artifact_id: str) -> bytes | None:
        return self._data.get(artifact_id)

    async def info(self, artifact_id: str) -> Artifact | None:
        return self._meta.get(artifact_id)


__all__ = [
    "Artifact",
    "ArtifactService",
    "InMemoryArtifactService",
    "SQLiteArtifactService",
    "PostgresArtifactService",
    "RedisArtifactService",
]


def __getattr__(name: str) -> Any:
    # Lazy imports so psycopg / redis are only needed when their backend is used.
    if name == "SQLiteArtifactService":
        from .sqlite import SQLiteArtifactService

        return SQLiteArtifactService
    if name == "PostgresArtifactService":
        from .postgres import PostgresArtifactService

        return PostgresArtifactService
    if name == "RedisArtifactService":
        from .redis import RedisArtifactService

        return RedisArtifactService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_backends() -> None:
    """Register artifact backends as ``artifact`` components (discoverable by name)."""
    from ..extensions import register

    register("artifact", "memory", lambda **kw: InMemoryArtifactService())

    def _sqlite(**kw: Any) -> Any:
        from .sqlite import SQLiteArtifactService

        return SQLiteArtifactService(**kw)

    def _pg(**kw: Any) -> Any:
        from .postgres import PostgresArtifactService

        return PostgresArtifactService(**kw)

    def _redis(**kw: Any) -> Any:
        from .redis import RedisArtifactService

        return RedisArtifactService(**kw)

    register("artifact", "sqlite", _sqlite)
    register("artifact", "postgres", _pg)
    register("artifact", "aurora", _pg)  # Aurora PostgreSQL via the same driver
    register("artifact", "redis", _redis)


_register_backends()
