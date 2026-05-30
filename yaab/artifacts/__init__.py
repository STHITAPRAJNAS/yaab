"""Artifact storage: binary/file blobs produced or consumed by agents.

Kept separate from session state (which is structured KV) and memory (which is
semantic). The default backend is in-process; swap in object storage for prod.
"""

from __future__ import annotations

import uuid
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: f"art_{uuid.uuid4().hex[:12]}")
    name: str
    mime_type: str = "application/octet-stream"
    size: int = 0
    metadata: dict = Field(default_factory=dict)


@runtime_checkable
class ArtifactService(Protocol):
    async def put(self, name: str, data: bytes, *, mime_type: str = ...) -> Artifact:
        ...

    async def get(self, artifact_id: str) -> Optional[bytes]:
        ...

    async def info(self, artifact_id: str) -> Optional[Artifact]:
        ...


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

    async def get(self, artifact_id: str) -> Optional[bytes]:
        return self._data.get(artifact_id)

    async def info(self, artifact_id: str) -> Optional[Artifact]:
        return self._meta.get(artifact_id)


__all__ = ["Artifact", "ArtifactService", "InMemoryArtifactService"]
