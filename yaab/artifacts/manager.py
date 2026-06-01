"""Artifact manager — named, versioned binary storage.

Adds ``(app_name, user_id, session_id)`` scoping and automatic versioning on
top of an :class:`ArtifactService`: saving the same name twice creates v2, v3,
…, and you can load any version or list them — all while staying
backend-agnostic.
"""

from __future__ import annotations

from . import ArtifactService, InMemoryArtifactService

DEFAULT_APP = "default"
DEFAULT_USER = "default"


class ArtifactManager:
    """Manage versioned artifacts scoped by app, user, and session."""

    def __init__(self, service: ArtifactService | None = None) -> None:
        self.service = service or InMemoryArtifactService()
        # scoped name -> ordered list of artifact ids (one per version)
        self._versions: dict[str, list[str]] = {}

    @staticmethod
    def _scope(app_name: str, user_id: str, session_id: str, name: str) -> str:
        return f"{app_name}:{user_id}:{session_id}:{name}"

    async def save(
        self,
        name: str,
        data: bytes,
        *,
        mime_type: str = "application/octet-stream",
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        session_id: str = "default",
    ) -> int:
        """Save a new version of ``name``; returns the version number (1-based)."""
        key = self._scope(app_name, user_id, session_id, name)
        artifact = await self.service.put(name, data, mime_type=mime_type)
        self._versions.setdefault(key, []).append(artifact.id)
        return len(self._versions[key])

    async def load(
        self,
        name: str,
        *,
        version: int | None = None,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        session_id: str = "default",
    ) -> bytes | None:
        """Load a version of ``name`` (latest if ``version`` is None)."""
        key = self._scope(app_name, user_id, session_id, name)
        ids = self._versions.get(key)
        if not ids:
            return None
        idx = (version - 1) if version else (len(ids) - 1)
        if idx < 0 or idx >= len(ids):
            return None
        return await self.service.get(ids[idx])

    async def list_versions(
        self,
        name: str,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        session_id: str = "default",
    ) -> int:
        key = self._scope(app_name, user_id, session_id, name)
        return len(self._versions.get(key, []))

    async def list_artifacts(
        self,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        session_id: str = "default",
    ) -> list[str]:
        prefix = f"{app_name}:{user_id}:{session_id}:"
        return [k[len(prefix) :] for k in self._versions if k.startswith(prefix)]


__all__ = ["ArtifactManager"]
