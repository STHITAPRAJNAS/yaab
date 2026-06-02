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
    """Manage versioned artifacts scoped by app, user, and session.

    When the underlying service exposes a durable version index (the durable
    backends do, via ``append_version``/``list_version_ids``), version history is
    read and written through the backend so it is shared across every replica.
    Otherwise an in-process index is used (the in-memory default).
    """

    def __init__(self, service: ArtifactService | None = None) -> None:
        self.service = service or InMemoryArtifactService()
        # scoped name -> ordered list of artifact ids (one per version). Used
        # only when the backend has no durable version index of its own.
        self._versions: dict[str, list[str]] = {}
        self._durable_index = hasattr(self.service, "append_version") and hasattr(
            self.service, "list_version_ids"
        )

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
        if self._durable_index:
            return await self.service.append_version(key, artifact.id)  # type: ignore[attr-defined]
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
        ids = await self._version_ids(key)
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
        return len(await self._version_ids(key))

    async def list_artifacts(
        self,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        session_id: str = "default",
    ) -> list[str]:
        prefix = f"{app_name}:{user_id}:{session_id}:"
        if self._durable_index and hasattr(self.service, "version_scopes"):
            scopes = await self.service.version_scopes(prefix)  # type: ignore[attr-defined]
        else:
            scopes = [k for k in self._versions if k.startswith(prefix)]
        return [k[len(prefix) :] for k in scopes]

    async def _version_ids(self, key: str) -> list[str]:
        """Ordered artifact ids for ``key`` from the durable or in-process index."""
        if self._durable_index:
            return await self.service.list_version_ids(key)  # type: ignore[attr-defined]
        return list(self._versions.get(key, []))


__all__ = ["ArtifactManager"]
