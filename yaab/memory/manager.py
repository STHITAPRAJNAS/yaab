"""Memory manager — scoped long-term memory with session ingestion (ADK-style).

Adds ``(app_name, user_id)`` namespacing on top of a :class:`MemoryService`,
plus :meth:`add_session_to_memory` which ingests a finished conversation into
long-term memory so future runs can recall it — the ADK ``MemoryService``
workflow, made backend-agnostic.
"""

from __future__ import annotations

from typing import Any

from ..types import Role
from . import InMemoryVectorMemory, MemoryRecord, MemoryService

DEFAULT_APP = "default"
DEFAULT_USER = "default"


class MemoryManager:
    """Manage long-term memory scoped by app and user."""

    def __init__(self, service: MemoryService | None = None) -> None:
        self.service = service or InMemoryVectorMemory()

    @staticmethod
    def _ns(app_name: str, user_id: str) -> dict[str, str]:
        return {"app_name": app_name, "user_id": user_id}

    async def add(
        self,
        text: str,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        meta = {**self._ns(app_name, user_id), **(metadata or {})}
        return await self.service.add(text, metadata=meta)

    async def search(
        self,
        query: str,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        k: int = 5,
    ) -> list[tuple[MemoryRecord, float]]:
        # Over-fetch, then filter to the requested namespace.
        hits = await self.service.search(query, k=k * 4)
        scoped = [
            (rec, score)
            for rec, score in hits
            if rec.metadata.get("app_name", app_name) == app_name
            and rec.metadata.get("user_id", user_id) == user_id
        ]
        return scoped[:k]

    async def add_session_to_memory(
        self,
        session: Any,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        roles: tuple[Role, ...] = (Role.USER, Role.ASSISTANT),
    ) -> list[MemoryRecord]:
        """Ingest a session's messages into long-term memory."""
        records: list[MemoryRecord] = []
        for msg in getattr(session, "messages", []):
            if msg.role in roles and msg.content:
                records.append(
                    await self.add(
                        msg.content,
                        app_name=app_name,
                        user_id=user_id,
                        metadata={"session_id": session.id, "role": msg.role.value},
                    )
                )
        return records


__all__ = ["MemoryManager"]
