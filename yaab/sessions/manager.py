"""Session manager — scoped, high-level session operations (ADK-style).

Where a :class:`SessionService` is the raw storage protocol, the
:class:`SessionManager` adds the ``(app_name, user_id, session_id)`` scoping
ADK developers expect, plus listing, state updates, and event appends. It
composes a stable storage key from the scope so any flat ``SessionService``
backend (in-memory, SQLite, Postgres, Redis) works unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from ..types import Message, Role
from .base import Session, SessionService
from .memory import InMemorySessionService

DEFAULT_APP = "default"
DEFAULT_USER = "default"


class SessionManager:
    """Manage conversation sessions scoped by app and user."""

    def __init__(self, service: Optional[SessionService] = None) -> None:
        self.service = service or InMemorySessionService()
        # (app, user) -> ordered list of session ids (best-effort index).
        self._index: dict[tuple[str, str], list[str]] = {}

    @staticmethod
    def _key(app_name: str, user_id: str, session_id: str) -> str:
        return f"{app_name}:{user_id}:{session_id}"

    async def create_session(
        self,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        session_id: Optional[str] = None,
        state: Optional[dict[str, Any]] = None,
    ) -> Session:
        session = await self.service.get_or_create(
            self._key(app_name, user_id, session_id) if session_id else None
        )
        if state:
            session.state.update(state)
            await self.service.save(session)
        self._index.setdefault((app_name, user_id), [])
        if session.id not in self._index[(app_name, user_id)]:
            self._index[(app_name, user_id)].append(session.id)
        return session

    async def get_session(
        self, *, app_name: str = DEFAULT_APP, user_id: str = DEFAULT_USER, session_id: str
    ) -> Optional[Session]:
        # Accept either a bare id or an already-scoped key.
        direct = await self.service.get(session_id)
        if direct is not None:
            return direct
        return await self.service.get(self._key(app_name, user_id, session_id))

    async def list_sessions(
        self, *, app_name: str = DEFAULT_APP, user_id: str = DEFAULT_USER
    ) -> list[str]:
        return list(self._index.get((app_name, user_id), []))

    async def delete_session(
        self, *, app_name: str = DEFAULT_APP, user_id: str = DEFAULT_USER, session_id: str
    ) -> None:
        await self.service.delete(session_id)
        bucket = self._index.get((app_name, user_id))
        if bucket and session_id in bucket:
            bucket.remove(session_id)

    async def append_message(self, session_id: str, message: Message) -> None:
        await self.service.append(session_id, message)

    async def append_text(self, session_id: str, role: Role, text: str) -> None:
        await self.service.append(session_id, Message(role=role, content=text))

    async def update_state(self, session_id: str, **changes: Any) -> Session:
        session = await self.service.get_or_create(session_id)
        session.state.update(changes)
        await self.service.save(session)
        return session

    async def get_state(self, session_id: str) -> dict[str, Any]:
        session = await self.service.get(session_id)
        return dict(session.state) if session else {}


__all__ = ["SessionManager"]
