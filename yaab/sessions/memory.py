"""In-memory session backend (default for dev and tests)."""

from __future__ import annotations

from typing import Optional

from ..types import Message
from .base import Session


class InMemorySessionService:
    """Keeps sessions in a process-local dict. Not durable across restarts."""

    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    async def get(self, session_id: str) -> Optional[Session]:
        return self._store.get(session_id)

    async def get_or_create(self, session_id: Optional[str] = None) -> Session:
        if session_id and session_id in self._store:
            return self._store[session_id]
        session = Session(id=session_id) if session_id else Session()
        self._store[session.id] = session
        return session

    async def save(self, session: Session) -> None:
        self._store[session.id] = session

    async def append(self, session_id: str, message: Message) -> None:
        session = await self.get_or_create(session_id)
        session.messages.append(message)

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
