"""SQLite session backend — durable sessions for single-node deployments."""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

from ..types import Message
from .base import Session


class SQLiteSessionService:
    """Persist sessions in a SQLite table keyed by session id."""

    def __init__(self, path: str = "yaab_sessions.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.commit()

    def _load(self, session_id: str) -> Optional[Session]:
        row = self._conn.execute(
            "SELECT data FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session.model_validate_json(row[0])

    def _store(self, session: Session) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (id, data) VALUES (?, ?)",
            (session.id, session.model_dump_json()),
        )
        self._conn.commit()

    async def get(self, session_id: str) -> Optional[Session]:
        return self._load(session_id)

    async def get_or_create(self, session_id: Optional[str] = None) -> Session:
        if session_id:
            existing = self._load(session_id)
            if existing is not None:
                return existing
            session = Session(id=session_id)
        else:
            session = Session()
        self._store(session)
        return session

    async def save(self, session: Session) -> None:
        self._store(session)

    async def append(self, session_id: str, message: Message) -> None:
        session = await self.get_or_create(session_id)
        session.messages.append(message)
        self._store(session)

    async def delete(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
