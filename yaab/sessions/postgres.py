"""Postgres session backend for production deployments.

Uses ``psycopg`` (v3), imported lazily so it is only required when this backend
is actually constructed. Sessions are stored as JSONB keyed by id; the same
``SessionService`` protocol as the in-memory/SQLite backends, so swapping it in
is a one-line change.
"""

from __future__ import annotations

from ..types import Message
from .base import Session


def _require_psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "psycopg is required for the Postgres backends. "
            "Install with `pip install 'yaab[postgres]'`."
        ) from exc
    return psycopg


class PostgresSessionService:
    """Persist sessions in a Postgres ``jsonb`` column keyed by session id."""

    def __init__(self, dsn: str, *, table: str = "yaab_sessions") -> None:
        psycopg = _require_psycopg()
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, data JSONB NOT NULL)"
        )

    def _load(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            f"SELECT data FROM {self._table} WHERE id = %s", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session.model_validate(row[0])

    def _store(self, session: Session) -> None:
        import json

        self._conn.execute(
            f"INSERT INTO {self._table} (id, data) VALUES (%s, %s) "
            f"ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data",
            (session.id, json.dumps(session.model_dump())),
        )

    async def get(self, session_id: str) -> Session | None:
        return self._load(session_id)

    async def get_or_create(self, session_id: str | None = None) -> Session:
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
        self._conn.execute(f"DELETE FROM {self._table} WHERE id = %s", (session_id,))


__all__ = ["PostgresSessionService"]
