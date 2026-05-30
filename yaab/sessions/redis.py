"""Redis session backend for distributed/cloud deployments.

Uses ``redis`` (``pip install redis``), imported lazily. Sessions are stored as
JSON under a key namespace; works against self-managed Redis, Amazon
ElastiCache / MemoryDB, Azure Cache for Redis, and Redis Cloud. Same
``SessionService`` protocol as the in-memory / SQLite / Postgres backends, so
swapping it in is a one-line change.
"""

from __future__ import annotations

from typing import Any

from ..types import Message
from .base import Session


class RedisSessionService:
    """Persist sessions as JSON values in Redis, keyed by ``<prefix>:<id>``."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "yaab:session",
        ttl_seconds: int | None = None,
        client: Any = None,
    ) -> None:
        self._redis: Any
        if client is not None:
            self._redis = client
        else:
            try:
                import redis  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "redis is required for RedisSessionService. `pip install redis`."
                ) from exc
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    def _load(self, session_id: str) -> Session | None:
        raw = self._redis.get(self._key(session_id))
        return Session.model_validate_json(raw) if raw else None

    def _store(self, session: Session) -> None:
        self._redis.set(self._key(session.id), session.model_dump_json(), ex=self._ttl)

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
        self._redis.delete(self._key(session_id))


__all__ = ["RedisSessionService"]
