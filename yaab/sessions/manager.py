"""Session manager — scoped, high-level session operations (ADK-style).

Where a :class:`SessionService` is the raw storage protocol, the
:class:`SessionManager` adds the ``(app_name, user_id, session_id)`` scoping
ADK developers expect, plus listing, state updates, and event appends. It
composes a stable storage key from the scope so any flat ``SessionService``
backend (in-memory, SQLite, Postgres, Redis) works unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..types import Message, Role
from .base import Session, SessionService
from .memory import InMemorySessionService

if TYPE_CHECKING:
    from ..state import State

DEFAULT_APP = "default"
DEFAULT_USER = "default"


class SessionManager:
    """Manage conversation sessions scoped by app and user."""

    def __init__(self, service: Optional[SessionService] = None) -> None:
        self.service = service or InMemorySessionService()
        # (app, user) -> ordered list of session ids (best-effort index).
        self._index: dict[tuple[str, str], list[str]] = {}
        # Shared state stores for the prefix scopes (ADK-style).
        self._app_state: dict[str, dict[str, Any]] = {}            # app -> app: state
        self._user_state: dict[tuple[str, str], dict[str, Any]] = {}  # (app,user) -> user: state

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
        self,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[str]:
        """List a user's session ids, with optional pagination (ADK #4621)."""
        ids = list(self._index.get((app_name, user_id), []))
        ids = ids[offset:]
        if limit is not None:
            ids = ids[:limit]
        return ids

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

    # --- prefix-scoped state (ADK temp:/user:/app:) -------------------
    async def resolve_state(
        self,
        session_id: str,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
    ) -> "State":
        """Build a prefix-routed :class:`~yaab.state.State` for a session.

        Reads/writes to ``app:``/``user:`` keys hit the shared stores; ``temp:``
        is ephemeral; unprefixed keys are session-scoped. Persist the durable
        subset back with :meth:`save_state`.
        """
        from ..state import State

        session = await self.service.get_or_create(session_id)
        app_store = self._app_state.setdefault(app_name, {})
        user_store = self._user_state.setdefault((app_name, user_id), {})
        return State(session=session.state, user=user_store, app=app_store)

    async def save_state(self, session_id: str, state: "State") -> None:
        """Persist the durable (non-temp) subset of a resolved State."""
        session = await self.service.get_or_create(session_id)
        # session.state is the same dict the State wrote into; just persist it.
        session.state.update(state.session)
        await self.service.save(session)


__all__ = ["SessionManager"]
