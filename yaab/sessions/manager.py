"""Session manager — scoped, high-level session operations.

Where a :class:`SessionService` is the raw storage protocol, the
:class:`SessionManager` adds ``(app_name, user_id, session_id)`` scoping,
plus listing, state updates, and event appends. It
composes a stable storage key from the scope so any flat ``SessionService``
backend (in-memory, SQLite, Postgres, Redis) works unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..types import Message, Role
from .base import Session, SessionService
from .memory import InMemorySessionService

if TYPE_CHECKING:
    from ..state import State

DEFAULT_APP = "default"
DEFAULT_USER = "default"


class SessionManager:
    """Manage conversation sessions scoped by app and user."""

    def __init__(self, service: SessionService | None = None) -> None:
        self.service = service or InMemorySessionService()
        # (app, user) -> ordered list of session ids (best-effort index).
        self._index: dict[tuple[str, str], list[str]] = {}
        # Shared state stores for the prefix scopes.
        self._app_state: dict[str, dict[str, Any]] = {}  # app -> app: state
        self._user_state: dict[tuple[str, str], dict[str, Any]] = {}  # (app,user) -> user: state

    @staticmethod
    def _key(app_name: str, user_id: str, session_id: str) -> str:
        return f"{app_name}:{user_id}:{session_id}"

    async def create_session(
        self,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        session_id: str | None = None,
        state: dict[str, Any] | None = None,
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
    ) -> Session | None:
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
        limit: int | None = None,
        offset: int = 0,
    ) -> list[str]:
        """List a user's session ids, with optional pagination."""
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

    # --- prefix-scoped state (temp:/user:/app:) -------------------
    async def resolve_state(
        self,
        session_id: str,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
    ) -> State:
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

    async def save_state(self, session_id: str, state: State) -> None:
        """Persist the durable (non-temp) subset of a resolved State."""
        session = await self.service.get_or_create(session_id)
        # session.state is the same dict the State wrote into; just persist it.
        session.state.update(state.session)
        await self.service.save(session)

    # --- rewind & migration ---------------------------------------
    @staticmethod
    def _turn_starts(messages: list[Message]) -> list[int]:
        """Indices where each conversational turn begins (each user message)."""
        return [i for i, m in enumerate(messages) if m.role is Role.USER]

    async def rewind(self, session_id: str, *, keep_turns: int) -> Session:
        """Roll the conversation back to keep only the first ``keep_turns`` turns.

        A *turn* starts at a user message and runs until the next one, so keeping
        N turns keeps the first N user messages and everything that followed each
        (up to the next user message). The session's structured ``state`` is kept
        intact — only the message history is truncated. The result is persisted.
        """
        session = await self.service.get_or_create(session_id)
        starts = self._turn_starts(session.messages)
        if keep_turns <= 0:
            cut = 0
        elif keep_turns >= len(starts):
            return session  # nothing to drop
        else:
            cut = starts[keep_turns]  # first index of the (keep_turns+1)-th turn
        session.messages = session.messages[:cut]
        await self.service.save(session)
        return session

    async def rewind_last(self, session_id: str, *, turns: int = 1) -> Session:
        """Drop the most recent ``turns`` turns (an "undo the last exchange")."""
        session = await self.service.get(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id!r}")
        total = len(self._turn_starts(session.messages))
        return await self.rewind(session_id, keep_turns=max(0, total - turns))

    async def migrate_session(self, session_id: str, *, to_service: SessionService) -> Session:
        """Copy a session (messages + state) into another backend.

        Reads the session from this manager's service and writes a copy — same id,
        messages, and structured state — into ``to_service``, so a conversation
        moves across stores or schema versions without loss. The source is left
        untouched. Returns the migrated copy.
        """
        session = await self.service.get(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id!r}")
        copy = Session(
            id=session.id,
            messages=list(session.messages),
            state=dict(session.state),
        )
        await to_service.save(copy)
        return copy


__all__ = ["SessionManager"]
