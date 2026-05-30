"""Session abstraction: conversation history + structured KV state.

Following ADK, a :class:`Session` keeps conversation history *and* a structured
key-value ``state`` store as distinct concerns. A :class:`SessionService` is
the pluggable backend (in-memory, SQLite, Postgres, Redis, ...).
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..types import Message


class Session(BaseModel):
    """A conversation thread plus its structured state."""

    id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    messages: list[Message] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class SessionService(Protocol):
    """Pluggable session backend."""

    async def get(self, session_id: str) -> Session | None: ...

    async def get_or_create(self, session_id: str | None = None) -> Session: ...

    async def save(self, session: Session) -> None: ...

    async def append(self, session_id: str, message: Message) -> None: ...

    async def delete(self, session_id: str) -> None: ...
