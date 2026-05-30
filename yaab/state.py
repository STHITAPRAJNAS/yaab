"""Prefix-scoped state — ADK-style ``temp:`` / ``user:`` / ``app:`` scoping.

A plain session ``state`` dict is session-scoped. Real apps also need values that
outlive a single session (a user's preferences) or span the whole app (a shared
config), plus scratch values that must *never* be persisted (a one-turn flag).
ADK expresses this with key prefixes; :class:`State` implements the same
convention over pluggable stores so the scope of every value is explicit and
auditable.

Prefixes (mirrors ADK):

* ``app:<key>``  — shared across all users and sessions of the app;
* ``user:<key>`` — shared across one user's sessions;
* ``temp:<key>`` — ephemeral, lives only for the current run, never persisted;
* ``<key>``      — session-scoped (the default).

Reads and writes route to the right backing store automatically; ``persisted()``
returns everything except ``temp:`` for durable storage.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any

APP_PREFIX = "app:"
USER_PREFIX = "user:"
TEMP_PREFIX = "temp:"


def scope_of(key: str) -> str:
    """Return the scope name for a key: 'app' | 'user' | 'temp' | 'session'."""
    if key.startswith(APP_PREFIX):
        return "app"
    if key.startswith(USER_PREFIX):
        return "user"
    if key.startswith(TEMP_PREFIX):
        return "temp"
    return "session"


class State(MutableMapping):
    """A dict-like view over session/user/app/temp scopes, routed by key prefix.

    Backing stores are plain dicts (or any MutableMapping), so they can be
    persisted independently — session state with the session, user state with
    the user, app state globally — while ``temp:`` is held only in memory.
    """

    def __init__(
        self,
        session: dict[str, Any] | None = None,
        *,
        user: MutableMapping[str, Any] | None = None,
        app: MutableMapping[str, Any] | None = None,
    ) -> None:
        self._session: dict[str, Any] = session if session is not None else {}
        self._user: MutableMapping[str, Any] = user if user is not None else {}
        self._app: MutableMapping[str, Any] = app if app is not None else {}
        self._temp: dict[str, Any] = {}

    def _store(self, key: str) -> MutableMapping[str, Any]:
        scope = scope_of(key)
        if scope == "app":
            return self._app
        if scope == "user":
            return self._user
        if scope == "temp":
            return self._temp
        return self._session

    def __getitem__(self, key: str) -> Any:
        return self._store(key)[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store(key)[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store(key)[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._app
        yield from self._user
        yield from self._session
        yield from self._temp

    def __len__(self) -> int:
        return len(self._app) + len(self._user) + len(self._session) + len(self._temp)

    def __repr__(self) -> str:
        return f"State({dict(self)!r})"

    # --- scope accessors ----------------------------------------------
    @property
    def session(self) -> dict[str, Any]:
        return self._session

    @property
    def user(self) -> MutableMapping[str, Any]:
        return self._user

    @property
    def app(self) -> MutableMapping[str, Any]:
        return self._app

    @property
    def temp(self) -> dict[str, Any]:
        return self._temp

    def persisted(self) -> dict[str, Any]:
        """Everything except ``temp:`` — the durable subset to write back."""
        out: dict[str, Any] = {}
        out.update(self._app)
        out.update(self._user)
        out.update(self._session)
        return out

    def to_dict(self) -> dict[str, Any]:
        """A flat snapshot across all scopes (including temp)."""
        return dict(self)


__all__ = ["State", "scope_of", "APP_PREFIX", "USER_PREFIX", "TEMP_PREFIX"]
