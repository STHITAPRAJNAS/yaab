"""Store-backed cancellation — cancel a run on any replica, stop it everywhere.

The runner already polls a :class:`~yaab.limits.CancellationToken` between steps
and before each tool call. :class:`StoreCancellationToken` keeps that exact
contract but layers in a durable signal: its ``cancelled`` also reflects the
run store's ``cancel_requested`` flag (poll-cached). A cancel recorded on one
replica is therefore honoured by whichever replica is actually executing the
run — with no in-process signal between them and no runner changes.

The base token's local flag and wall-clock deadline still short-circuit, so
nothing about timeouts or explicit ``cancel()`` calls changes.
"""

from __future__ import annotations

import time
from typing import Any

from ..limits import CancellationToken
from .base import RunStore


def _drive(coro: Any) -> Any:
    """Run a coroutine to completion from sync code.

    ``cancelled`` is a synchronous property (the runner calls it between steps),
    but a :class:`RunStore` read is ``async``. The in-memory / SQLite stores
    never actually suspend, so stepping the coroutine resolves it immediately.
    If a backend does suspend, we fall back to a fresh event loop for that one
    read rather than blocking the runner's loop.
    """
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    # The coroutine suspended on real I/O — finish it on a throwaway loop.
    import asyncio

    async def _wrap() -> Any:
        return await coro

    return asyncio.run(_wrap())


class StoreCancellationToken(CancellationToken):
    """A cancellation token whose ``cancelled`` also reflects a run store flag.

    Args:
        run_id: The run this token guards.
        store: The durable run store to consult for ``cancel_requested``.
        deadline: Optional wall-clock deadline (``time.monotonic()`` seconds),
            same as the base token.
        poll_interval: Minimum seconds between store reads. The durable flag is
            cached between reads so checking ``cancelled`` on every step does not
            hammer the backend; a cancel is observed after at most one interval.
            ``0`` reads the store on every check (useful in tests).
    """

    def __init__(
        self,
        run_id: str,
        store: RunStore,
        *,
        deadline: float | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        super().__init__(deadline=deadline)
        self._run_id = run_id
        self._store = store
        self._poll_interval = poll_interval
        self._last_poll: float | None = None
        self._store_cancelled = False

    def _refresh_from_store(self) -> None:
        """Read the durable cancel flag if the cache is stale."""
        now = time.monotonic()
        if self._last_poll is not None and now - self._last_poll < self._poll_interval:
            return
        self._last_poll = now
        record = _drive(self._store.get(self._run_id))
        if record is not None and record.cancel_requested:
            self._store_cancelled = True

    @property
    def cancelled(self) -> bool:
        # Local flag or wall-clock deadline first (cheap, no store read).
        if super().cancelled:
            return True
        # Then the durable, cross-replica signal (poll-cached).
        if not self._store_cancelled:
            self._refresh_from_store()
        if self._store_cancelled:
            self._reason = "cancelled"
            return True
        return False


__all__ = ["StoreCancellationToken"]
