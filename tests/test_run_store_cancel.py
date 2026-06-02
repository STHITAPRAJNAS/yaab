"""Cross-replica cancel — a cancel issued anywhere stops the run everywhere.

The store carries a durable ``cancel_requested`` flag. A
:class:`StoreCancellationToken` reads that flag (poll-cached) on top of the
existing local/deadline cancel logic, so a cancel recorded by one replica is
honoured by whichever replica is actually executing the run — without any
in-process signal between them.

These tests simulate two replicas as two store views over one SQLite file.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from yaab.exceptions import RunCancelled
from yaab.limits import CancellationToken
from yaab.runs import RunRecord, SQLiteRunStore
from yaab.runs.cancel import StoreCancellationToken


def _record(run_id: str) -> RunRecord:
    now = time.time()
    return RunRecord(run_id=run_id, agent="svc", created_at=now, updated_at=now)


def test_is_a_cancellation_token():
    """The store-backed token is drop-in for the runner's existing token."""
    store = SQLiteRunStore(":memory:")
    tok = StoreCancellationToken("r1", store)
    assert isinstance(tok, CancellationToken)


def test_local_cancel_still_works(tmp_path):
    """The inherited local flag short-circuits without touching the store."""
    store = SQLiteRunStore(str(tmp_path / "c.db"))

    async def go() -> None:
        await store.create(_record("r1"))
        tok = StoreCancellationToken("r1", store)
        assert tok.cancelled is False
        tok.cancel("api_cancel")
        assert tok.cancelled is True
        with pytest.raises(RunCancelled):
            tok.raise_if_cancelled()

    asyncio.run(go())


def test_deadline_cancel_still_works():
    """The inherited wall-clock deadline path is preserved."""
    store = SQLiteRunStore(":memory:")
    tok = StoreCancellationToken("r1", store, deadline=time.monotonic() - 1.0)
    assert tok.cancelled is True


def test_store_flag_flips_cancelled(tmp_path):
    """A cancel recorded in the store flips ``cancelled`` without setting the
    local ``_cancelled`` flag first — proving the signal came from the store."""
    store = SQLiteRunStore(str(tmp_path / "c.db"))

    async def go() -> None:
        await store.create(_record("r1"))
        tok = StoreCancellationToken("r1", store, poll_interval=0.0)
        assert tok.cancelled is False
        # The in-memory flag is untouched.
        assert tok._cancelled is False

        await store.request_cancel("r1")
        # Next read sees the durable flag (poll_interval=0 forces a refresh).
        assert tok.cancelled is True

    asyncio.run(go())


def test_cross_pod_cancel_over_one_file(tmp_path):
    """Cancel issued on pod B's store view stops the run executing on pod A.

    Pod A holds the executing token; pod B issues the cancel through its own
    independent store instance over the same SQLite file. A's token must honour
    it — the cross-replica contract.
    """
    path = str(tmp_path / "shared.db")
    pod_a_store = SQLiteRunStore(path)
    pod_b_store = SQLiteRunStore(path)

    async def go() -> None:
        await pod_a_store.create(_record("r1"))
        token_on_a = StoreCancellationToken("r1", pod_a_store, poll_interval=0.0)
        assert token_on_a.cancelled is False

        # Pod B cancels via its own store view.
        found = await pod_b_store.request_cancel("r1")
        assert found is True

        # Pod A's executing token now reports cancelled.
        assert token_on_a.cancelled is True
        with pytest.raises(RunCancelled) as exc:
            token_on_a.raise_if_cancelled()
        assert "cancel" in str(exc.value).lower()

    asyncio.run(go())


def test_poll_interval_caches_store_reads(tmp_path):
    """Within ``poll_interval`` the token does not re-read the store, then it
    refreshes — so a cancel is observed after at most one interval, and the
    store isn't hammered on every check."""
    path = str(tmp_path / "poll.db")
    store = SQLiteRunStore(path)

    async def go() -> None:
        await store.create(_record("r1"))
        tok = StoreCancellationToken("r1", store, poll_interval=100.0)
        # First check primes the cache (not cancelled).
        assert tok.cancelled is False
        await store.request_cancel("r1")
        # Still within the long interval -> cached, not yet cancelled.
        assert tok.cancelled is False
        # Force the cache to look stale and re-read.
        tok._last_poll = time.monotonic() - 1000.0
        assert tok.cancelled is True

    asyncio.run(go())


def test_missing_record_is_not_cancelled():
    """A token for a run that doesn't exist yet defaults to not-cancelled."""
    store = SQLiteRunStore(":memory:")
    tok = StoreCancellationToken("ghost", store, poll_interval=0.0)
    assert tok.cancelled is False
