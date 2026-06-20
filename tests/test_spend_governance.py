from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_inmemory_ledger_record_and_total():
    from yaab.governance.budget import InMemorySpendStore

    store = InMemorySpendStore()
    await store.record("id:alice", 0.5, at=100.0)
    await store.record("id:alice", 0.25, at=200.0)
    await store.record("id:bob", 1.0, at=150.0)

    assert await store.total("id:alice") == pytest.approx(0.75)
    assert await store.total("id:bob") == pytest.approx(1.0)
    assert await store.total("id:carol") == 0.0
    # `since` excludes earlier entries.
    assert await store.total("id:alice", since=150.0) == pytest.approx(0.25)


def test_budget_window_start():
    from yaab.governance.budget import Budget, _window_start

    now = 1_000_000.0
    assert _window_start(Budget(1.0, window="lifetime"), now) is None
    assert _window_start(Budget(1.0, window="day"), now) == pytest.approx(now - 86_400)
    assert _window_start(Budget(1.0, window="month"), now) == pytest.approx(now - 2_592_000)
