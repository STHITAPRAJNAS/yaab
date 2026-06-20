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


@pytest.mark.asyncio
async def test_sqlite_ledger_is_shared_across_instances(tmp_path):
    # Multi-pod consistency: spend recorded by one process instance is visible to
    # another opened on the same database file — the floor for cross-pod caps.
    from yaab.governance.budget import SQLiteSpendStore

    path = str(tmp_path / "spend.db")
    pod_a = SQLiteSpendStore(path)
    await pod_a.record("tenant:acme", 0.7, at=100.0)
    await pod_a.record("tenant:acme", 0.4, at=200.0)

    pod_b = SQLiteSpendStore(path)  # a "second pod" over the same store
    assert await pod_b.total("tenant:acme") == pytest.approx(1.1)
    assert await pod_b.total("tenant:acme", since=150.0) == pytest.approx(0.4)


def test_postgres_store_requires_driver_when_absent():
    try:
        import psycopg  # noqa: F401

        pytest.skip("psycopg is installed")
    except ImportError:
        pass
    from yaab.governance.budget import PostgresSpendStore

    with pytest.raises(RuntimeError, match="psycopg"):
        PostgresSpendStore("postgresql://x")


def test_budget_window_start():
    from yaab.governance.budget import Budget, _window_start

    now = 1_000_000.0
    assert _window_start(Budget(1.0, window="lifetime"), now) is None
    assert _window_start(Budget(1.0, window="day"), now) == pytest.approx(now - 86_400)
    assert _window_start(Budget(1.0, window="month"), now) == pytest.approx(now - 2_592_000)


def _response(cost):
    from yaab.models.base import ModelResponse
    from yaab.types import Usage

    return ModelResponse(content="ok", usage=Usage(cost_usd=cost, requests=1))


def _ctx(identity):
    from yaab.types import RunContext

    return RunContext(identity=identity)


@pytest.mark.asyncio
async def test_after_model_records_cost_then_before_run_blocks():
    from yaab.governance.budget import Budget, InMemorySpendStore, SpendGovernancePlugin
    from yaab.plugins.builtins import BudgetExceeded

    store = InMemorySpendStore()
    plugin = SpendGovernancePlugin(store, {"id:alice": Budget(1.0, window="lifetime")})

    # Under budget: before_run allows, after_model records each call's cost.
    await plugin.before_run(_ctx("alice"), "a", "hi")
    await plugin.after_model(_ctx("alice"), "a", _response(0.6))
    await plugin.before_run(_ctx("alice"), "a", "hi")  # $0.60 < $1.00, still allowed
    await plugin.after_model(_ctx("alice"), "a", _response(0.6))  # total $1.20

    # Now over budget -> before_run blocks the next run.
    with pytest.raises(BudgetExceeded):
        await plugin.before_run(_ctx("alice"), "a", "hi")
    # A different identity is unaffected.
    await plugin.before_run(_ctx("bob"), "a", "hi")


@pytest.mark.asyncio
async def test_tenant_cap_pools_across_identities():
    from yaab.governance.budget import Budget, InMemorySpendStore, SpendGovernancePlugin
    from yaab.plugins.builtins import BudgetExceeded

    store = InMemorySpendStore()
    plugin = SpendGovernancePlugin(
        store,
        {"tenant:acme": Budget(1.0, window="lifetime")},
        tenant_of=lambda identity: "acme" if identity in ("alice", "bob") else None,
    )
    await plugin.after_model(_ctx("alice"), "a", _response(0.6))  # tenant acme $0.60
    await plugin.after_model(_ctx("bob"), "a", _response(0.6))  # tenant acme $1.20 (pooled)
    with pytest.raises(BudgetExceeded):
        await plugin.before_run(_ctx("alice"), "a", "hi")  # tenant over cap


@pytest.mark.asyncio
async def test_rolling_window_expires_old_spend():
    from yaab.governance.budget import Budget, InMemorySpendStore, SpendGovernancePlugin
    from yaab.plugins.builtins import BudgetExceeded

    clock = {"now": 1_000_000.0}
    store = InMemorySpendStore()
    plugin = SpendGovernancePlugin(
        store, {"id:alice": Budget(1.0, window="day")}, clock=lambda: clock["now"]
    )
    await plugin.after_model(_ctx("alice"), "a", _response(0.8))  # $0.80 today
    with pytest.raises(BudgetExceeded):
        # Re-check immediately would still be under; push it over first.
        await plugin.after_model(_ctx("alice"), "a", _response(0.8))  # $1.60 in-window
        await plugin.before_run(_ctx("alice"), "a", "hi")
    # Two days later the old spend is outside the 1-day window -> allowed again.
    clock["now"] += 2 * 86_400
    await plugin.before_run(_ctx("alice"), "a", "hi")


@pytest.mark.asyncio
async def test_plugin_hook_fires_in_a_real_run():
    # Pre-seed the store over budget; a real run must be blocked by before_run.
    from yaab import Agent, Runner
    from yaab.governance.budget import Budget, InMemorySpendStore, SpendGovernancePlugin
    from yaab.models.test_model import TestModel
    from yaab.plugins.builtins import BudgetExceeded

    store = InMemorySpendStore()
    await store.record("id:alice", 5.0, at=1.0)  # already over a $1 cap
    plugin = SpendGovernancePlugin(store, {"id:alice": Budget(1.0)})
    runner = Runner(plugins=[plugin])
    agent = Agent("a", model=TestModel(custom_output="hi"))

    with pytest.raises(BudgetExceeded):
        await runner.run(agent, "hello", identity="alice")
    # An unbudgeted identity runs normally.
    result = await runner.run(agent, "hello", identity="bob")
    assert result.output == "hi"
