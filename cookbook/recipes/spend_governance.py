"""Recipe: spend governance — per-identity / per-tenant budget caps.

``SpendGovernancePlugin`` records each model call's cost to a durable
``SpendStore`` and blocks a run whose key is already over budget.

    python -m cookbook.recipes.spend_governance
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent, Runner
from yaab.governance.budget import Budget, InMemorySpendStore, SpendGovernancePlugin
from yaab.models.test_model import TestModel
from yaab.plugins.builtins import BudgetExceeded


async def run() -> dict:
    store = InMemorySpendStore()
    await store.record("id:alice", 5.0, at=1.0)  # alice has already spent over her $1 cap
    plugin = SpendGovernancePlugin(store, {"id:alice": Budget(1.0, window="lifetime")})
    runner = Runner(plugins=[plugin])
    agent: Agent = Agent("assistant", model=TestModel(custom_output="hello"))

    blocked = False
    try:
        await runner.run(agent, "hi", identity="alice")
    except BudgetExceeded:
        blocked = True
    expect(blocked, "alice should be blocked: she is over budget")

    # An identity with no budget cap runs normally.
    ok = await runner.run(agent, "hi", identity="bob")
    expect(str(ok.output) == "hello", "bob is unbudgeted and should run")
    return {"alice_blocked": blocked, "bob_answer": str(ok.output)}


if __name__ == "__main__":
    print(asyncio.run(run()))
