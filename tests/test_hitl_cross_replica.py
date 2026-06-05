"""The headline durability guarantee: resume a paused run from a fresh process.

Two genuinely separate "processes" share only durable files (a SQLite checkpoint
db + a SQLite approval db) and a single string — the ``approval_id``. Process A
starts a run, it pauses, and A discards every in-memory object. Process B rebuilds
the agent/store *from the file paths only*, decides using just the ``approval_id``,
and resumes to completion. No live ``Agent``/``Pending``/``RunResult`` crosses the
boundary — this is what makes pause/resume pod-agnostic.
"""

from __future__ import annotations

import pytest

from yaab import Agent
from yaab.governance import SQLiteApprovalStore, ToolApprovalPlugin, approvals
from yaab.graph.checkpoint import SQLiteSaver
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.tools.base import FunctionTool

_LOG: list[str] = []


def _wire(amount: int = 0, to: str = "") -> str:
    """Send money."""
    _LOG.append(f"{amount}->{to}")
    return f"sent ${amount} to {to}"


wire_transfer = FunctionTool(_wire, name="wire_transfer")


def _model() -> TestModel:
    """Process A's model: request the guarded tool, then (if reached) answer."""
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[{"name": "wire_transfer", "arguments": {"amount": 5000, "to": "ACME"}}],
                finish_reason="tool_calls",
            ),
            "sent $5000 to ACME",
        ]
    )


def _answer_model() -> TestModel:
    """Process B's model: on resume the captured tool-call turn is NOT re-requested,
    so the resumed model only needs to take the post-tool continuation (answer)."""
    return TestModel("sent $5000 to ACME")


@pytest.fixture(autouse=True)
def _reset():
    _LOG.clear()
    yield
    _LOG.clear()


def _build_agent(approvals_db: str, runs_db: str, *, model: TestModel | None = None) -> Agent:
    """Reconstruct the SAME wired agent from file paths only (no shared objects)."""
    store = SQLiteApprovalStore(approvals_db)
    runner = Runner(run_checkpointer=SQLiteSaver(runs_db))
    runner.add_plugin(ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store))
    return Agent("banker", model=model or _model(), tools=[wire_transfer], runner=runner)


@pytest.mark.asyncio
async def test_resume_from_approval_id_only_across_fresh_process(tmp_path):
    approvals_db = str(tmp_path / "approvals.db")
    runs_db = str(tmp_path / "runs.db")

    # ---------- process A: start the run, it pauses, A "exits" ----------
    async def process_a() -> str:
        agent = _build_agent(approvals_db, runs_db)
        result = await agent.run("wire $5000 to ACME", resume_id="run-001")
        assert result.paused
        assert _LOG == []  # the guarded tool did NOT run
        return result.pending[0].approval_id  # the ONLY thing B needs

    approval_id = await process_a()
    assert approval_id

    # ---------- process B: fresh objects, only approval_id + db paths ----------
    async def process_b(ap_id: str) -> str:
        # A brand-new store instance over the SAME file (no in-memory carryover).
        fresh_store = SQLiteApprovalStore(approvals_db)
        # Decide using ONLY the approval_id string.
        decision = await approvals.approve(ap_id, by="alice", store=fresh_store)
        # A brand-new agent reconstructed from paths.
        agent_b = _build_agent(approvals_db, runs_db, model=_answer_model())
        result = await agent_b.run(resume=decision)
        assert not result.paused
        return result.output

    output = await process_b(approval_id)
    assert output == "sent $5000 to ACME"
    # The tool ran exactly once, in process B.
    assert _LOG == ["5000->ACME"]


@pytest.mark.asyncio
async def test_business_key_resume_across_processes(tmp_path):
    approvals_db = str(tmp_path / "approvals.db")
    runs_db = str(tmp_path / "runs.db")

    def _build_keyed(model: TestModel | None = None) -> Agent:
        store = SQLiteApprovalStore(approvals_db)
        runner = Runner(run_checkpointer=SQLiteSaver(runs_db))
        runner.add_plugin(
            ToolApprovalPlugin(
                tools=["wire_transfer"],
                mode="queue",
                store=store,
                correlation_key=lambda tool, args, ctx: f"customer:{args['to']}",
            )
        )
        return Agent("banker", model=model or _model(), tools=[wire_transfer], runner=runner)

    # Process A pauses with a business key set.
    agent_a = _build_keyed()
    paused = await agent_a.run("wire $5000 to ACME", resume_id="run-key-1")
    assert paused.paused
    assert paused.pending[0].correlation_key == "customer:ACME"

    # Process B finds the pending by business key alone (no approval_id, no run id).
    fresh_store = SQLiteApprovalStore(approvals_db)
    found = await fresh_store.list_by_key("customer:ACME")
    assert len(found) == 1
    decision = await approvals.approve(found[0].approval_id, by="alice", store=fresh_store)

    agent_b = _build_keyed(_answer_model())
    result = await agent_b.run(resume=decision)
    assert result.output == "sent $5000 to ACME"
    assert _LOG == ["5000->ACME"]
