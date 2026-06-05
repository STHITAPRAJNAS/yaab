"""The unified human decision surface: approve / deny / edit / respond.

Pins the canonical resume vocabulary built on top of the durable pause engine:

* one ``Decision`` model is the single value ``agent.run(resume=...)`` consumes;
* four verbs (``approve``/``deny``/``edit``/``respond``) share one body and accept
  a ``RunResult`` | ``Pending`` | bare ``approval_id`` string as their target;
* a target with several pendings requires ``approval_id=`` to disambiguate;
* every verb validates the payload BEFORE the store mutates, and ``decide`` is
  first-write-wins (a double approve resumes once).
"""

from __future__ import annotations

import pytest

from yaab import Agent
from yaab.governance import (
    ApprovalDecision,
    ApprovalRequest,
    InMemoryApprovalStore,
    ToolApprovalPlugin,
    approvals,
)
from yaab.governance.approvals_decide import Decision, DecisionValidationError
from yaab.graph.checkpoint import MemorySaver
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.tools.base import FunctionTool
from yaab.types import Pending, ToolCall

_EXECUTED: list[int] = []


def _wire(amount: int = 0, to: str = "") -> str:
    """Send money."""
    _EXECUTED.append(amount)
    return f"sent {amount} to {to}"


wire_transfer = FunctionTool(_wire, name="wire_transfer")


def _calls_wire_then_answers(amount: int = 100) -> TestModel:
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[
                    ToolCall(name="wire_transfer", arguments={"amount": amount, "to": "ACME"})
                ],
                finish_reason="tool_calls",
            ),
            "transfer complete",
        ]
    )


@pytest.fixture(autouse=True)
def _reset():
    _EXECUTED.clear()
    yield
    _EXECUTED.clear()


def _build(store, model):
    runner = Runner(
        run_checkpointer=MemorySaver(),
        plugins=[ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store)],
    )
    return Agent("banker", model=model, tools=[wire_transfer], runner=runner)


# --------------------------------------------------------------------------
# approve(RunResult) -> Decision; resume runs the tool.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_approve_runresult_returns_decision_and_resume_runs_tool():
    store = InMemoryApprovalStore()
    agent = _build(store, _calls_wire_then_answers())

    paused = await agent.run("wire 100 to ACME", resume_id="d1")
    assert paused.paused
    assert isinstance(paused.pending, list)
    assert len(paused.pending) == 1
    assert isinstance(paused.pending[0], Pending)
    assert paused.pending[0].kind == "approval"
    assert paused.pending[0].tool == "wire_transfer"

    decision = await approvals.approve(paused, by="alice", store=store)
    assert isinstance(decision, Decision)
    assert decision.verdict == "approved"
    assert decision.by == "alice"
    assert decision.approval_id == paused.pending[0].approval_id
    assert decision.resume_id  # copied from the store row

    final = await agent.run(resume=decision)
    assert not final.paused
    assert final.output == "transfer complete"
    assert _EXECUTED == [100]


# --------------------------------------------------------------------------
# A bare approval_id string is an accepted target (cross-process form).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_approve_accepts_bare_approval_id_string():
    store = InMemoryApprovalStore()
    agent = _build(store, _calls_wire_then_answers())
    paused = await agent.run("wire 100 to ACME", resume_id="d2")
    approval_id = paused.pending[0].approval_id

    decision = await approvals.approve(approval_id, by="alice", store=store)
    assert decision.approval_id == approval_id
    final = await agent.run(resume=decision)
    assert final.output == "transfer complete"


# --------------------------------------------------------------------------
# A single Pending is also a valid target.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_approve_accepts_single_pending():
    store = InMemoryApprovalStore()
    agent = _build(store, _calls_wire_then_answers())
    paused = await agent.run("wire 100 to ACME", resume_id="d3")
    decision = await approvals.approve(paused.pending[0], by="alice", store=store)
    final = await agent.run(resume=decision)
    assert final.output == "transfer complete"


# --------------------------------------------------------------------------
# deny feeds the reviewer's reason back to the model (not a fixed string).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deny_feeds_reason_to_model():
    store = InMemoryApprovalStore()
    agent = _build(store, _calls_wire_then_answers())
    paused = await agent.run("wire 100 to ACME", resume_id="d4")

    decision = await approvals.deny(paused, by="bob", reason="amount too large", store=store)
    assert decision.verdict == "denied"
    assert decision.reason == "amount too large"

    tool_results: list[str] = []
    final = await agent.run(resume=decision)
    for ev in final.events:
        if ev.type.value == "tool_result":
            tool_results.append(str(ev.payload.get("result")))
    assert _EXECUTED == []
    assert any("amount too large" in r for r in tool_results)


# --------------------------------------------------------------------------
# edit approves with corrected arguments; the tool runs with the new args.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_edit_runs_tool_with_corrected_arguments():
    store = InMemoryApprovalStore()
    agent = _build(store, _calls_wire_then_answers(amount=100))
    paused = await agent.run("wire 100 to ACME", resume_id="d5")

    decision = await approvals.edit(
        paused, by="alice", arguments={"amount": 25, "to": "ACME"}, store=store
    )
    assert decision.verdict == "approved"
    assert decision.arguments == {"amount": 25, "to": "ACME"}

    final = await agent.run(resume=decision)
    assert final.output == "transfer complete"
    # The tool ran with the EDITED amount, not the model's original 100.
    assert _EXECUTED == [25]


# --------------------------------------------------------------------------
# decide is first-write-wins: a second decision is a no-op on the SAME row.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_double_decide_is_idempotent_first_write_wins():
    store = InMemoryApprovalStore()
    agent = _build(store, _calls_wire_then_answers())
    paused = await agent.run("wire 100 to ACME", resume_id="d6")

    first = await approvals.approve(paused, by="alice", store=store)
    second = await approvals.deny(paused, by="mallory", reason="nope", store=store)
    # The second decision did NOT overwrite the first.
    assert first.verdict == "approved"
    assert second.verdict == "approved"  # returns the existing decided record
    assert second.by == "alice" or second.reason is None or second.verdict == "approved"
    row = await store.get(paused.pending[0].approval_id)
    assert row.decision is ApprovalDecision.APPROVED
    assert row.reviewer == "alice"


# --------------------------------------------------------------------------
# Multiple pendings require approval_id= to disambiguate.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multiple_pendings_require_approval_id():
    store = InMemoryApprovalStore()
    # Build a fake two-pending RunResult by creating two approval rows + Pendings.
    p1 = Pending(kind="approval", approval_id="ap_a", tool="x")
    p2 = Pending(kind="approval", approval_id="ap_b", tool="y")
    for ap_id in ("ap_a", "ap_b"):
        await store.create(
            ApprovalRequest(approval_id=ap_id, run_id="r", resume_id="r", agent="a", tool="t")
        )

    class _Res:
        pending = [p1, p2]

    with pytest.raises(ValueError, match="approval_id"):
        await approvals.approve(_Res(), by="alice", store=store)

    # Disambiguated, it works.
    d = await approvals.approve(_Res(), approval_id="ap_a", by="alice", store=store)
    assert d.approval_id == "ap_a"


# --------------------------------------------------------------------------
# An unknown approval id is a clear error.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_approval_id_raises():
    store = InMemoryApprovalStore()
    with pytest.raises(KeyError):
        await approvals.approve("ap_does_not_exist", by="alice", store=store)


# --------------------------------------------------------------------------
# respond validates the typed answer against the declared schema BEFORE mutate.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_respond_validates_answer_against_schema_before_mutate():
    store = InMemoryApprovalStore()
    req = ApprovalRequest(
        approval_id="ap_q",
        run_id="r",
        resume_id="r",
        agent="a",
        tool="ask_user",
        kind="question",
        prompt="How many?",
        answer_schema={"type": "integer", "minimum": 1},
    )
    await store.create(req)

    # A wrong-typed answer is rejected and NOTHING is written.
    with pytest.raises(DecisionValidationError):
        await approvals.respond("ap_q", by="user", answer="four", store=store)
    still_pending = await store.get("ap_q")
    assert still_pending.decision is ApprovalDecision.PENDING

    # A valid answer is accepted and recorded.
    decision = await approvals.respond("ap_q", by="user", answer=4, store=store)
    assert decision.answer == 4
    decided = await store.get("ap_q")
    assert decided.decision is ApprovalDecision.APPROVED
    assert decided.answer == 4
