"""Round-trip AG-UI: emit state to the frontend and accept input/HITL back.

The AG-UI bridge was output-only. Now it emits STATE_SNAPSHOT at run start and a
STATE_DELTA whenever a step writes shared state, surfaces an INPUT_REQUIRED event
when a run pauses for human sign-off, and `resume_agui` streams the continuation
once the frontend sends the decision back — so a frontend can both *see* state
and *drive* the run.
"""

from __future__ import annotations

import pytest

from yaab import Agent, Runner, tool
from yaab.agui import AGUIEventType, resume_agui, run_agui
from yaab.governance import approvals
from yaab.governance.approval import ToolApprovalPlugin
from yaab.governance.approvals import InMemoryApprovalStore
from yaab.graph.checkpoint import MemorySaver
from yaab.models.base import ModelResponse
from yaab.testing import FunctionModel, TestModel
from yaab.types import ToolCall


async def _collect(stream):
    return [e async for e in stream]


@pytest.mark.asyncio
async def test_emits_state_snapshot_at_start():
    agent = Agent("a", model=TestModel("hi"))
    events = await _collect(run_agui(agent, "go"))
    types = [e["type"] for e in events]
    assert AGUIEventType.STATE_SNAPSHOT in types
    # The snapshot comes before any text content.
    assert types.index(AGUIEventType.STATE_SNAPSHOT) < types.index(AGUIEventType.RUN_FINISHED)


@pytest.mark.asyncio
async def test_emits_state_delta_on_write():
    writer = Agent("w", model=TestModel("classified"), writes="intent")
    events = await _collect(run_agui(writer, "go"))
    deltas = [e for e in events if e["type"] == AGUIEventType.STATE_DELTA]
    assert deltas
    assert any("intent" in str(d.get("delta", "")) or d.get("key") == "intent" for d in deltas)


@pytest.mark.asyncio
async def test_input_required_on_pause_then_resume():
    calls = {"n": 0}

    @tool
    def wire(amount: int) -> str:
        calls["n"] += 1
        return f"sent {amount}"

    n = {"i": 0}

    def model_fn(messages):
        n["i"] += 1
        if n["i"] == 1:
            return ModelResponse(
                content="", tool_calls=[ToolCall(id="c1", name="wire", arguments={"amount": 5})]
            )
        return ModelResponse(content="done")

    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    runner.add_plugin(ToolApprovalPlugin(tools=["wire"], mode="queue", store=store))
    agent = Agent("banker", model=FunctionModel(model_fn), tools=[wire], runner=runner)

    # First stream pauses and emits INPUT_REQUIRED.
    events = await _collect(run_agui(agent, "wire 5", runner=runner))
    input_events = [e for e in events if e["type"] == AGUIEventType.INPUT_REQUIRED]
    assert input_events
    approval_id = input_events[0]["approvalId"]
    assert calls["n"] == 0  # tool not run yet

    # Frontend approves; resume_agui streams the continuation.
    decision = await approvals.approve(approval_id, by="user", store=store)
    resumed = await _collect(resume_agui(agent, decision, runner=runner))
    assert calls["n"] == 1
    assert any(e["type"] == AGUIEventType.RUN_FINISHED for e in resumed)
