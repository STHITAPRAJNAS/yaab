"""Run history / inspect / fork over the checkpointer.history().

Time-travel debugging for durable runs: list a run's checkpoints, inspect the
state at any checkpoint (including a paused one), and fork-from-checkpoint to
re-run from step N with optional state edits — without disturbing the original
timeline.
"""

from __future__ import annotations

import pytest

from yaab import Flow, Runner
from yaab.graph.checkpoint import MemorySaver
from yaab.runs.history import RunHistory


def _counter_flow() -> Flow:
    def inc(state, ctx):
        return {"count": state.get("count", 0) + 1}

    return (
        Flow[None, int]("counter")
        .step("inc", fn=inc)
        .loop("inc", until="state.count >= 4", max_iterations=10)
        .start_at("inc")
        .returns("count")
    )


@pytest.mark.asyncio
async def test_history_lists_checkpoints():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)
    flow = _counter_flow()
    await runner.run(flow, "go", session_id="hist-1")

    history = RunHistory(saver)
    checkpoints = history.list("hist-1")
    assert len(checkpoints) >= 2
    # Each checkpoint exposes (step, state) so a UI can render the timeline.
    step, state = checkpoints[0]
    assert isinstance(step, int)
    assert "count" in state["state"] or "count" in state


@pytest.mark.asyncio
async def test_inspect_a_specific_checkpoint():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)
    await runner.run(_counter_flow(), "go", session_id="hist-2")

    history = RunHistory(saver)
    checkpoints = history.list("hist-2")
    # Inspect the state captured at the first checkpoint.
    first = history.inspect("hist-2", checkpoints[0][0])
    assert first is not None
    assert "state" in first


@pytest.mark.asyncio
async def test_inspect_paused_state():
    from yaab.governance.approvals import InMemoryApprovalStore
    from yaab.types import RunContext

    def gate(state, ctx: RunContext):
        d = ctx.pause_for({"need": "ok"})
        return {"d": d}

    flow = Flow[None, str]("p").step("gate", fn=gate).start_at("gate").returns("d")
    saver = MemorySaver()
    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=saver, approval_store=store)
    result = await runner.run(flow, "go", session_id="hist-3")
    assert result.paused

    history = RunHistory(saver)
    latest = history.latest("hist-3")
    assert latest is not None
    # The paused checkpoint carries the parked frontier so a reviewer can see it.
    assert latest.get("frontier") is not None


@pytest.mark.asyncio
async def test_fork_from_checkpoint_with_state_edit():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)
    await runner.run(_counter_flow(), "go", session_id="fork-src")

    history = RunHistory(saver)
    checkpoints = history.list("fork-src")
    # Fork from the first checkpoint into a new thread, editing the count so the
    # forked run starts from a different state than the original.
    fork_step = checkpoints[0][0]
    history.fork(
        "fork-src",
        fork_step,
        to_thread="fork-dst",
        edits={"count": 100},
    )
    # The fork landed a checkpoint on the destination thread with the edit.
    forked = history.latest("fork-dst")
    assert forked is not None
    assert forked["state"]["count"] == 100
    # The original timeline is untouched.
    assert history.latest("fork-src")["state"]["count"] != 100


@pytest.mark.asyncio
async def test_fork_and_rerun_from_edited_checkpoint():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)
    flow = _counter_flow()
    await runner.run(flow, "go", session_id="rerun-src")

    history = RunHistory(saver)
    checkpoints = history.list("rerun-src")
    # Fork from step 0 with count pre-seeded to 2, then re-run the flow on the
    # forked thread: it should continue from 2 and reach the until= threshold.
    history.fork("rerun-src", checkpoints[0][0], to_thread="rerun-dst", edits={"count": 2})
    resumed = await runner.run(flow, "go", session_id="rerun-dst", resume_from_checkpoint=True)
    assert resumed.output == 4
