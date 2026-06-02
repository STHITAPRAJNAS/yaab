"""The configurable durability knob for fast-path runs.

A run can persist progress after every completed step (``checkpoint_mode="step"``,
the default — fault-tolerant from any point) or only write the terminal marker
(``checkpoint_mode="final"`` — cheap for short runs, still idempotent on a
re-invoke). These tests pin both modes and confirm the default is bit-for-bit
the existing per-step behavior.
"""

from __future__ import annotations

import pytest

from yaab import Agent
from yaab.graph.checkpoint import MemorySaver
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.tools.base import FunctionTool


def ping_impl() -> str:
    """Return pong."""
    return "pong"


ping = FunctionTool(ping_impl, name="ping")


@pytest.mark.asyncio
async def test_default_mode_is_step_and_checkpoints_each_step():
    """Default ``checkpoint_mode`` keeps today's per-step checkpoint behavior."""
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)
    assert runner.checkpoint_mode == "step"

    agent = Agent(
        "a", model=TestModel(custom_output="done", call_tools=["ping"]), tools=[ping], runner=runner
    )
    await runner.run(agent, "go", resume_id="job-step")

    hist = saver.history("job-step")
    steps = [s.get("step") for _, s in hist]
    # The tool round (step 0) was checkpointed AND a terminal marker exists.
    assert 0 in steps
    assert any(s.get("finished") for _, s in hist)


@pytest.mark.asyncio
async def test_final_mode_writes_only_the_terminal_marker():
    """``checkpoint_mode="final"`` skips per-step writes; only the marker lands."""
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver, checkpoint_mode="final")

    agent = Agent(
        "a", model=TestModel(custom_output="done", call_tools=["ping"]), tools=[ping], runner=runner
    )
    await runner.run(agent, "go", resume_id="job-final")

    hist = saver.history("job-final")
    # Exactly one entry: the terminal finished marker (no intermediate steps).
    assert len(hist) == 1
    _, terminal = hist[0]
    assert terminal.get("finished") is True


@pytest.mark.asyncio
async def test_final_mode_finished_resume_is_idempotent():
    """A run checkpointed in ``final`` mode still replays idempotently."""
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver, checkpoint_mode="final")

    first = TestModel("the answer")
    agent = Agent("a", model=first, tools=[ping], runner=runner)
    r1 = await runner.run(agent, "go", resume_id="final-done")
    assert r1.output == "the answer"
    assert len(first.calls) == 1

    # Re-invoke: the persisted result returns, the model is never called.
    second = TestModel("SHOULD-NOT-BE-USED")
    agent2 = Agent("a", model=second, tools=[ping], runner=runner)
    r2 = await runner.run(agent2, "go", resume_id="final-done")
    assert r2.output == "the answer"
    assert second.calls == []
