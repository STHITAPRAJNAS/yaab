"""Resumable / fault-tolerant fast-path runs.

The durable graph already resumes by ``thread_id``; the model-driven fast path
(:meth:`Runner.run_stream`) used to lose everything on a crash. These tests pin
the new checkpoint/resume contract:

* with a ``run_checkpointer`` and a ``resume_id``, the runner checkpoints loop
  progress after every completed step (model response + executed tools);
* a crashed run leaves a checkpoint that a re-run with the same ``resume_id``
  picks up from — WITHOUT re-requesting the model responses already captured;
* on success a terminal ``{finished: true}`` marker is written so a finished
  ``resume_id`` re-invoked returns the persisted result with zero model calls;
* a run with no ``resume_id`` (or no checkpointer) is byte-for-byte the old
  behavior — zero overhead, nothing persisted.
"""

from __future__ import annotations

from typing import Any

import pytest

# A module-level Agent stand-in is overkill; build a tiny one per test via the
# real Agent so tool wiring/output coercion is exercised end to end.
from yaab import Agent, tool
from yaab.graph.checkpoint import MemorySaver
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.types import EventType, ToolCall, Usage


class CrashModel:
    """A scripted model that raises on a chosen ``complete`` call (1-indexed).

    Records every set of messages it is asked to complete in :attr:`calls` so a
    test can assert exactly which turns reached the model (i.e. that a resumed
    run does not re-request a checkpointed turn).
    """

    __test__ = False

    def __init__(self, responses: list[ModelResponse | str], *, crash_on: int) -> None:
        self.name = "crash"
        self.responses = responses
        self.crash_on = crash_on
        self._index = 0
        self.calls: list[list[Any]] = []

    async def complete(
        self,
        messages: list[Any],
        *,
        tools: Any = None,
        output_schema: Any = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self._index += 1
        if self._index == self.crash_on:
            raise RuntimeError("simulated provider crash")
        self.calls.append(list(messages))
        item = self.responses[min(self._index - 1, len(self.responses) - 1)]
        usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        if isinstance(item, str):
            return ModelResponse(content=item, usage=usage, model="crash")
        item.usage = usage
        return item


def _tool_then_answer() -> list[ModelResponse | str]:
    """Response 1 calls ``ping``; response 2 is the final answer."""
    return [
        ModelResponse(tool_calls=[ToolCall(name="ping", arguments={})], finish_reason="tool_calls"),
        "all done",
    ]


@tool
def ping() -> str:
    """Return pong."""
    return "pong"


# --------------------------------------------------------------------------
# 1. No resume_id / no checkpointer -> exactly current behavior, no overhead.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_resume_id_is_unchanged_and_persists_nothing():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)
    agent = Agent("a", model=TestModel(custom_output="done", call_tools=["ping"]), tools=[ping])

    result = await runner.run(agent, "go")  # no resume_id

    assert result.output == "done"
    # Nothing was checkpointed because no resume_id was supplied.
    assert saver._store == {}


@pytest.mark.asyncio
async def test_no_checkpointer_ignores_resume_id():
    runner = Runner()  # no checkpointer at all
    agent = Agent("a", model=TestModel("done"))
    # resume_id is accepted but inert without a checkpointer.
    result = await runner.run(agent, "go", resume_id="abc")
    assert result.output == "done"


# --------------------------------------------------------------------------
# 2. Checkpoint after every completed step.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_checkpoints_each_step_and_writes_terminal_marker():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)
    agent = Agent("a", model=TestModel(custom_output="done", call_tools=["ping"]), tools=[ping])

    await runner.run(agent, "go", resume_id="job-1")

    history = saver.history("job-1")
    # Step 0 (tool round) + step 1 (final answer) + terminal marker.
    assert len(history) >= 2
    steps = [state.get("step") for _, state in history]
    assert 0 in steps  # the tool round was checkpointed
    # The last entry is the terminal marker.
    _, terminal = history[-1]
    assert terminal.get("finished") is True


# --------------------------------------------------------------------------
# 3. Crash mid-run, then resume from the checkpoint without re-requesting the
#    already-captured model turn.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_continues_without_replaying_first_model_turn():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)

    # Crash on the 2nd model call: step 0 (tool round) completes & checkpoints,
    # then the model raises before producing the final answer.
    crash = CrashModel(_tool_then_answer(), crash_on=2)
    agent = Agent("a", model=crash, tools=[ping])

    with pytest.raises(RuntimeError):
        await runner.run(agent, "go", resume_id="job-2")

    # A checkpoint for step 0 exists; no terminal marker yet.
    hist = saver.history("job-2")
    assert hist, "expected a checkpoint after the first completed step"
    assert not any(s.get("finished") for _, s in hist)
    assert any(s.get("step") == 0 for _, s in hist)

    # Re-run with the SAME resume_id and a fixed model that just answers.
    fixed = TestModel("recovered answer")
    agent2 = Agent("a", model=fixed, tools=[ping])
    result2 = await runner.run(agent2, "go", resume_id="job-2")

    assert result2.output == "recovered answer"
    # The fixed model received exactly ONE call — the post-checkpoint
    # continuation. The tool-calling turn was NOT re-requested.
    assert len(fixed.calls) == 1
    # And that single call already carried the restored history: the assistant
    # tool-call message and the tool result from step 0.
    restored = fixed.calls[0]
    roles = [m.role.value for m in restored]
    assert "tool" in roles  # the ping result was rehydrated
    # A terminal marker is now written so the job never re-runs.
    assert any(s.get("finished") for _, s in saver.history("job-2"))


def _is_error_result(result: Any) -> bool:
    # run() raises on error normally; here we tolerate either a raised error
    # captured by the test harness or an error-bearing result. The crash path
    # raises, so this run() call is wrapped below.
    return False


# The crash run actually raises out of run(); wrap it to assert and persist.
@pytest.mark.asyncio
async def test_resume_full_flow():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)

    crash = CrashModel(_tool_then_answer(), crash_on=2)
    agent = Agent("a", model=crash, tools=[ping])

    with pytest.raises(RuntimeError):
        await runner.run(agent, "go", resume_id="job-3")

    hist = saver.history("job-3")
    assert any(s.get("step") == 0 for _, s in hist)
    assert not any(s.get("finished") for _, s in hist)

    fixed = TestModel("recovered answer")
    agent2 = Agent("a", model=fixed, tools=[ping])
    result2 = await runner.run(agent2, "go", resume_id="job-3")

    assert result2.output == "recovered answer"
    assert len(fixed.calls) == 1
    assert any(s.get("finished") for _, s in saver.history("job-3"))


# --------------------------------------------------------------------------
# 4. A finished resume_id is idempotent: re-invoking returns the persisted
#    result with ZERO model calls.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_finished_resume_id_is_idempotent_no_model_calls():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)

    first_model = TestModel("the answer")
    agent = Agent("a", model=first_model, tools=[ping])
    r1 = await runner.run(agent, "go", resume_id="done-1")
    assert r1.output == "the answer"
    assert len(first_model.calls) == 1

    # Re-invoke with the same resume_id but a model that would explode if called.
    boom = CrashModel(["unused"], crash_on=1)  # crashes on first call
    agent2 = Agent("a", model=boom, tools=[ping])
    r2 = await runner.run(agent2, "go", resume_id="done-1")

    # Idempotent: persisted result returned, model never invoked.
    assert r2.output == "the answer"
    assert boom.calls == []


# --------------------------------------------------------------------------
# 5. run_stream resume yields a normal terminal RUN_END with the recovered
#    output (the streaming entry point, not just run()).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_stream_resume_emits_run_end():
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)

    crash = CrashModel(_tool_then_answer(), crash_on=2)
    agent = Agent("a", model=crash, tools=[ping])
    # Drain the stream; the crash surfaces as an ERROR event (run_stream does
    # not raise — it emits a terminal ERROR).
    saw_error = False
    async for ev in runner.run_stream(agent, "go", resume_id="job-4"):
        if ev.type is EventType.ERROR:
            saw_error = True
    assert saw_error

    fixed = TestModel("streamed recovery")
    agent2 = Agent("a", model=fixed, tools=[ping])
    final_output = None
    async for ev in runner.run_stream(agent2, "go", resume_id="job-4"):
        if ev.type is EventType.RUN_END:
            final_output = ev.payload["result"].output
    assert final_output == "streamed recovery"
    assert len(fixed.calls) == 1
