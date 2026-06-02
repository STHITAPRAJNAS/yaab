"""Resume on the public ``Agent`` surface.

``Runner.run``/``run_stream`` already accept a ``resume_id`` for fault-tolerant
fast-path runs; these tests pin that the same knob is reachable from the
developer-facing :class:`Agent` (``run``/``run_sync``) as a pure pass-through —
so a crashed run resumes from its last checkpoint without re-requesting the
model turns already captured.
"""

from __future__ import annotations

from typing import Any

import pytest

from yaab import Agent
from yaab.graph.checkpoint import MemorySaver
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.tools.base import FunctionTool
from yaab.types import ToolCall, Usage


def ping_impl() -> str:
    """Return pong."""
    return "pong"


ping = FunctionTool(ping_impl, name="ping")


class CrashModel:
    """A scripted model that raises on a chosen ``complete`` call (1-indexed)."""

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
    return [
        ModelResponse(tool_calls=[ToolCall(name="ping", arguments={})], finish_reason="tool_calls"),
        "all done",
    ]


@pytest.mark.asyncio
async def test_agent_run_accepts_resume_id_passthrough():
    """``Agent.run(resume_id=...)`` reaches the checkpointer and resumes."""
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)

    crash = CrashModel(_tool_then_answer(), crash_on=2)
    agent = Agent("a", model=crash, tools=[ping], runner=runner)

    with pytest.raises(RuntimeError):
        await agent.run("go", resume_id="agent-job-1")

    # The tool round checkpointed before the crash.
    hist = saver.history("agent-job-1")
    assert any(s.get("step") == 0 for _, s in hist)
    assert not any(s.get("finished") for _, s in hist)

    # Resume with a fresh model that just answers — only ONE call (the
    # post-checkpoint continuation), the captured tool turn is not replayed.
    fixed = TestModel("recovered")
    agent2 = Agent("a", model=fixed, tools=[ping], runner=runner)
    result = await agent2.run("go", resume_id="agent-job-1")

    assert result.output == "recovered"
    assert len(fixed.calls) == 1


@pytest.mark.asyncio
async def test_agent_run_without_resume_id_is_unchanged():
    """Omitting ``resume_id`` keeps the zero-overhead path (nothing persisted)."""
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)
    agent = Agent("a", model=TestModel("done"), runner=runner)

    result = await agent.run("go")

    assert result.output == "done"
    assert saver._store == {}


def test_agent_run_sync_accepts_resume_id():
    """``run_sync`` forwards ``resume_id`` (finished marker => idempotent replay)."""
    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)

    agent = Agent("a", model=TestModel("the answer"), runner=runner)
    r1 = agent.run_sync("go", resume_id="sync-1")
    assert r1.output == "the answer"

    # Re-invoking the finished resume_id replays the persisted result with zero
    # model calls (a model that would explode is never reached).
    boom = CrashModel(["unused"], crash_on=1)
    agent2 = Agent("a", model=boom, runner=runner)
    r2 = agent2.run_sync("go", resume_id="sync-1")
    assert r2.output == "the answer"
    assert boom.calls == []
