"""Regression tests for issues surfaced by the live end-to-end harness.

Each test pins a bug that a real provider exposed but the offline suite missed:

* ``tool_choice="required"`` must force the first call only, then relax, so the
  loop can finalize instead of forcing a tool call forever (MaxStepsExceeded).
* ``parse_partial_json`` must tolerate Markdown code fences (Gemini/Claude wrap
  JSON despite a JSON-only instruction), so structured-output streaming yields.
* The Runner must thread ``identity`` (and an app scope) into a namespace-aware
  memory backend so scoped long-term memory is reachable from the Agent path.
"""

from __future__ import annotations

import pytest

from yaab import Agent, Runner, tool
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.streaming import parse_partial_json
from yaab.types import ToolCall


# --- 1. tool_choice="required" must not loop forever --------------------
@pytest.mark.asyncio
async def test_required_tool_choice_relaxes_after_first_call():
    calls = {"n": 0}

    @tool
    def lookup(q: str = "") -> str:
        """Look up an answer."""
        calls["n"] += 1
        return "42"

    # First model call: a forced tool call. Second call: a final answer.
    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="lookup", arguments={"q": "x"})],
                finish_reason="tool_calls",
            ),
            "the answer is 42",
        ]
    )
    agent = Agent("t", model=model, tools=[lookup], tool_choice="required", max_steps=8)
    result = await agent.run("go")
    assert result.output == "the answer is 42"
    assert calls["n"] == 1
    # The first call was forced; the second was relaxed to "auto".
    assert model.tool_choices[0] == "required"
    assert model.tool_choices[1] == "auto"


@pytest.mark.asyncio
async def test_pinned_function_tool_choice_relaxes():
    @tool
    def lookup(q: str = "") -> str:
        """Look up an answer."""
        return "42"

    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="lookup", arguments={})],
                finish_reason="tool_calls",
            ),
            "done",
        ]
    )
    agent = Agent("t", model=model, tools=[lookup], tool_choice="lookup", max_steps=5)
    result = await agent.run("go")
    assert result.output == "done"
    # A pinned-function dict on the first call, relaxed to "auto" after.
    assert isinstance(model.tool_choices[0], dict)
    assert model.tool_choices[0]["function"]["name"] == "lookup"
    assert model.tool_choices[1] == "auto"


# --- 2. parse_partial_json tolerates Markdown fences --------------------
def test_parse_partial_json_strips_code_fence():
    assert parse_partial_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_partial_json("```\n{\"a\": 1}\n```") == {"a": 1}
    # Partial (closing fence not yet streamed) still parses.
    assert parse_partial_json('```json\n{"a": 1, "b":') == {"a": 1}
    # Plain JSON unaffected.
    assert parse_partial_json('{"a": 1}') == {"a": 1}
    assert parse_partial_json("") is None


@pytest.mark.asyncio
async def test_structured_streaming_with_fenced_model():
    from pydantic import BaseModel

    class Profile(BaseModel):
        name: str
        age: int

    # A model that streams JSON wrapped in a Markdown fence (like Gemini).
    fenced = TestModel('```json\n{"name": "Alice", "age": 30}\n```')
    agent = Agent("p", model=fenced, output_type=Profile)
    seen = [p async for p in agent.stream_structured("make a profile", output_type=Profile)]
    assert seen, "no partials emitted from fenced JSON"
    assert isinstance(seen[-1], Profile)
    assert seen[-1].name == "Alice" and seen[-1].age == 30


# --- 3. Runner threads identity -> user_id into scoped memory -----------
@pytest.mark.asyncio
async def test_runner_threads_identity_into_scoped_memory():
    from yaab.memory.manager import MemoryManager

    mem = MemoryManager()
    # Stored under a specific user namespace (not "default").
    await mem.add("The deadline is March 15.", app_name="bank", user_id="alice")

    captured = {}

    def echo(messages):
        # Surface whatever system "Relevant memory" got injected.
        joined = "\n".join(m.content or "" for m in messages)
        captured["had_memory"] = "March 15" in joined
        return "ok"

    from yaab.models.test_model import FunctionModel

    runner = Runner(memory_service=mem, memory_app_name="bank")
    agent = Agent("a", model=FunctionModel(echo))
    # identity is threaded as user_id -> the scoped record is found.
    await runner.run(agent, "When is the deadline?", identity="alice")
    assert captured["had_memory"] is True

    # A different identity must NOT see alice's memory.
    captured.clear()
    await runner.run(agent, "When is the deadline?", identity="bob")
    assert captured.get("had_memory") is False
