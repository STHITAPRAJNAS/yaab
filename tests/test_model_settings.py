"""Model-settings passthrough (out-of-the-box provider kwargs).

Agent(model_settings={...}) forwards arbitrary provider kwargs (temperature,
top_p, seed, max_tokens, reasoning_effort, stop, extra_body, ...) straight to the
model provider on both complete() and stream(), so anything LiteLLM / the
underlying model supports is reachable without subclassing.
"""

from __future__ import annotations

import pytest

from yaab import Agent, tool
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.types import ToolCall


@pytest.mark.asyncio
async def test_model_settings_forwarded_on_complete():
    model = TestModel("ok")
    agent = Agent(
        "a",
        model=model,
        model_settings={"temperature": 0.2, "seed": 42, "max_tokens": 128},
    )
    await agent.run("hi")
    assert model.call_kwargs, "no kwargs captured"
    kw = model.call_kwargs[0]
    assert kw.get("temperature") == 0.2
    assert kw.get("seed") == 42
    assert kw.get("max_tokens") == 128


@pytest.mark.asyncio
async def test_model_settings_forwarded_through_tool_loop():
    @tool
    def ping() -> str:
        """ping"""
        return "pong"

    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="ping", arguments={})], finish_reason="tool_calls"
            ),
            "done",
        ]
    )
    agent = Agent("a", model=model, tools=[ping], model_settings={"top_p": 0.9})
    await agent.run("go")
    # Forwarded on every model call in the loop (the tool turn and the final turn).
    assert all(kw.get("top_p") == 0.9 for kw in model.call_kwargs)
    assert len(model.call_kwargs) == 2


@pytest.mark.asyncio
async def test_model_settings_forwarded_on_stream():
    model = TestModel("hello there")
    agent = Agent("a", model=model, model_settings={"temperature": 0.0})
    _ = [c async for c in agent.stream("hi")]
    assert model.call_kwargs and model.call_kwargs[0].get("temperature") == 0.0


@pytest.mark.asyncio
async def test_no_model_settings_is_clean():
    model = TestModel("ok")
    agent = Agent("a", model=model)
    await agent.run("hi")
    # No spurious kwargs injected when model_settings is unset.
    assert model.call_kwargs == [{}]
