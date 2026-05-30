"""Tests for the AG-UI compatibility middleware."""

from __future__ import annotations

import pytest

from yaab import Agent, tool
from yaab.agui import AGUIEventType, run_agui
from yaab.models.test_model import TestModel


@pytest.mark.asyncio
async def test_agui_basic_run_emits_lifecycle():
    agent = Agent("a", model=TestModel("hello world"))
    events = [e async for e in run_agui(agent, "hi")]
    types = [e["type"] for e in events]
    assert types[0] == AGUIEventType.RUN_STARTED
    assert types[-1] == AGUIEventType.RUN_FINISHED
    assert AGUIEventType.TEXT_MESSAGE_CONTENT in types
    # the final text is delivered
    content = next(e for e in events if e["type"] == AGUIEventType.TEXT_MESSAGE_CONTENT)
    assert content["delta"] == "hello world"


@pytest.mark.asyncio
async def test_agui_tool_call_events():
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    model = TestModel(custom_output="done", call_tools=["ping"])
    agent = Agent("a", model=model, tools=[ping])
    events = [e async for e in run_agui(agent, "go")]
    types = [e["type"] for e in events]
    assert AGUIEventType.TOOL_CALL_START in types
    assert AGUIEventType.TOOL_CALL_END in types
    assert AGUIEventType.TOOL_CALL_RESULT in types
    result = next(e for e in events if e["type"] == AGUIEventType.TOOL_CALL_RESULT)
    assert result["toolCallName"] == "ping"


@pytest.mark.asyncio
async def test_agui_reasoning_thinking_event():
    agent = Agent("a", model=TestModel("answer", reasoning="let me think"))
    events = [e async for e in run_agui(agent, "q")]
    thinking = [e for e in events if e["type"] == AGUIEventType.THINKING]
    assert thinking
    assert "think" in thinking[0]["delta"]


@pytest.mark.asyncio
async def test_agui_threads_and_run_ids_propagate():
    agent = Agent("a", model=TestModel("x"))
    events = [e async for e in run_agui(agent, "hi", thread_id="t1", run_id="r1")]
    started = events[0]
    assert started["threadId"] == "t1"
    assert started["runId"] == "r1"


def test_agui_sse_app_endpoint():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.agui import agui_sse_app

    agent = Agent("a", model=TestModel("streamed agui"))
    client = TestClient(agui_sse_app(agent))
    with client.stream("POST", "/agui", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "RUN_STARTED" in body
    assert "RUN_FINISHED" in body
    assert "streamed agui" in body


def test_agui_sse_app_accepts_messages_format():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.agui import agui_sse_app

    agent = Agent("a", model=TestModel("ok"))
    client = TestClient(agui_sse_app(agent))
    body = {"threadId": "t", "messages": [{"role": "user", "content": "hello"}]}
    with client.stream("POST", "/agui", json=body) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "RUN_STARTED" in text
