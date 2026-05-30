"""Tests for the tracing toggle, trace redaction, and pagination (Tier 2b)."""

from __future__ import annotations

import pytest

from yaab import Agent, SessionManager
from yaab.models.test_model import TestModel
from yaab.observability import (
    genai_span,
    set_trace_redactor,
    set_tracing_enabled,
    tracing_enabled,
)


def test_tracing_toggle():
    original = tracing_enabled()
    set_tracing_enabled(False)
    # When disabled, genai_span yields None regardless of OTel availability.
    with genai_span("chat", {"k": "v"}) as span:
        assert span is None
    set_tracing_enabled(original)


def test_redactor_registration_roundtrip():
    seen = {}

    def redactor(key, value):
        seen[key] = value
        return "REDACTED" if key == "secret" else value

    set_trace_redactor(redactor)
    try:
        # With OTel absent the span is None, but the redactor API must still
        # register/unregister cleanly without error.
        with genai_span("chat", {"secret": "abc"}):
            pass
    finally:
        set_trace_redactor(None)


@pytest.mark.asyncio
async def test_agent_runs_with_tracing_disabled():
    set_tracing_enabled(False)
    try:
        agent = Agent("a", model=TestModel("ok"))
        result = await agent.run("hi")
        assert result.output == "ok"
    finally:
        set_tracing_enabled(True)


@pytest.mark.asyncio
async def test_session_pagination():
    mgr = SessionManager()
    for _ in range(5):
        await mgr.create_session(app_name="app", user_id="u")
    page1 = await mgr.list_sessions(app_name="app", user_id="u", limit=2, offset=0)
    page2 = await mgr.list_sessions(app_name="app", user_id="u", limit=2, offset=2)
    page3 = await mgr.list_sessions(app_name="app", user_id="u", limit=2, offset=4)
    assert len(page1) == 2 and len(page2) == 2 and len(page3) == 1
    # Pages are disjoint and cover all five.
    assert len(set(page1 + page2 + page3)) == 5
