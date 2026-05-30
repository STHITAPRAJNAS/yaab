"""Tests for context-window management strategies."""

from __future__ import annotations

import pytest

from yaab import Agent, SummarizeHistory, TruncateMessages
from yaab.context import KeepAll, approx_tokens
from yaab.models.test_model import TestModel
from yaab.types import Message, Role


def _history(n: int) -> list[Message]:
    msgs = [Message(role=Role.SYSTEM, content="You are helpful.")]
    for i in range(n):
        msgs.append(Message(role=Role.USER, content=f"q{i}"))
        msgs.append(Message(role=Role.ASSISTANT, content=f"a{i}"))
    return msgs


@pytest.mark.asyncio
async def test_keep_all_noop():
    msgs = _history(10)
    out = await KeepAll().apply(msgs)
    assert out == msgs


@pytest.mark.asyncio
async def test_truncate_keeps_system_and_recent():
    msgs = _history(20)  # 1 system + 40 turns
    out = await TruncateMessages(max_messages=6).apply(msgs)
    assert out[0].role is Role.SYSTEM
    assert len(out) == 1 + 6
    # the most recent turns survive
    assert out[-1].content == "a19"


@pytest.mark.asyncio
async def test_summarize_below_budget_is_noop():
    msgs = _history(2)
    out = await SummarizeHistory(max_tokens=10_000).apply(msgs, model=TestModel("sum"))
    assert out == msgs


@pytest.mark.asyncio
async def test_summarize_folds_old_history():
    msgs = _history(50)  # large
    strat = SummarizeHistory(max_tokens=10, keep_recent=4)
    out = await strat.apply(msgs, model=TestModel("SUMMARY"))
    # system + summary + last 4
    assert out[0].role is Role.SYSTEM
    assert any("SUMMARY" in (m.content or "") for m in out)
    assert len(out) <= 1 + 1 + 4
    assert out[-1].content == "a49"


@pytest.mark.asyncio
async def test_summarize_without_model_falls_back_to_truncate():
    msgs = _history(50)
    out = await SummarizeHistory(max_tokens=10, keep_recent=4).apply(msgs, model=None)
    assert len(out) <= 1 + 4


def test_approx_tokens():
    msgs = [Message(role=Role.USER, content="a" * 400)]
    assert approx_tokens(msgs) == 100


@pytest.mark.asyncio
async def test_agent_uses_context_strategy():
    # An agent with a truncating strategy still produces output.
    agent = Agent("a", model=TestModel("ok"), context_strategy=TruncateMessages(2))
    result = await agent.run("hello")
    assert result.output == "ok"
