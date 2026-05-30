"""Tests for the built-in tool library."""

from __future__ import annotations

import pytest

from yaab.tools.builtin import (
    calculator,
    current_time,
    default_toolset,
    python_exec,
    web_search,
)
from yaab.tools.builtin.search import set_search_provider
from yaab.types import RunContext


@pytest.mark.asyncio
async def test_calculator_basic():
    assert await calculator.execute(RunContext(), expression="2 * (3 + 4) ** 2") == "98"


@pytest.mark.asyncio
async def test_calculator_rejects_names():
    out = await calculator.execute(RunContext(), expression="__import__('os')")
    assert out.startswith("error")


@pytest.mark.asyncio
async def test_calculator_div_zero():
    assert (await calculator.execute(RunContext(), expression="1/0")).startswith("error")


@pytest.mark.asyncio
async def test_current_time_iso():
    out = await current_time.execute(RunContext())
    assert "T" in out and out[:4].isdigit()


@pytest.mark.asyncio
async def test_python_exec_runs_and_captures_stdout():
    out = await python_exec.execute(RunContext(), code="print(sum(range(10)))")
    assert out == "45"


@pytest.mark.asyncio
async def test_python_exec_timeout():
    out = await python_exec.execute(RunContext(), code="while True: pass", timeout_seconds=0.5)
    assert "timeout" in out


@pytest.mark.asyncio
async def test_python_exec_error_reported():
    out = await python_exec.execute(RunContext(), code="raise ValueError('boom')")
    assert "error" in out and "boom" in out


@pytest.mark.asyncio
async def test_web_search_without_provider_is_graceful():
    set_search_provider(None)
    out = await web_search.execute(RunContext(), query="anything")
    assert "no web search provider" in out


@pytest.mark.asyncio
async def test_web_search_with_provider():
    async def fake(query, k):
        return [{"title": "T", "url": "http://x", "snippet": "hello"}]

    set_search_provider(fake)
    try:
        out = await web_search.execute(RunContext(), query="hi")
        assert "T" in out and "hello" in out
    finally:
        set_search_provider(None)


def test_default_toolset_is_read_only():
    names = {t.name for t in default_toolset()}
    assert "python_exec" not in names  # code exec excluded from safe default
    assert "calculator" in names
