"""Reuse tools from other ecosystems as native YAAB tools.

``from_langchain_tool`` / ``from_crewai_tool`` (and the auto-detecting
``adapt_tool``) wrap a foreign tool object as a native :class:`FunctionTool` —
its name/description/args carry over, and calling it runs the foreign tool. The
adapters duck-type the foreign interface, so neither library needs to be
installed to use (or test) them.
"""

from __future__ import annotations

import pytest

from yaab import Agent
from yaab.testing import FunctionModel
from yaab.tools.adapters import adapt_tool, from_crewai_tool, from_langchain_tool


class _FakeLangChainTool:
    """Mimics langchain_core.tools.BaseTool's public surface."""

    name = "search"
    description = "Search the web for a query."

    def invoke(self, args):
        # LangChain passes a dict for structured tools or a str for simple ones.
        q = args["query"] if isinstance(args, dict) else args
        return f"results for {q}"


class _FakeCrewAITool:
    """Mimics crewai.tools.BaseTool's public surface."""

    name = "calculator"
    description = "Evaluate a simple arithmetic expression."

    def run(self, **kwargs):
        return str(eval(kwargs["expression"], {"__builtins__": {}}))  # noqa: S307 - test only


@pytest.mark.asyncio
async def test_from_langchain_tool_wraps_name_and_runs():
    tool = from_langchain_tool(_FakeLangChainTool())
    assert tool.name == "search"
    assert "Search the web" in tool.description
    out = await tool.execute(None, query="cats")
    assert out == "results for cats"


@pytest.mark.asyncio
async def test_from_crewai_tool_wraps_name_and_runs():
    tool = from_crewai_tool(_FakeCrewAITool())
    assert tool.name == "calculator"
    out = await tool.execute(None, expression="2 + 3")
    assert out == "5"


def test_adapt_tool_autodetects():
    lc = adapt_tool(_FakeLangChainTool())
    cw = adapt_tool(_FakeCrewAITool())
    assert lc.name == "search"
    assert cw.name == "calculator"


def test_adapt_tool_rejects_unknown_object():
    with pytest.raises(TypeError):
        adapt_tool(object())


@pytest.mark.asyncio
async def test_adapted_tool_drops_into_an_agent():
    tool = from_langchain_tool(_FakeLangChainTool())

    def model_fn(messages):
        # First turn calls the adapted tool; second answers.
        from yaab.models.base import ModelResponse
        from yaab.types import ToolCall

        if not any(getattr(m, "role", None) == "tool" for m in messages):
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "dogs"})],
            )
        return ModelResponse(content="done")

    agent = Agent("a", model=FunctionModel(model_fn), tools=[tool])
    result = await agent.run("search for dogs")
    assert result.output == "done"
