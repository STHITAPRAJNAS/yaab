"""Tests for sub-agent delegation (sub_agents + transfer_to_agent).

These exercise the framework-managed handoff: a parent declares ``sub_agents``,
the framework injects a ``transfer_to_agent`` tool, the model picks a sub-agent
by name, and the sub-agent's answer becomes the run's answer. All TestModel-
driven — no network.
"""

from __future__ import annotations

import pytest

from yaab import Agent, EventType
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.types import ToolCall


def _transfer_model(agent_name: str) -> TestModel:
    """A parent model that calls transfer_to_agent(agent_name) then nothing else."""
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[
                    ToolCall(name="transfer_to_agent", arguments={"agent_name": agent_name})
                ],
                finish_reason="tool_calls",
            ),
            # A fallback final answer if the loop ever continues (it shouldn't on
            # a successful transfer).
            ModelResponse(content="parent fallback"),
        ]
    )


def _sub(name: str, answer: str, **kwargs) -> Agent:
    return Agent(name, model=TestModel(answer), **kwargs)


# --------------------------------------------------------------------------
# description defaults


def test_description_defaults_from_instructions_first_line():
    a = Agent(
        "a", model=TestModel("x"), instructions="Handle billing questions.\nMore detail here."
    )
    assert a.description == "Handle billing questions."


def test_description_empty_when_instructions_callable():
    a = Agent("a", model=TestModel("x"), instructions=lambda ctx: "dynamic")
    assert a.description == ""


def test_explicit_description_wins():
    a = Agent("a", model=TestModel("x"), instructions="ignored line", description="the real desc")
    assert a.description == "the real desc"


# --------------------------------------------------------------------------
# tool injection


def test_no_sub_agents_means_no_transfer_tool():
    a = Agent("plain", model=TestModel("x"))
    assert all(t.name != "transfer_to_agent" for t in a.tools)
    assert a.sub_agents == []


def test_sub_agents_injects_transfer_tool_with_descriptions_in_docstring():
    billing = _sub("billing", "b", description="Handles invoices and refunds.")
    tech = _sub("tech", "t", description="Handles outages and bugs.")
    parent = Agent("router", model=TestModel("x"), sub_agents=[billing, tech])

    transfer = next(t for t in parent.tools if t.name == "transfer_to_agent")
    # The docstring/description must list each sub-agent as 'name: description'
    # so the LLM can route.
    assert "billing: Handles invoices and refunds." in transfer.description
    assert "tech: Handles outages and bugs." in transfer.description
    # Schema must accept an agent_name string argument.
    schema = transfer.schema()
    props = schema["function"]["parameters"]["properties"]
    assert "agent_name" in props


# --------------------------------------------------------------------------
# delegation


@pytest.mark.asyncio
async def test_transfer_delegates_to_named_sub_agent():
    billing = _sub("billing", "billing answer", description="Invoices and refunds.")
    tech = _sub("tech", "tech answer", description="Outages and bugs.")
    parent = Agent("router", model=_transfer_model("billing"), sub_agents=[billing, tech])
    result = await parent.run("I was double charged")
    # The sub-agent's output becomes the parent run's final output.
    assert result.output == "billing answer"


@pytest.mark.asyncio
async def test_transfer_emits_agent_transfer_event():
    billing = _sub("billing", "billing answer", description="Invoices.")
    parent = Agent("router", model=_transfer_model("billing"), sub_agents=[billing])
    result = await parent.run("help")
    transfer_events = [e for e in result.events if e.type is EventType.AGENT_TRANSFER]
    assert len(transfer_events) == 1
    assert transfer_events[0].payload.get("to") == "billing"


@pytest.mark.asyncio
async def test_transfer_passes_original_prompt_to_sub_agent():
    seen: dict[str, str] = {}

    class RecordingModel(TestModel):
        async def complete(self, messages, **kwargs):
            # The last user message reaching the sub-agent must be the ORIGINAL prompt.
            for m in messages:
                if m.role.value == "user":
                    seen["prompt"] = m.content
            return await super().complete(messages, **kwargs)

    billing = Agent("billing", model=RecordingModel("billing answer"), description="Invoices.")
    parent = Agent("router", model=_transfer_model("billing"), sub_agents=[billing])
    await parent.run("ORIGINAL PROMPT TEXT")
    assert seen["prompt"] == "ORIGINAL PROMPT TEXT"


@pytest.mark.asyncio
async def test_unknown_agent_name_returns_error_and_run_completes():
    billing = _sub("billing", "billing answer", description="Invoices.")
    # Parent asks to transfer to a non-existent agent, then answers itself.
    parent_model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="transfer_to_agent", arguments={"agent_name": "nope"})],
                finish_reason="tool_calls",
            ),
            ModelResponse(content="parent answered after failed transfer"),
        ]
    )
    parent = Agent("router", model=parent_model, sub_agents=[billing])
    result = await parent.run("help")
    # No delegation happened; the parent finalized its own answer.
    assert result.output == "parent answered after failed transfer"
    # No transfer event was emitted (the transfer failed at the tool level).
    assert all(e.type is not EventType.AGENT_TRANSFER for e in result.events)
    # The error tool result was surfaced to the model.
    tool_results = [e for e in result.events if e.type is EventType.TOOL_RESULT]
    assert any("unknown agent" in str(e.payload.get("result", "")) for e in tool_results)


@pytest.mark.asyncio
async def test_transfer_depth_cap():
    """A chain of transfers exceeding the depth cap returns an error, not a loop."""
    # leaf transfers to itself-like target but depth runs out before answering.
    # Build a chain: parent -> mid -> deep -> deeper ... each transfers onward.
    deepest = _sub("d3", "deep answer", description="deepest")
    d2 = Agent("d2", model=_transfer_model("d3"), sub_agents=[deepest], description="layer 2")
    d1 = Agent("d1", model=_transfer_model("d2"), sub_agents=[d2], description="layer 1")
    parent = Agent("router", model=_transfer_model("d1"), sub_agents=[d1], transfer_depth=2)
    # parent -> d1 (depth 1) -> d2 (depth 2) -> d3 would be depth 3 > cap(2): blocked.
    result = await parent.run("go deep")
    # The run still completes; the over-depth transfer is reported as an error
    # result rather than delegating further. The final output is whatever the
    # last in-budget agent (d2) produces from its own loop after the blocked
    # transfer — i.e. its fallback answer, not "deep answer".
    assert result.output != "deep answer"


@pytest.mark.asyncio
async def test_transfer_depth_allows_within_budget():
    """A chain within the depth budget delegates all the way down."""
    deepest = _sub("d2", "deep answer", description="deepest")
    d1 = Agent("d1", model=_transfer_model("d2"), sub_agents=[deepest], description="layer 1")
    parent = Agent("router", model=_transfer_model("d1"), sub_agents=[d1], transfer_depth=3)
    result = await parent.run("go")
    assert result.output == "deep answer"


# --------------------------------------------------------------------------
# streaming path (stream_run)


@pytest.mark.asyncio
async def test_transfer_works_in_streaming_run():
    billing = _sub("billing", "billing answer", description="Invoices.")
    parent = Agent("router", model=_transfer_model("billing"), sub_agents=[billing])
    events = [e async for e in parent._get_runner().stream_run(parent, "help")]
    types = [e.type for e in events]
    assert EventType.AGENT_TRANSFER in types
    final = events[-1]
    assert final.type is EventType.RUN_END
    assert final.payload["result"].output == "billing answer"


# --------------------------------------------------------------------------
# regression: plain agents unaffected


@pytest.mark.asyncio
async def test_plain_agent_still_works():
    agent = Agent("a", model=TestModel("hello"))
    result = await agent.run("hi")
    assert result.output == "hello"
    assert "__transfer_to__" not in {}  # sanity
