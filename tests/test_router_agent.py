"""Tests for RouterAgent and Branch (exclusive-choice routing)."""

from __future__ import annotations

import pytest

from yaab import Agent, RouterAgent, SequentialAgent
from yaab.conditions import Branch, Status
from yaab.models.test_model import FunctionModel, TestModel
from yaab.types import EventType


def _agent(name, text):
    return Agent(name, model=TestModel(text))


@pytest.mark.asyncio
async def test_first_match_wins():
    router = RouterAgent(
        "route",
        [
            Branch(when="input == 'a'", agent=_agent("first", "FIRST")),
            Branch(when="input == 'a'", agent=_agent("second", "SECOND")),
        ],
        default=_agent("def", "DEFAULT"),
    )
    result = await router.run("a")
    assert result.output == "FIRST"


@pytest.mark.asyncio
async def test_falls_through_to_default():
    router = RouterAgent(
        "route",
        [Branch(when="input == 'a'", agent=_agent("first", "FIRST"))],
        default=_agent("def", "DEFAULT"),
    )
    result = await router.run("zzz")
    assert result.output == "DEFAULT"


@pytest.mark.asyncio
async def test_on_no_match_error_raises():
    router = RouterAgent(
        "route",
        [Branch(when="input == 'a'", agent=_agent("first", "FIRST"))],
        on_no_match="error",
    )
    with pytest.raises(ValueError):
        await router.run("zzz")


@pytest.mark.asyncio
async def test_no_match_no_default_returns_skipped():
    router = RouterAgent(
        "route",
        [Branch(when="input == 'a'", agent=_agent("first", "FIRST"))],
    )
    result = await router.run("zzz")
    assert result.status == Status.SKIPPED


def test_constructor_requires_branch_or_default():
    with pytest.raises(ValueError):
        RouterAgent("route", [])


def test_constructor_rejects_bad_on_no_match():
    with pytest.raises(ValueError):
        RouterAgent("route", [Branch(when=True, agent=_agent("x", "X"))], on_no_match="bogus")


@pytest.mark.asyncio
async def test_zero_model_calls_for_routing():
    # The chosen branch makes exactly one model call; routing itself makes none.
    chosen = _agent("chosen", "CHOSEN")
    skipped_agent = _agent("skip", "NOPE")
    router = RouterAgent(
        "route",
        [
            Branch(when="input == 'go'", agent=chosen),
            Branch(when="input == 'never'", agent=skipped_agent),
        ],
        default=_agent("def", "DEF"),
    )
    result = await router.run("go")
    assert result.output == "CHOSEN"
    # Exactly one branch ran => exactly one request.
    assert result.usage.requests == 1


@pytest.mark.asyncio
async def test_from_picker_routes_by_label():
    router = RouterAgent.from_picker(
        "route",
        picker=lambda v, ctx: "billing" if "bill" in v else "tech",
        to={"billing": _agent("billing", "BILL"), "tech": _agent("tech", "TECH")},
        default=_agent("def", "DEF"),
    )
    assert (await router.run("billing question")).output == "BILL"
    assert (await router.run("my server is down")).output == "TECH"


@pytest.mark.asyncio
async def test_from_picker_unknown_label_raises():
    router = RouterAgent.from_picker(
        "route",
        picker=lambda v, ctx: "typo_label",
        to={"billing": _agent("billing", "BILL")},
        default=_agent("def", "DEF"),
    )
    with pytest.raises(ValueError):
        await router.run("anything")


@pytest.mark.asyncio
async def test_writes_captures_chosen_output():
    from yaab.state import State

    st = State()
    router = RouterAgent(
        "route",
        [Branch(when="input == 'go'", agent=_agent("c", "RESULT"))],
        default=_agent("def", "DEF"),
        writes="chosen",
    )
    await router.run("go", state=st)
    assert st["chosen"] == "RESULT"


@pytest.mark.asyncio
async def test_router_nests_in_sequential():
    router = RouterAgent(
        "route",
        [Branch(when="input == 'go'", agent=_agent("c", "ROUTED"))],
        default=_agent("def", "DEF"),
    )
    after = Agent("after", model=FunctionModel(lambda msgs: f"saw: {msgs[-1].content}"))
    seq = SequentialAgent("pipe", [router, after])
    result = await seq.run("go")
    assert "ROUTED" in result.output


@pytest.mark.asyncio
async def test_router_emits_decision_events():
    router = RouterAgent(
        "route",
        [Branch(when="input == 'go'", agent=_agent("c", "X"))],
        default=_agent("def", "DEF"),
    )
    result = await router.run("go")
    types = {e.type for e in result.events}
    assert EventType.ROUTER_EVALUATED in types
    assert EventType.ROUTER_MATCHED in types


@pytest.mark.asyncio
async def test_router_run_id_records_branch():
    router = RouterAgent(
        "route",
        [Branch(when="input == 'go'", agent=_agent("c", "X"), name="mybranch")],
        default=_agent("def", "DEF"),
    )
    result = await router.run("go")
    assert "mybranch" in result.run_id


@pytest.mark.asyncio
async def test_router_as_tool_roundtrip():
    router = RouterAgent(
        "route",
        [Branch(when="input == 'go'", agent=_agent("c", "TOOLOUT"))],
        default=_agent("def", "DEF"),
    )
    t = router.as_tool()
    assert hasattr(t, "schema")
    assert hasattr(t, "execute")


def test_router_run_sync():
    router = RouterAgent(
        "route",
        [Branch(when="input == 'go'", agent=_agent("c", "SYNC"))],
        default=_agent("def", "DEF"),
    )
    result = router.run_sync("go")
    assert result.output == "SYNC"
