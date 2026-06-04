"""The shared-state foundation: one State object, observed everywhere.

These tests assert the single invariant the whole orchestration layer rests on:
for one top-level run there is exactly ONE state object, and the same object is
seen by a tool in agent A, then agent B in a sequence, then a parallel branch,
then loop iterations, then swarm handoffs. Plus persistence round-trip and
resume rehydration.
"""

from __future__ import annotations

import pytest

from yaab import (
    Agent,
    LoopAgent,
    ParallelAgent,
    SequentialAgent,
    Swarm,
)
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel
from yaab.multiagent import MapAgent, SwarmState
from yaab.runner import Runner
from yaab.sessions.memory import InMemorySessionService
from yaab.state import State
from yaab.testing import TestModel
from yaab.types import RunContext, ToolCall


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _calls_tool_each_run(tool_name: str, answer: str = "done"):
    """A model that requests ``tool_name`` once per *run*, then answers.

    Unlike ``TestModel(call_tools=...)`` (whose one-shot flag is sticky across
    runs of the same instance), this re-arms every run — needed for LoopAgent /
    MapAgent which reuse one agent (and so one model) across iterations/items.
    """

    def respond(messages):
        # The user turn is the last message whenever a fresh run begins (no tool
        # result yet this turn); after the tool runs, finalize with the answer.
        last = messages[-1]
        if last.role.value == "tool":
            return answer
        return ModelResponse(
            tool_calls=[ToolCall(name=tool_name, arguments={})],
            finish_reason="tool_calls",
        )

    return FunctionModel(respond)


# --------------------------------------------------------------------------
# S0 / S3 — one State object per run; standalone agent builds it.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_standalone_agent_state_is_a_State():
    captured: dict = {}
    agent = Agent("a", model=TestModel(custom_output="done", call_tools=["peek"]))

    @agent.tool
    async def peek(ctx: RunContext) -> str:
        captured["state"] = ctx.state
        return "ok"

    await agent.run("hi")
    assert isinstance(captured["state"], State)


@pytest.mark.asyncio
async def test_same_state_object_across_sequential_steps():
    """A tool in step A and a tool in step B see the SAME state object."""
    seen: list[int] = []

    a = Agent("a", model=TestModel(custom_output="da", call_tools=["mark_a"]))

    @a.tool
    async def mark_a(ctx: RunContext) -> str:
        seen.append(id(ctx.state))
        ctx.state["from_a"] = "value-a"
        return "a-done"

    b = Agent("b", model=TestModel(custom_output="db", call_tools=["mark_b"]))

    @b.tool
    async def mark_b(ctx: RunContext) -> str:
        seen.append(id(ctx.state))
        # B can read what A wrote into the shared state.
        ctx.state["b_saw"] = ctx.state.get("from_a")
        return "b-done"

    seq = SequentialAgent("seq", [a, b])
    await seq.run("start")

    assert len(seen) == 2
    assert seen[0] == seen[1], "A and B must observe the same state object"


@pytest.mark.asyncio
async def test_sequential_b_reads_what_a_wrote():
    a = Agent("a", model=TestModel(custom_output="da", call_tools=["mark_a"]))

    @a.tool
    async def mark_a(ctx: RunContext) -> str:
        ctx.state["from_a"] = "hello"
        return "a-done"

    saw: dict = {}
    b = Agent("b", model=TestModel(custom_output="db", call_tools=["mark_b"]))

    @b.tool
    async def mark_b(ctx: RunContext) -> str:
        saw["value"] = ctx.state.get("from_a")
        return "b-done"

    seq = SequentialAgent("seq", [a, b])
    await seq.run("start")
    assert saw["value"] == "hello"


# --------------------------------------------------------------------------
# S0 — parallel branches share one state object.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parallel_branches_share_one_state_object():
    seen: list[int] = []

    def branch(tag: str) -> Agent:
        agent = Agent(tag, model=TestModel(custom_output=tag, call_tools=[f"mark_{tag}"]))

        async def mark(ctx: RunContext) -> str:
            seen.append(id(ctx.state))
            return "marked"

        agent.tool(mark, name=f"mark_{tag}")
        return agent

    par = ParallelAgent("par", [branch("x"), branch("y")])
    await par.run("q")

    assert len(seen) == 2
    assert seen[0] == seen[1], "parallel branches must share one state object"


# --------------------------------------------------------------------------
# S0 — loop iterations share one accumulating state object.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_loop_iterations_share_one_state_object():
    seen: list[int] = []

    agent = Agent("a", model=_calls_tool_each_run("tick"))

    @agent.tool
    async def tick(ctx: RunContext) -> str:
        seen.append(id(ctx.state))
        ctx.state["count"] = ctx.state.get("count", 0) + 1
        return "ticked"

    loop = LoopAgent("loop", agent, max_iterations=3)
    await loop.run("go")

    # All iterations saw the same object.
    assert len(seen) == 3
    assert len(set(seen)) == 1


@pytest.mark.asyncio
async def test_loop_accumulates_in_shared_state():
    agent = Agent("a", model=_calls_tool_each_run("tick"))

    captured: dict = {}

    @agent.tool
    async def tick(ctx: RunContext) -> str:
        ctx.state["count"] = ctx.state.get("count", 0) + 1
        captured["count"] = ctx.state["count"]
        return "ticked"

    loop = LoopAgent("loop", agent, max_iterations=3)
    await loop.run("go")
    # Accumulation across iterations is visible: 3 increments on one shared dict.
    assert captured["count"] == 3


# --------------------------------------------------------------------------
# S0 — swarm handoffs share one state object.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_swarm_handoffs_share_one_state_object():
    seen: list[int] = []

    triage = Agent(
        "triage",
        model=TestModel(custom_output="routed", call_tools=["mark_t", "handoff_to_specialist"]),
    )

    @triage.tool
    async def mark_t(ctx: RunContext) -> str:
        seen.append(id(ctx.state))
        return "t"

    specialist = Agent("specialist", model=TestModel(custom_output="answer", call_tools=["mark_s"]))

    @specialist.tool
    async def mark_s(ctx: RunContext) -> str:
        seen.append(id(ctx.state))
        return "s"

    swarm = Swarm("support", [triage, specialist], entry="triage")
    await swarm.run("help", deps=SwarmState())

    assert len(seen) == 2
    assert seen[0] == seen[1], "swarm members must observe one shared state object"


# --------------------------------------------------------------------------
# S0 — map children share one state object and keep their session_id.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_map_children_share_state_and_keep_session_id():
    seen_state: list[int] = []
    seen_session: list[str | None] = []

    agent = Agent("worker", model=_calls_tool_each_run("mark"))

    @agent.tool
    async def mark(ctx: RunContext) -> str:
        seen_state.append(id(ctx.state))
        seen_session.append(ctx.session_id)
        return "marked"

    mp = MapAgent("map", agent)
    await mp.run(["one", "two"], session_id="sess-map")

    assert len(seen_state) == 2
    assert seen_state[0] == seen_state[1]
    # MapAgent used to drop session_id for its children; it must be threaded now.
    assert seen_session == ["sess-map", "sess-map"]


# --------------------------------------------------------------------------
# Full chain: one state observed across sequence -> parallel -> loop.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_state_threads_through_nested_patterns():
    ids: list[int] = []

    def marker(tag: str) -> Agent:
        agent = Agent(tag, model=_calls_tool_each_run(f"m_{tag}", answer=tag))

        async def mark(ctx: RunContext) -> str:
            ids.append(id(ctx.state))
            return "ok"

        agent.tool(mark, name=f"m_{tag}")
        return agent

    inner_par = ParallelAgent("par", [marker("p1"), marker("p2")])
    loop_agent = marker("loop_body")
    inner_loop = LoopAgent("loop", loop_agent, max_iterations=2)
    seq = SequentialAgent("seq", [marker("first"), inner_par, inner_loop])

    await seq.run("go")
    # first(1) + par(2) + loop(2) = 5 marks, all the same object.
    assert len(ids) == 5
    assert len(set(ids)) == 1


# --------------------------------------------------------------------------
# Persistence round-trip: session-scoped writes survive across runs.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_state_persists_to_session_and_rehydrates():
    service = InMemorySessionService()
    runner = Runner(session_service=service)

    agent = Agent("a", model=TestModel(custom_output="done", call_tools=["write_state"]))

    @agent.tool
    async def write_state(ctx: RunContext) -> str:
        ctx.state["remembered"] = "yes"
        ctx.state["temp:scratch"] = "should-not-persist"
        return "wrote"

    await runner.run(agent, "first", session_id="s1")

    session = await service.get("s1")
    assert session is not None
    assert session.state.get("remembered") == "yes"
    # temp: must never reach durable storage.
    assert "temp:scratch" not in session.state

    # A later run on the same session rehydrates the durable value.
    saw: dict = {}
    agent2 = Agent("a", model=TestModel(custom_output="done2", call_tools=["read_state"]))

    @agent2.tool
    async def read_state(ctx: RunContext) -> str:
        saw["value"] = ctx.state.get("remembered")
        return "read"

    await runner.run(agent2, "second", session_id="s1")
    assert saw["value"] == "yes"


@pytest.mark.asyncio
async def test_internal_engine_keys_never_persist():
    """Engine bookkeeping (run-local temp:) must not leak into session.state."""
    service = InMemorySessionService()
    runner = Runner(session_service=service)

    agent = Agent("a", model=TestModel(custom_output="done", call_tools=["noop"]))

    @agent.tool
    async def noop(ctx: RunContext) -> str:
        # A regular write persists; a temp: write and any internal bookkeeping
        # (e.g. the resume id the runner stashes) must not.
        ctx.state["kept"] = 1
        ctx.state["temp:scratch"] = "ephemeral"
        return "ok"

    await runner.run(agent, "go", session_id="sX", resume_id="rid-x")
    session = await service.get("sX")
    assert session is not None
    assert session.state == {"kept": 1}, "only the durable user write should persist"
