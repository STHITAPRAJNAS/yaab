"""State threading across patterns: writes= capture, instruction injection,
read-only rendering, parallel conflict detection, and resume rehydration.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from yaab import Agent, ParallelAgent, SequentialAgent
from yaab.runner import Runner
from yaab.state import ReadonlyState, State, StateConflictError, StateKeyError
from yaab.testing import TestModel
from yaab.types import RunContext, RunResult


# --------------------------------------------------------------------------
# ReadonlyState — a Mapping view that forbids mutation.
# --------------------------------------------------------------------------
def test_readonly_state_reads_but_not_writes():
    st = State()
    st["k"] = 1
    ro = ReadonlyState(st)
    assert ro["k"] == 1
    assert "k" in ro
    assert dict(ro) == {"k": 1}
    with pytest.raises(TypeError):
        ro["k"] = 2  # type: ignore[index]


def test_readonly_state_reflects_live_writes():
    st = State()
    ro = ReadonlyState(st)
    st["later"] = "value"
    assert ro["later"] == "value"


# --------------------------------------------------------------------------
# writes= — typed output capture into shared state.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sequential_writes_capture_into_state():
    captured: dict = {}

    a = Agent("a", model=TestModel("summary-text"), writes="summary")
    b = Agent("b", model=TestModel(custom_output="reply", call_tools=["read_summary"]))

    @b.tool
    async def read_summary(ctx: RunContext) -> str:
        captured["summary"] = ctx.state.get("summary")
        return "ok"

    seq = SequentialAgent("seq", [a, b])
    await seq.run("start")
    assert captured["summary"] == "summary-text"


@pytest.mark.asyncio
async def test_writes_stores_typed_object_not_text():
    class Review(BaseModel):
        verdict: str
        confidence: float

    a = Agent(
        "reviewer",
        model=TestModel(structured_output={"verdict": "pass", "confidence": 0.9}),
        output_type=Review,
        writes="review",
    )
    captured: dict = {}
    b = Agent("b", model=TestModel(custom_output="done", call_tools=["read_review"]))

    @b.tool
    async def read_review(ctx: RunContext) -> str:
        captured["review"] = ctx.state.get("review")
        return "ok"

    seq = SequentialAgent("seq", [a, b])
    await seq.run("review this")
    review = captured["review"]
    assert isinstance(review, Review)
    assert review.confidence == 0.9


@pytest.mark.asyncio
async def test_writes_honors_temp_prefix():
    from yaab.sessions.memory import InMemorySessionService

    service = InMemorySessionService()
    runner = Runner(session_service=service)

    a = Agent("a", model=TestModel("scratch-value"), writes="temp:scratch", runner=runner)
    seq = SequentialAgent("seq", [a])
    await seq.run("go", session_id="sess-temp")

    session = await service.get("sess-temp")
    # temp: write is run-local; it must never persist.
    assert session is None or "temp:scratch" not in session.state


# --------------------------------------------------------------------------
# Parallel branches: distinct writes= keys readable by key; conflict detected.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parallel_distinct_writes_readable_by_key():
    a = Agent("a", model=TestModel("out-a"), writes="left")
    b = Agent("b", model=TestModel("out-b"), writes="right")

    captured: dict = {}
    reader = Agent("reader", model=TestModel(custom_output="done", call_tools=["read"]))

    @reader.tool
    async def read(ctx: RunContext) -> str:
        captured["left"] = ctx.state.get("left")
        captured["right"] = ctx.state.get("right")
        return "ok"

    par = ParallelAgent("par", [a, b])
    seq = SequentialAgent("seq", [par, reader])
    await seq.run("q")
    assert captured["left"] == "out-a"
    assert captured["right"] == "out-b"


@pytest.mark.asyncio
async def test_parallel_same_writes_key_raises_conflict():
    a = Agent("a", model=TestModel("out-a"), writes="shared")
    b = Agent("b", model=TestModel("out-b"), writes="shared")
    par = ParallelAgent("par", [a, b])
    with pytest.raises(StateConflictError):
        await par.run("q")


@pytest.mark.asyncio
async def test_parallel_output_map_still_returned():
    """Back-compat: ParallelAgent.output is still a name->result map."""
    a = Agent("a", model=TestModel("ans-a"))
    b = Agent("b", model=TestModel("ans-b"))
    par = ParallelAgent("fan", [a, b])
    result = await par.run("q")
    assert result.output == {"a": "ans-a", "b": "ans-b"}


# --------------------------------------------------------------------------
# Instruction injection — {key}, {key?}, literal braces.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_instruction_injects_state_key():
    model = TestModel("ok")
    agent = Agent(
        "responder",
        model=model,
        instructions="Use this summary: {summary}.",
    )
    runner = Runner()
    # Pre-seed state via an inherited State object.
    state = State()
    state["summary"] = "the-summary"
    await runner.run(agent, "hi", state=state)
    system = model.calls[0][0]
    assert system.content == "Use this summary: the-summary."


@pytest.mark.asyncio
async def test_instruction_optional_key_missing_is_blank():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Tone: {tone?}.")
    runner = Runner()
    await runner.run(agent, "hi", state=State())
    system = model.calls[0][0]
    assert system.content == "Tone: ."


@pytest.mark.asyncio
async def test_instruction_missing_required_key_raises():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Need: {missing}.")
    runner = Runner()
    with pytest.raises(StateKeyError):
        await runner.run(agent, "hi", state=State())


@pytest.mark.asyncio
async def test_instruction_leaves_literal_braces_untouched():
    model = TestModel("ok")
    # JSON-ish braces and numeric placeholders must pass through unchanged.
    tmpl = 'Return {"role": "user"} and item {0} verbatim.'
    agent = Agent("a", model=model, instructions=tmpl)
    runner = Runner()
    await runner.run(agent, "hi", state=State())
    system = model.calls[0][0]
    assert system.content == tmpl


@pytest.mark.asyncio
async def test_instruction_callable_receives_readonly_state():
    seen: dict = {}

    def make_instructions(ctx: RunContext) -> str:
        seen["state_type"] = type(ctx.state).__name__
        seen["is_readonly"] = isinstance(ctx.state, ReadonlyState)
        return "computed"

    model = TestModel("ok")
    agent = Agent("a", model=model, instructions=make_instructions)
    runner = Runner()
    await runner.run(agent, "hi", state=State())
    assert seen["is_readonly"] is True


# --------------------------------------------------------------------------
# Build-once / inherit-always: a passed-in State is not rebuilt.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inherited_state_is_not_rebuilt():
    captured: dict = {}
    agent = Agent("a", model=TestModel(custom_output="done", call_tools=["peek"]))

    @agent.tool
    async def peek(ctx: RunContext) -> str:
        captured["state"] = ctx.state
        return "ok"

    runner = Runner()
    state = State()
    state["preexisting"] = 1
    await runner.run(agent, "hi", state=state)
    assert captured["state"] is state


# --------------------------------------------------------------------------
# RunResult.paused contract (S12).
# --------------------------------------------------------------------------
def test_run_result_paused_defaults_false():
    r: RunResult = RunResult(output="x")
    assert r.paused is False
    assert r.pause_value is None


def test_run_result_pending_defaults_empty():
    """A normal (non-paused) result carries an empty ``pending`` list."""
    r: RunResult = RunResult(output="x")
    assert r.pending == []
    # The list is per-instance (default_factory), never a shared class-level list.
    r.pending.append("sentinel")
    assert RunResult(output="y").pending == []


def test_run_result_pending_is_typed_field():
    """``pending`` is a real, settable field of typed ``Pending`` parked decisions."""
    from yaab.types import Pending

    r: RunResult = RunResult(
        output=None, paused=True, pending=[Pending(kind="approval", tool="wire")]
    )
    assert r.paused is True
    assert len(r.pending) == 1
    assert r.pending[0].kind == "approval"
    assert r.pending[0].tool == "wire"


# --------------------------------------------------------------------------
# Resume rehydration: committed state survives a resume.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_step_checkpoint_carries_persisted_state():
    from yaab.graph.checkpoint import MemorySaver
    from yaab.models.base import ModelResponse
    from yaab.types import ToolCall

    saver = MemorySaver()
    runner = Runner(run_checkpointer=saver)

    # An agent that writes state then calls a tool (so a step checkpoint lands).
    agent = Agent(
        "a",
        model=TestModel(
            responses=[
                ModelResponse(
                    tool_calls=[ToolCall(name="remember", arguments={})],
                    finish_reason="tool_calls",
                ),
                "final-answer",
            ]
        ),
    )

    @agent.tool
    async def remember(ctx: RunContext) -> str:
        ctx.state["committed"] = "durable"
        ctx.state["temp:scratch"] = "ephemeral"
        return "remembered"

    result = await runner.run(agent, "go", resume_id="r-state")
    assert result.output == "final-answer"

    # The per-step checkpoint payload carries persisted() state (temp excluded).
    history = saver.history("r-state")
    assert history, "expected at least one checkpoint"
    step_states = [s for _, s in history if "state" in s]
    assert step_states, "checkpoint payload must include a 'state' field"
    committed = step_states[-1]["state"]
    assert committed.get("committed") == "durable"
    assert "temp:scratch" not in committed


@pytest.mark.asyncio
async def test_resume_rehydrates_committed_state_after_pause():
    """A run paused for approval and resumed restores its committed state."""
    from yaab.governance.approval import ToolApprovalPlugin
    from yaab.governance.approvals import InMemoryApprovalStore
    from yaab.graph.checkpoint import MemorySaver
    from yaab.models.base import ModelResponse
    from yaab.types import ToolCall

    saver = MemorySaver()
    store = InMemoryApprovalStore()
    approval = ToolApprovalPlugin(tools=["wire"], mode="queue", store=store)
    runner = Runner(run_checkpointer=saver, plugins=[approval])

    seen: dict = {}

    agent = Agent(
        "a",
        model=TestModel(
            responses=[
                ModelResponse(
                    tool_calls=[ToolCall(name="remember", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    tool_calls=[ToolCall(name="wire", arguments={"amount": 1})],
                    finish_reason="tool_calls",
                ),
                "final-answer",
            ]
        ),
    )

    @agent.tool
    async def remember(ctx: RunContext) -> str:
        ctx.state["committed"] = "durable"
        return "remembered"

    @agent.tool
    async def wire(ctx: RunContext, amount: int) -> str:
        seen["state_on_resume"] = ctx.state.get("committed")
        return f"wired {amount}"

    # First invocation: pauses at the wire approval.
    r1 = await runner.run(agent, "go", resume_id="r-pause")
    assert r1.paused is True

    # Resume with an approval: the held tool runs and must see committed state.
    r2 = await runner.run(agent, "go", resume_id="r-pause", approval_decision="approved")
    assert r2.output == "final-answer"
    assert seen["state_on_resume"] == "durable"
