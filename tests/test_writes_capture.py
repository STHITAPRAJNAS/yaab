"""``writes=`` output auto-capture — the declarative inter-agent handoff.

An agent (or workflow step) declares ``writes="key"`` and the framework lands its
*validated* output into shared state under that key after it completes; a
downstream step reads it by name. These tests pin the documented behavior:

* capture stores the typed object, not stringified text;
* the key's prefix (``temp:``/``user:``/``app:``/session) selects the scope;
* two agents can communicate purely via ``writes=`` + ``{key}`` injection, with
  no prompt piping;
* a ParallelAgent's branches each write a distinct key that a downstream
  Sequential step reads;
* the capture is visible in the trace as a ``STATE_WRITE`` event;
* an agent without ``writes=`` behaves exactly as before (no capture).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from yaab import Agent, ParallelAgent, SequentialAgent
from yaab.runner import Runner
from yaab.sessions.memory import InMemorySessionService
from yaab.state import State
from yaab.testing import TestModel
from yaab.types import EventType, RunContext


# --------------------------------------------------------------------------
# Standalone capture: a plain Agent.run lands its output under writes=.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_standalone_agent_writes_into_inherited_state():
    """A standalone agent with writes= lands its output into the shared State."""
    agent = Agent("a", model=TestModel("captured-text"), writes="result")
    state = State()
    await agent.run("go", state=state)
    assert state["result"] == "captured-text"


@pytest.mark.asyncio
async def test_writes_stores_typed_object_not_stringified():
    """The captured value is the validated object, not a JSON string of it."""

    class Review(BaseModel):
        verdict: str
        confidence: float

    reviewer = Agent(
        "reviewer",
        model=TestModel(structured_output={"verdict": "pass", "confidence": 0.91}),
        output_type=Review,
        writes="review",
    )
    state = State()
    await reviewer.run("review this", state=state)

    review = state["review"]
    assert isinstance(review, Review)
    # The downstream contract is a typed attribute read, identical at every site.
    assert review.confidence == 0.91
    assert review.verdict == "pass"


# --------------------------------------------------------------------------
# Two agents communicating PURELY via writes= + {key} (no prompt piping).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_two_agents_handoff_via_writes_and_template_no_piping():
    """Agent A writes 'summary'; Agent B reads it via {summary} injection.

    Piping is disabled, so the only channel between them is shared state: the
    handoff is declarative on both ends (writes= on A, {summary} on B).
    """
    writer = Agent("writer", model=TestModel("the-distilled-summary"), writes="summary")

    reader_model = TestModel("acknowledged")
    reader = Agent(
        "reader",
        model=reader_model,
        instructions="Reply using only this summary: {summary}.",
    )

    seq = SequentialAgent("seq", [writer, reader], pipe_output=False)
    await seq.run("the original long document")

    # The reader's rendered system instruction carries A's captured output —
    # proving the value crossed via state, not via a piped prompt.
    system = reader_model.calls[0][0]
    assert system.content == "Reply using only this summary: the-distilled-summary."
    # And piping is off, so the reader's user turn is the original prompt, not A's text.
    user = reader_model.calls[0][-1]
    assert user.content == "the original long document"


# --------------------------------------------------------------------------
# ParallelAgent: each branch writes a distinct key; a downstream step reads both.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parallel_branches_write_distinct_keys_downstream_reads_both():
    """Two parallel branches write distinct keys; a Sequential step reads both."""
    left = Agent("left", model=TestModel("L"), writes="left_result")
    right = Agent("right", model=TestModel("R"), writes="right_result")
    fan_out = ParallelAgent("fan", [left, right])

    combiner_model = TestModel("combined")
    combiner = Agent(
        "combiner",
        model=combiner_model,
        instructions="Combine {left_result} and {right_result}.",
    )

    seq = SequentialAgent("seq", [fan_out, combiner], pipe_output=False)
    await seq.run("split this")

    system = combiner_model.calls[0][0]
    assert system.content == "Combine L and R."


@pytest.mark.asyncio
async def test_parallel_branches_via_tool_read_both_keys():
    """The downstream reader can also read the two branch keys from a tool."""
    left = Agent("left", model=TestModel("out-left"), writes="k_left")
    right = Agent("right", model=TestModel("out-right"), writes="k_right")
    fan_out = ParallelAgent("fan", [left, right])

    captured: dict = {}
    reader = Agent("reader", model=TestModel(custom_output="ok", call_tools=["read"]))

    @reader.tool
    async def read(ctx: RunContext) -> str:
        captured["left"] = ctx.state.get("k_left")
        captured["right"] = ctx.state.get("k_right")
        return "read"

    seq = SequentialAgent("seq", [fan_out, reader], pipe_output=False)
    await seq.run("q")
    assert captured == {"left": "out-left", "right": "out-right"}


# --------------------------------------------------------------------------
# Prefix behavior: writes= honors temp:/user:/app: scopes.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_writes_temp_prefix_is_run_local_and_visible_in_run():
    """A temp: writes= key is visible to a later step in the run, but never persisted."""
    service = InMemorySessionService()
    runner = Runner(session_service=service)

    writer = Agent("writer", model=TestModel("scratch"), writes="temp:draft", runner=runner)

    saw: dict = {}
    reader = Agent(
        "reader",
        model=TestModel(custom_output="ok", call_tools=["peek"]),
        runner=runner,
    )

    @reader.tool
    async def peek(ctx: RunContext) -> str:
        saw["draft"] = ctx.state.get("temp:draft")
        return "ok"

    seq = SequentialAgent("seq", [writer, reader], pipe_output=False)
    await seq.run("go", session_id="sess-temp")

    # Visible within the run...
    assert saw["draft"] == "scratch"
    # ...but excluded from durable storage.
    session = await service.get("sess-temp")
    assert session is not None
    assert "temp:draft" not in session.state


@pytest.mark.asyncio
async def test_writes_session_prefix_persists_to_session():
    """An unprefixed (session-scoped) writes= key is written back to the session."""
    service = InMemorySessionService()
    runner = Runner(session_service=service)

    writer = Agent("writer", model=TestModel("kept-value"), writes="durable", runner=runner)
    await writer.run("go", session_id="sess-keep")

    session = await service.get("sess-keep")
    assert session is not None
    assert session.state.get("durable") == "kept-value"


@pytest.mark.asyncio
async def test_writes_user_prefix_routes_to_user_scope():
    """A user: writes= key routes to the user scope, not the session scope."""
    writer = Agent("writer", model=TestModel("tone-value"), writes="user:tone")
    state = State()
    await writer.run("go", state=state)

    assert state.user.get("user:tone") == "tone-value"
    # It is not a plain session key.
    assert "user:tone" not in state.session


# --------------------------------------------------------------------------
# Observability: capture emits a STATE_WRITE event with key + scope.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_capture_emits_state_write_event_with_scope():
    agent = Agent("a", model=TestModel("v"), writes="temp:note")
    result = await agent.run("go", state=State())

    writes = [e for e in result.events if e.type is EventType.STATE_WRITE]
    assert len(writes) == 1
    assert writes[0].payload["key"] == "temp:note"
    assert writes[0].payload["scope"] == "temp"


@pytest.mark.asyncio
async def test_session_scoped_capture_reports_session_scope():
    agent = Agent("a", model=TestModel("v"), writes="topic")
    result = await agent.run("go", state=State())
    writes = [e for e in result.events if e.type is EventType.STATE_WRITE]
    assert writes and writes[0].payload["scope"] == "session"


# --------------------------------------------------------------------------
# Backward compatibility: no writes= -> no capture, no event, byte-for-byte same.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_writes_means_no_capture_and_no_event():
    agent = Agent("a", model=TestModel("answer"))  # no writes=
    state = State()
    result = await agent.run("go", state=state)

    assert result.output == "answer"
    # Nothing was written into shared state...
    assert dict(state) == {}
    # ...and no STATE_WRITE event was emitted.
    assert not [e for e in result.events if e.type is EventType.STATE_WRITE]


@pytest.mark.asyncio
async def test_sequential_pipe_output_unchanged_without_writes():
    """Back-compat: classic pipe_output still feeds A's text to B as the prompt."""
    a = Agent("a", model=TestModel("piped-text"))  # no writes=
    b_model = TestModel("done")
    b = Agent("b", model=b_model)

    seq = SequentialAgent("seq", [a, b])  # pipe_output defaults to True
    await seq.run("original")

    # B's user turn is A's stringified output, exactly as before.
    user = b_model.calls[0][-1]
    assert user.content == "piped-text"
