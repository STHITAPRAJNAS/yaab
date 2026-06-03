"""State-templated instructions — ``{key}`` injection from shared state.

A string instruction's ``{key}`` placeholders are substituted from shared state
before the model call; ``{key?}`` is optional (missing -> empty string); a
missing *required* key raises a clear, actionable error; anything that is not an
identifier-style field (JSON braces, numeric placeholders, CSS) is left
untouched. Callable instructions receive a *read-only* state view and are never
auto-injected. Substitutions are visible in the trace as a ``STATE_TEMPLATE``
event. An instruction with no placeholders renders byte-for-byte as before.
"""

from __future__ import annotations

import pytest

from yaab import Agent, SequentialAgent
from yaab.runner import Runner
from yaab.state import ReadonlyState, State, StateKeyError
from yaab.testing import TestModel
from yaab.types import EventType, RunContext


def _system_of(model: TestModel) -> str:
    """The rendered system message content of the model's first complete() call."""
    return model.calls[0][0].content


# --------------------------------------------------------------------------
# {key} substitution from shared state.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_required_key_substitutes_from_state():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Use this summary: {summary}.")
    state = State()
    state["summary"] = "the-summary"
    await Runner().run(agent, "hi", state=state)
    assert _system_of(model) == "Use this summary: the-summary."


@pytest.mark.asyncio
async def test_multiple_keys_substitute_in_one_template():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="{greeting}, {name}!")
    state = State()
    state["greeting"] = "Hello"
    state["name"] = "Ada"
    await Runner().run(agent, "hi", state=state)
    assert _system_of(model) == "Hello, Ada!"


@pytest.mark.asyncio
async def test_typed_state_value_is_rendered_as_text():
    """A non-string state value is rendered to text for the prompt (read side)."""
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Count is {count}.")
    state = State()
    state["count"] = 7
    await Runner().run(agent, "hi", state=state)
    assert _system_of(model) == "Count is 7."


# --------------------------------------------------------------------------
# Prefix routing: {user:...} / {app:...} resolve from their scope.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prefixed_key_resolves_from_its_scope():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Tone: {user:tone}. Glossary: {app:glossary}.")
    state = State()
    state["user:tone"] = "warm"
    state["app:glossary"] = "house-style"
    await Runner().run(agent, "hi", state=state)
    assert _system_of(model) == "Tone: warm. Glossary: house-style."


# --------------------------------------------------------------------------
# Optional {key?} and missing required key.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_optional_key_missing_substitutes_empty_string():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Tone:{tone?} end.")
    await Runner().run(agent, "hi", state=State())
    assert _system_of(model) == "Tone: end."


@pytest.mark.asyncio
async def test_optional_key_present_substitutes_value():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Tone: {tone?}.")
    state = State()
    state["tone"] = "formal"
    await Runner().run(agent, "hi", state=state)
    assert _system_of(model) == "Tone: formal."


@pytest.mark.asyncio
async def test_missing_required_key_raises_actionable_error():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Need: {missing}.")
    with pytest.raises(StateKeyError) as exc:
        await Runner().run(agent, "hi", state=State())
    # The error names the offending key and how to fix it.
    msg = str(exc.value)
    assert "missing" in msg
    assert "{missing?}" in msg


# --------------------------------------------------------------------------
# JSON-literal safety: only identifier-style fields are treated as state.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_json_and_numeric_braces_pass_through_untouched():
    model = TestModel("ok")
    tmpl = 'Return {"role": "user"}, item {0}, and set { color: red }.'
    agent = Agent("a", model=model, instructions=tmpl)
    await Runner().run(agent, "hi", state=State())
    assert _system_of(model) == tmpl


@pytest.mark.asyncio
async def test_unknown_brace_with_state_field_mixed():
    """A real field substitutes; adjacent JSON-ish braces are left alone."""
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions='Schema {"k": 1}; value {v}.')
    state = State()
    state["v"] = "X"
    await Runner().run(agent, "hi", state=state)
    assert _system_of(model) == 'Schema {"k": 1}; value X.'


# --------------------------------------------------------------------------
# Callable instructions: read-only state, no auto-injection.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_callable_instruction_receives_readonly_state():
    seen: dict = {}

    def make(ctx: RunContext) -> str:
        seen["is_readonly"] = isinstance(ctx.state, ReadonlyState)
        seen["value"] = ctx.state["seed"]
        return "computed instruction"

    model = TestModel("ok")
    agent = Agent("a", model=model, instructions=make)
    state = State()
    state["seed"] = "from-state"
    await Runner().run(agent, "hi", state=state)

    assert seen["is_readonly"] is True
    assert seen["value"] == "from-state"
    # The callable's return value is used verbatim — no {..} injection applied.
    assert _system_of(model) == "computed instruction"


@pytest.mark.asyncio
async def test_callable_instruction_cannot_mutate_state():
    def make(ctx: RunContext) -> str:
        with pytest.raises(TypeError):
            ctx.state["x"] = 1  # type: ignore[index]
        return "ro"

    model = TestModel("ok")
    agent = Agent("a", model=model, instructions=make)
    await Runner().run(agent, "hi", state=State())
    assert _system_of(model) == "ro"


@pytest.mark.asyncio
async def test_callable_return_with_braces_is_not_injected():
    """A callable that returns literal braces is left untouched (author control)."""
    model = TestModel("ok")

    def make(ctx: RunContext) -> str:
        return "Return {payload} as-is."

    agent = Agent("a", model=model, instructions=make)
    state = State()
    state["payload"] = "SHOULD-NOT-APPEAR"
    await Runner().run(agent, "hi", state=state)
    assert _system_of(model) == "Return {payload} as-is."


# --------------------------------------------------------------------------
# Observability: substitution emits a STATE_TEMPLATE event listing the keys.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_substitution_emits_state_template_event():
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="{greeting}, {name}!")
    state = State()
    state["greeting"] = "Hi"
    state["name"] = "Bo"
    result = await Runner().run(agent, "hi", state=state)

    tmpl_events = [e for e in result.events if e.type is EventType.STATE_TEMPLATE]
    assert len(tmpl_events) == 1
    assert set(tmpl_events[0].payload["keys"]) == {"greeting", "name"}


@pytest.mark.asyncio
async def test_plain_instruction_emits_no_state_template_event():
    """Back-compat: an instruction with no fields renders as before, no event."""
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Be concise and helpful.")
    result = await Runner().run(agent, "hi", state=State())

    assert _system_of(model) == "Be concise and helpful."
    assert not [e for e in result.events if e.type is EventType.STATE_TEMPLATE]


@pytest.mark.asyncio
async def test_optional_missing_key_emits_no_template_event():
    """An optional key that resolved to nothing is not reported as a substitution."""
    model = TestModel("ok")
    agent = Agent("a", model=model, instructions="Tone:{tone?}.")
    result = await Runner().run(agent, "hi", state=State())
    # Nothing was actually pulled from state, so no STATE_TEMPLATE event.
    assert not [e for e in result.events if e.type is EventType.STATE_TEMPLATE]


# --------------------------------------------------------------------------
# End-to-end: writes= on A drives {key} on B (read side of the handoff).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_writes_then_template_end_to_end():
    a = Agent("a", model=TestModel("distilled"), writes="summary")
    b_model = TestModel("done")
    b = Agent("b", model=b_model, instructions="Summary was: {summary}.")
    seq = SequentialAgent("seq", [a, b], pipe_output=False)
    await seq.run("long input")
    assert _system_of(b_model) == "Summary was: distilled."
