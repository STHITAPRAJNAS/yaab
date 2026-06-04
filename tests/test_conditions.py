"""Tests for the Condition concept (yaab.conditions).

Covers Guard, Phase, Status, the Condition object and its combinators, the
as_condition coercion surface, status helpers, and the Step wrapper.
"""

from __future__ import annotations

import pytest

from yaab.conditions import (
    Condition,
    Guard,
    Phase,
    Status,
    Step,
    and_,
    as_condition,
    failed,
    not_,
    or_,
    output_contains,
    skipped,
    state_eq,
    state_ge,
    timed_out,
    when,
)
from yaab.state import ReadonlyState, State
from yaab.types import RunContext


def _guard(value, *, phase=Phase.INPUT, state=None, deps=None, status="ok"):
    st = state if state is not None else State()
    ctx = RunContext(deps=deps, state=st)
    return Guard(value=value, state=ReadonlyState(st), ctx=ctx, phase=phase, status=status)


# --- Condition basics -------------------------------------------------------


def test_condition_from_callable_single_arg():
    cond = as_condition(lambda v: v > 3, phase=Phase.INPUT)
    assert cond.check(_guard(5)) is True
    assert cond.check(_guard(1)) is False


def test_condition_from_callable_two_arg_gets_ctx():
    cond = as_condition(lambda v, ctx: ctx.identity == "alice", phase=Phase.INPUT)
    st = State()
    ctx = RunContext(state=st, identity="alice")
    g = Guard(value=None, state=ReadonlyState(st), ctx=ctx, phase=Phase.INPUT)
    assert cond.check(g) is True


def test_condition_from_bool():
    assert as_condition(True, phase=Phase.INPUT).check(_guard(None)) is True
    assert as_condition(False, phase=Phase.INPUT).check(_guard(None)) is False


def test_condition_from_string_expression():
    cond = as_condition("input == 'go'", phase=Phase.INPUT)
    assert cond.check(_guard("go")) is True


def test_condition_passthrough():
    base = Condition(lambda g: True, label="x")
    assert as_condition(base, phase=Phase.INPUT) is base


def test_as_condition_rejects_garbage():
    with pytest.raises(TypeError):
        as_condition(object(), phase=Phase.INPUT)


# --- combinators ------------------------------------------------------------


def test_and_combinator():
    c = Condition(lambda g: g.value > 1, label="a") & Condition(lambda g: g.value < 10, label="b")
    assert c.check(_guard(5)) is True
    assert c.check(_guard(50)) is False


def test_or_combinator():
    c = Condition(lambda g: g.value == 1, label="a") | Condition(lambda g: g.value == 2, label="b")
    assert c.check(_guard(2)) is True
    assert c.check(_guard(3)) is False


def test_invert_combinator():
    c = ~Condition(lambda g: g.value == 1, label="a")
    assert c.check(_guard(2)) is True
    assert c.check(_guard(1)) is False


def test_module_level_helpers():
    a = when("input > 1")
    b = when("input < 10")
    assert and_(a, b).check(_guard(5)) is True
    assert or_(when("input == 1"), when("input == 2")).check(_guard(2)) is True
    assert not_(when("input == 1")).check(_guard(2)) is True


def test_state_eq_and_ge_helpers():
    st = State()
    st["intent"] = "refund"
    st["score"] = 0.95
    assert state_eq("intent", "refund").check(_guard(None, state=st)) is True
    assert state_ge("score", 0.9).check(_guard(None, state=st)) is True
    assert state_ge("score", 0.99).check(_guard(None, state=st)) is False


def test_output_contains_helper():
    c = output_contains("done")
    assert c.check(_guard("all done", phase=Phase.OUTPUT)) is True
    assert c.check(_guard("nope", phase=Phase.OUTPUT)) is False


# --- status helpers ---------------------------------------------------------


def test_status_helpers_read_status_channel():
    assert failed().check(_guard(None, status="failed")) is True
    assert failed().check(_guard(None, status="ok")) is False
    assert timed_out().check(_guard(None, status="timeout")) is True
    assert skipped().check(_guard(None, status="skipped")) is True


def test_status_orthogonal_to_value():
    # A failed status with a "good" value is still failed (channels are distinct).
    assert failed().check(_guard("great output", status="failed")) is True


# --- Guard / ReadonlyState immutability (C3) --------------------------------


def test_guard_state_is_readonly():
    g = _guard(None)
    with pytest.raises(TypeError):
        g.state["x"] = 1  # type: ignore[index]


def test_guard_phase_fixed():
    g = _guard("x", phase=Phase.INPUT)
    assert g.phase == Phase.INPUT


# --- Status enum ------------------------------------------------------------


def test_status_values():
    assert Status.OK.value == "ok"
    assert Status.SKIPPED.value == "skipped"
    assert Status.FAILED.value == "failed"
    assert Status.TIMEOUT.value == "timeout"


# --- Step wrapper -----------------------------------------------------------


def test_step_holds_metadata():
    s = Step("unit", when="input == 1", stop="output == 2", else_="other", writes="k")
    assert s.unit == "unit"
    assert s.when == "input == 1"
    assert s.stop == "output == 2"
    assert s.else_ == "other"
    assert s.writes == "k"


def test_bare_step_has_no_guards():
    s = Step("unit")
    assert s.when is None
    assert s.stop is None
    assert s.else_ is None


# --- isomorphism: string and callable reach the same data (C2) --------------


def test_string_and_callable_isomorphic():
    st = State()
    st["intent"] = "refund"
    g = _guard(None, state=st)
    str_form = as_condition("state.intent == 'refund'", phase=Phase.INPUT)
    call_form = as_condition(lambda v, ctx: ctx.state["intent"] == "refund", phase=Phase.INPUT)
    assert str_form.check(g) == call_form.check(g) is True
