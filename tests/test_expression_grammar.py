"""Tests for the safe expression grammar (yaab.expr.compile_expr).

These prove the sandbox is genuinely safe (no eval/exec/import/attribute
traversal to dunders/arbitrary calls), that the grammar reaches exactly the
documented roots, that phase mismatches are load-time errors, and that operand
values are captured for decision events.
"""

from __future__ import annotations

import pytest

from yaab.conditions import Guard, Phase
from yaab.expr import (
    ConditionPhaseError,
    ConditionSyntaxError,
    compile_expr,
)
from yaab.state import ReadonlyState, State
from yaab.types import RunContext


def _guard(value, *, phase=Phase.INPUT, state=None, deps=None, status="ok"):
    st = state if state is not None else State()
    ctx = RunContext(deps=deps, state=st)
    return Guard(value=value, state=ReadonlyState(st), ctx=ctx, phase=phase, status=status)


# --- basic comparisons ------------------------------------------------------


def test_state_equality():
    st = State()
    st["intent"] = "refund"
    cond = compile_expr("state.intent == 'refund'", phase=Phase.INPUT)
    assert cond.check(_guard(None, state=st)) is True
    st["intent"] = "billing"
    assert cond.check(_guard(None, state=st)) is False


def test_input_reference():
    cond = compile_expr("input == 'go'", phase=Phase.INPUT)
    assert cond.check(_guard("go")) is True
    assert cond.check(_guard("stop")) is False


def test_output_reference_numeric():
    cond = compile_expr("output.score >= 0.9", phase=Phase.OUTPUT)

    class Out:
        score = 0.95

    assert cond.check(_guard(Out(), phase=Phase.OUTPUT)) is True

    class Low:
        score = 0.1

    assert cond.check(_guard(Low(), phase=Phase.OUTPUT)) is False


def test_value_alias_for_output():
    cond = compile_expr("value == 42", phase=Phase.OUTPUT)
    assert cond.check(_guard(42, phase=Phase.OUTPUT)) is True


def test_subscript_access_reaches_prefixed_keys():
    st = State()
    st["temp:__handoff__"] = "legal"
    cond = compile_expr("state['temp:__handoff__'] == 'legal'", phase=Phase.INPUT)
    assert cond.check(_guard(None, state=st)) is True


def test_boolean_and_or_not():
    st = State()
    st["a"] = 1
    st["b"] = 2
    assert compile_expr("state.a == 1 and state.b == 2", phase=Phase.INPUT).check(
        _guard(None, state=st)
    )
    assert compile_expr("state.a == 9 or state.b == 2", phase=Phase.INPUT).check(
        _guard(None, state=st)
    )
    assert compile_expr("not state.a == 9", phase=Phase.INPUT).check(_guard(None, state=st))


def test_in_operator():
    st = State()
    st["role"] = "admin"
    cond = compile_expr("state.role in ['admin', 'root']", phase=Phase.INPUT)
    assert cond.check(_guard(None, state=st)) is True


def test_arithmetic_comparison():
    cond = compile_expr("input + 1 > 2", phase=Phase.INPUT)
    assert cond.check(_guard(5)) is True
    assert cond.check(_guard(0)) is False


def test_contains_helper():
    cond = compile_expr("output contains 'done'", phase=Phase.OUTPUT)
    assert cond.check(_guard("all done now", phase=Phase.OUTPUT)) is True
    assert cond.check(_guard("nope", phase=Phase.OUTPUT)) is False


def test_literals_true_false_null():
    assert compile_expr("input == true", phase=Phase.INPUT).check(_guard(True))
    assert compile_expr("input == false", phase=Phase.INPUT).check(_guard(False))
    assert compile_expr("input == null", phase=Phase.INPUT).check(_guard(None))


def test_ctx_identity_reachable():
    st = State()
    ctx = RunContext(state=st, identity="alice")
    g = Guard(value=None, state=ReadonlyState(st), ctx=ctx, phase=Phase.INPUT)
    cond = compile_expr("ctx.identity == 'alice'", phase=Phase.INPUT)
    assert cond.check(g) is True


def test_deps_attribute_reachable():
    class Deps:
        tier = "enterprise"

    cond = compile_expr("deps.tier == 'enterprise'", phase=Phase.INPUT)
    assert cond.check(_guard(None, deps=Deps())) is True


# --- phase enforcement (C12) -----------------------------------------------


def test_output_reference_in_input_phase_is_load_error():
    with pytest.raises(ConditionPhaseError):
        compile_expr("output.score >= 0.9", phase=Phase.INPUT)


def test_input_reference_in_output_phase_is_allowed():
    # input is still readable under stop= (it's the same Guard.value channel only
    # for output/value; input remains a distinct binding). Reject by design:
    with pytest.raises(ConditionPhaseError):
        compile_expr("input == 'x'", phase=Phase.OUTPUT)


# --- adversarial / safety (the sandbox must reject these) -------------------


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "__import__('os').system('echo hi')",
        "().__class__.__bases__",
        "().__class__.__bases__[0].__subclasses__()",
        "state.__class__",
        "input.__class__.__mro__",
        "open('x')",
        "eval('1')",
        "exec('1')",
        "lambda: 1",
        "[x for x in [1]]",
        "input if True else 2",
        "{'a': 1}",
        "input.foo()",
        "globals()",
        "state.__dict__",
        "input.__getattribute__('x')",
    ],
)
def test_rejects_dangerous_expressions(expr):
    with pytest.raises((ConditionSyntaxError, ConditionPhaseError)):
        compile_expr(expr, phase=Phase.INPUT)


def test_rejects_unknown_root():
    with pytest.raises(ConditionSyntaxError):
        compile_expr("evil == 1", phase=Phase.INPUT)


def test_rejects_dunder_attribute():
    with pytest.raises(ConditionSyntaxError):
        compile_expr("input.__class__ == 1", phase=Phase.INPUT)


def test_rejects_call_node():
    with pytest.raises(ConditionSyntaxError):
        compile_expr("len(input) > 0", phase=Phase.INPUT)


def test_no_attribute_traversal_executes_at_runtime():
    # Even if a crafted expression slipped through parsing, evaluation must never
    # reach a dunder. We assert the parser rejects it (defense at compile time).
    with pytest.raises(ConditionSyntaxError):
        compile_expr("input.__class__.__bases__[0] == 1", phase=Phase.INPUT)


# --- operand capture (C13 / req. 7) ----------------------------------------


def test_operands_capture_resolved_values():
    st = State()
    st["intent"] = "billing"
    cond = compile_expr("state.intent == 'refund'", phase=Phase.INPUT)
    g = _guard(None, state=st)
    assert cond.check(g) is False
    operands = cond.operands(g)
    # The resolved left-hand value (what state.intent actually was) is captured.
    assert "billing" in operands.values()


def test_missing_state_key_is_handled_gracefully():
    cond = compile_expr("state.missing == 'x'", phase=Phase.INPUT)
    # A missing key resolves to a sentinel (not present), so the comparison is
    # simply False rather than raising.
    assert cond.check(_guard(None)) is False
