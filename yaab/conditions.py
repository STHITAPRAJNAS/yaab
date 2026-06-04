"""Conditions — one decision concept that reads the same state everywhere.

A *condition* answers one of three questions about a unit of work, and it does
so identically on every composition pattern, sub-agent list, and tool list:

* ``when=`` — *"do I run?"* An **input guard**, asked **before** a unit runs.
  Sees ``(input, state, ctx)``. False means the unit is **skipped**.
* ``stop=`` — *"does the pattern stop now?"* An **output guard**, asked
  **after** a unit runs. Sees ``(output, state, ctx)``. True stops the
  enclosing pattern.
* ``else=`` — *"what runs instead?"* A fallback **unit**, run when the guarded
  unit is skipped (by ``when=``) **or** fails/times out.

Every guard is evaluated against one object, :class:`Guard`, that collapses each
condition source into a single shape. ``Guard.state`` is the run's
:class:`~yaab.state.ReadonlyState` — the *same* read-only view instruction
rendering uses — so a condition reads exactly the values a tool wrote, a branch
produced, or an instruction rendered. A condition can read shared state; it
physically cannot mutate it.

A condition may be written as a Python callable or as a sandboxed string
expression (:mod:`yaab.expr`); the two compile to the same ``Guard -> bool``
over the same data, so swapping one for the other never changes behavior.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic

from .state import ReadonlyState
from .types import Deps, RunContext


class Phase(str, Enum):
    """Which kind of guard a condition is — fixed by the call site, not the user.

    ``INPUT`` guards (``when=``) run before a unit and bind ``Guard.value`` to
    the unit's input; ``OUTPUT`` guards (``stop=``) run after and bind it to the
    output. The phase is what makes ``"output.score >= 0.9"`` meaningful under
    ``stop=`` and a load-time error under ``when=`` — never a silent mis-read.
    """

    INPUT = "input"
    OUTPUT = "output"


class Status(str, Enum):
    """The terminal status of a guarded unit, orthogonal to its output value.

    ``OK`` — ran and produced a meaningful output. ``SKIPPED`` — a ``when=``
    guard was false. ``FAILED`` — the unit raised. ``TIMEOUT`` — the unit (or
    its enclosing loop) hit a deadline / iteration cap. ``output`` is meaningful
    only when the status is ``OK``.
    """

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Guard(Generic[Deps]):
    """The single context every :class:`Condition` is evaluated against.

    ``value`` is the unit's **input** under a ``when=`` guard and its **output**
    under a ``stop=`` guard — its meaning is fixed by ``phase``, never ambiguous.
    ``state`` is the run's :class:`~yaab.state.ReadonlyState`: a read-only view
    over the one shared :class:`~yaab.state.State`. A condition can read it; it
    cannot mutate it. ``status``/``error`` carry the failure/timeout channel,
    kept separate from ``value`` so "this step failed" and "this step's output
    is bad" are distinct, testable conditions.
    """

    value: Any
    state: ReadonlyState
    ctx: RunContext[Deps]
    phase: Phase
    status: str = Status.OK.value
    error: BaseException | None = None


CondFn = Callable[["Guard"], bool]
ProbeFn = Callable[["Guard"], "dict[str, Any]"]


class Condition(Generic[Deps]):
    """A composable, observable boolean test over a :class:`Guard`.

    Build one from a predicate or a safe expression string; compose with
    ``&`` / ``|`` / ``~``; evaluate with :meth:`check`. ``label`` is the
    human-readable form used in decision events; for string forms a *probe*
    captures the resolved operand values so a trace answers *why* a guard fired.
    """

    __slots__ = ("_fn", "label", "_expr", "_probe")

    def __init__(
        self,
        fn: CondFn,
        *,
        label: str = "",
        expr: str | None = None,
        probe: ProbeFn | None = None,
    ) -> None:
        self._fn = fn
        self.label = label or expr or getattr(fn, "__name__", "condition")
        self._expr = expr
        self._probe = probe

    def check(self, guard: Guard) -> bool:
        """Evaluate the condition against ``guard`` and return a plain bool."""
        return bool(self._fn(guard))

    def operands(self, guard: Guard) -> dict[str, Any]:
        """The resolved operand values this check saw, for decision events."""
        if self._probe is not None:
            return self._probe(guard)
        return {"value": _reprish(guard.value), "status": guard.status}

    def __and__(self, other: Condition) -> Condition:
        return Condition(
            lambda g: self.check(g) and other.check(g),
            label=f"({self.label} & {other.label})",
        )

    def __or__(self, other: Condition) -> Condition:
        return Condition(
            lambda g: self.check(g) or other.check(g),
            label=f"({self.label} | {other.label})",
        )

    def __invert__(self) -> Condition:
        return Condition(lambda g: not self.check(g), label=f"~{self.label}")

    def __repr__(self) -> str:
        return f"Condition({self.label!r})"


def _reprish(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            return repr(value)
    return repr(value)


def _arity(fn: Callable[..., Any]) -> int:
    """Count the positional parameters a predicate accepts (best-effort)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 1
    count = 0
    for p in params:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            count += 1
        elif p.kind == p.VAR_POSITIONAL:
            return 2  # accepts (value, ctx, ...) — treat as the two-arg form
    return count


def _name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", "condition")


def as_condition(spec: Any, *, phase: Phase, label: str = "") -> Condition:
    """Coerce any accepted spec into one :class:`Condition` over a :class:`Guard`.

    Accepts a :class:`Condition` (returned as-is), a ``bool`` constant, a
    sandboxed expression string (:mod:`yaab.expr`), or a callable of either
    ``(value)`` or ``(value, ctx)``. The ``phase`` is supplied by the call site
    (``when=`` passes INPUT, ``stop=`` passes OUTPUT), which is what guarantees
    an ``output`` reference can never be evaluated by an input guard.
    """
    if isinstance(spec, Condition):
        return spec
    if isinstance(spec, bool):
        const = bool(spec)
        return Condition(lambda g: const, label=str(spec))
    if isinstance(spec, str):
        from .expr import compile_expr

        return compile_expr(spec, phase=phase)
    if callable(spec):
        if _arity(spec) >= 2:
            return Condition(lambda g: spec(g.value, g.ctx), label=label or _name(spec))
        return Condition(lambda g: spec(g.value), label=label or _name(spec))
    raise TypeError(f"not a condition: {spec!r}")


# --- module-level builders / helpers ---------------------------------------


def when(spec: Any, *, label: str = "") -> Condition:
    """Build an input-phase :class:`Condition` from a spec (string or callable)."""
    return as_condition(spec, phase=Phase.INPUT, label=label)


def and_(*conditions: Condition) -> Condition:
    """Conjunction of conditions (all must hold)."""
    if not conditions:
        return always()
    result = conditions[0]
    for c in conditions[1:]:
        result = result & c
    return result


def or_(*conditions: Condition) -> Condition:
    """Disjunction of conditions (any may hold)."""
    if not conditions:
        return Condition(lambda g: False, label="never")
    result = conditions[0]
    for c in conditions[1:]:
        result = result | c
    return result


def not_(condition: Condition) -> Condition:
    """Negation of a condition."""
    return ~condition


def always() -> Condition:
    """A condition that is always true."""
    return Condition(lambda g: True, label="always()")


def failed() -> Condition:
    """True when the guarded unit's status is ``FAILED``."""
    return Condition(lambda g: g.status == Status.FAILED.value, label="failed()")


def timed_out() -> Condition:
    """True when the guarded unit's status is ``TIMEOUT``."""
    return Condition(lambda g: g.status == Status.TIMEOUT.value, label="timed_out()")


def ok() -> Condition:
    """True when the guarded unit ran cleanly (status ``OK``)."""
    return Condition(lambda g: g.status == Status.OK.value, label="ok()")


def skipped() -> Condition:
    """True when the guarded unit was skipped by a ``when=`` guard."""
    return Condition(lambda g: g.status == Status.SKIPPED.value, label="skipped()")


def loop_exhausted() -> Condition:
    """True when a loop reached its iteration cap without ``stop=`` firing.

    A capped loop carries the ``TIMEOUT`` status (it ran out of room, not out of
    correctness), so ``LoopAgent(..., else_=review)`` fires on exhaustion.
    """
    return timed_out()


def output_contains(needle: Any) -> Condition:
    """Output-phase condition: the output contains ``needle`` (substring/member)."""

    def _check(g: Guard) -> bool:
        try:
            return needle in g.value
        except TypeError:
            return False

    return Condition(_check, label=f"output_contains({needle!r})")


def state_eq(key: str, value: Any) -> Condition:
    """Condition: ``state[key] == value`` (a missing key is simply not equal)."""

    def _check(g: Guard) -> bool:
        try:
            return g.state[key] == value
        except KeyError:
            return False

    return Condition(_check, label=f"state[{key!r}] == {value!r}")


def state_ge(key: str, value: Any) -> Condition:
    """Condition: ``state[key] >= value`` (a missing key is False)."""

    def _check(g: Guard) -> bool:
        try:
            return g.state[key] >= value
        except (KeyError, TypeError):
            return False

    return Condition(_check, label=f"state[{key!r}] >= {value!r}")


# --- the one wrapper carrying conditional metadata for any unit ------------


@dataclass
class Step:
    """The uniform carrier of conditional metadata for any unit.

    Wraps an :class:`~yaab.agent.Agent`, a workflow agent, a tool, or a
    sub-agent with optional ``when=`` (input guard), ``stop=`` (output guard),
    ``else_=`` (fallback unit on skip or failure), ``writes=`` (capture the
    unit's output into shared state under a key), and a per-step ``timeout``.

    A bare unit in a list is implicitly ``Step(unit)`` with no guards, which is
    what keeps every existing plain ``[agent_a, agent_b]`` list working.
    """

    unit: Any
    when: Any = None
    stop: Any = None
    else_: Any = None
    writes: str | None = None
    timeout: float | None = None
    name: str | None = None


@dataclass
class Branch:
    """One guarded branch of a :class:`~yaab.multiagent.RouterAgent`.

    ``when`` is the input-guard form of a :class:`Condition` (a Condition, a
    ``(input, ctx) -> bool`` callable, or an expression string over
    ``input``/``state``/``deps``). Branches are evaluated in declared order; the
    first whose guard is true is the only agent run.
    """

    when: Any
    agent: Any
    name: str | None = None


def as_step(entry: Any) -> Step:
    """Wrap a bare unit in a :class:`Step`; pass an existing Step through."""
    return entry if isinstance(entry, Step) else Step(entry)


def unit_name(entry: Any) -> str:
    """The name of a unit or step, for events/labels."""
    step = entry if isinstance(entry, Step) else None
    unit = step.unit if step is not None else entry
    if step is not None and step.name:
        return step.name
    return getattr(unit, "name", getattr(unit, "__name__", "unit"))


# --- decision events -------------------------------------------------------


@dataclass
class _DecisionEvent:
    """A captured decision (skip/stop/fallback/route) for the event stream.

    Buffered by the workflow patterns (which have no Runner ``emit`` seam of
    their own) and materialized into :class:`~yaab.types.Event` objects on the
    final :class:`~yaab.types.RunResult`.
    """

    type: Any
    unit: str
    pattern: str
    payload: dict[str, Any] = field(default_factory=dict)


def make_condition_event(
    *,
    event_type: Any,
    unit: str,
    decision: str,
    condition: Condition | None,
    result: bool,
    operands: dict[str, Any],
    status: str,
    source: str,
    pattern: str,
    run_id: str,
    agent_name: str,
) -> Any:
    """Build a JSON-safe decision :class:`~yaab.types.Event`."""
    from .types import Event

    return Event(
        type=event_type,
        agent=agent_name,
        run_id=run_id,
        payload={
            "unit": unit,
            "decision": decision,
            "condition": condition.label if condition is not None else None,
            "result": result,
            "operands": operands,
            "status": status,
            "source": source,
            "pattern": pattern,
        },
    )


__all__ = [
    "Phase",
    "Status",
    "Guard",
    "Condition",
    "Step",
    "Branch",
    "as_condition",
    "as_step",
    "unit_name",
    "when",
    "and_",
    "or_",
    "not_",
    "always",
    "failed",
    "timed_out",
    "ok",
    "skipped",
    "loop_exhausted",
    "output_contains",
    "state_eq",
    "state_ge",
    "make_condition_event",
]
