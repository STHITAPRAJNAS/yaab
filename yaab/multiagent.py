"""Multi-agent orchestration patterns over the one runtime.

These *workflow agents* compose other agents and expose the same ``run`` /
``run_sync`` / ``as_tool`` surface as a plain :class:`~yaab.agent.Agent`, so they
nest arbitrarily and drop into tools, graphs, and servers:

* :class:`SequentialAgent` — run sub-agents in order, sharing one state;
* :class:`ParallelAgent`   — run sub-agents concurrently on the same input;
* :class:`MapAgent`        — fan one sub-agent across many inputs;
* :class:`LoopAgent`       — re-run a sub-agent until a condition or a cap;
* :class:`Swarm`           — autonomous hand-off between peer agents;
* :class:`RouterAgent`     — run exactly one of N branches (exclusive choice).

Every pattern shares **one** :class:`~yaab.state.State` object across all its
children for a run, so a value written by one step is read by the next by key.
A step can declare ``writes="key"`` to capture its (typed) output into that
shared state; the next step reads it via ``{key}`` instruction injection or a
tool. Usage is rolled up across all sub-agents so cost/token accounting stays
whole.

Any unit in a pattern may be wrapped in a :class:`~yaab.conditions.Step` to add
``when=`` (an input guard — *do I run?*), ``stop=`` (an output guard — *does the
pattern stop?*), or ``else_=`` (a fallback unit on skip or failure). The same
three keywords behave identically on every pattern, because every pattern runs
its children through one guarded-execution seam (:func:`_run_guarded`).
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from .conditions import (
    Branch,
    Guard,
    Phase,
    Status,
    Step,
    as_condition,
    as_step,
    make_condition_event,
    unit_name,
)
from .exceptions import RunCancelled, YaabError
from .state import State, StateConflictError, scope_of
from .types import EventType, RunContext, RunResult, Usage


def _capture(state: State, child: Any, result: RunResult) -> None:
    """Land a child's output into shared state under its ``writes=`` key.

    The *typed* ``result.output`` is stored exactly as produced (a model stays a
    model, a list stays a list) — it never round-trips through text. The key's
    prefix chooses the scope (``temp:``/``user:``/``app:``/session) for free. A
    skipped/failed result (no meaningful output) is not captured.
    """
    key = getattr(child, "writes", None)
    if key is not None and result is not None and result.status == Status.OK.value:
        state[key] = result.output


def _step_writes(state: State, step: Step, result: RunResult) -> None:
    """Apply a :class:`Step`'s ``writes=`` capture for an OK result."""
    if step.writes is not None and result is not None and result.status == Status.OK.value:
        state[step.writes] = result.output


def _state_for_run(state: State | None) -> State:
    """Inherit the parent's State, or build a run-local one for a top-level run.

    A workflow agent invoked as a child is handed the parent's State; only the
    outermost entity builds one. A workflow run with a ``session_id`` still lets
    each child's Runner reuse the session — the shared State is the in-run
    communication medium, and child runs persist through their own session seam.
    """
    return state if state is not None else State()


def _is_timeout(exc: BaseException) -> bool:
    """True when an exception represents a deadline/timeout (not a plain cancel)."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    return isinstance(exc, RunCancelled) and getattr(exc, "reason", "") == "timeout"


async def _invoke(
    unit: Any,
    child_input: Any,
    *,
    state: State,
    deps: Any,
    session_id: str | None,
    identity: str | None,
    timeout: float | None,
) -> RunResult:
    """Run one unit, sharing the run's State; forward a per-step timeout to leaf agents."""
    if isinstance(unit, _WorkflowBase):
        # Nested workflows share the State and roll up their own children; they
        # do not take a per-call timeout (each leaf inside them does).
        return await unit.run(
            child_input, deps=deps, session_id=session_id, identity=identity, state=state
        )
    if timeout is not None:
        return await unit.run(
            child_input,
            deps=deps,
            session_id=session_id,
            identity=identity,
            state=state,
            timeout=timeout,
        )
    return await unit.run(
        child_input, deps=deps, session_id=session_id, identity=identity, state=state
    )


async def _run_guarded(
    step: Step,
    child_input: Any,
    *,
    state: State,
    deps: Any,
    session_id: str | None,
    identity: str | None,
    events: list[Any],
    run_id: str,
    agent_name: str,
    pattern: str,
    per_step_timeout: float | None,
) -> RunResult:
    """Run one :class:`Step` under its ``when=``/``else_=`` guards.

    The single place a condition meets execution — every pattern calls this, so
    ``when=``/``else_=`` behave uniformly regardless of pattern. ``stop=`` is the
    enclosing pattern's concern (it inspects the output) and is evaluated by
    :func:`_should_stop`.
    """
    ctx = RunContext(deps=deps, session_id=session_id, identity=identity, state=state)
    ro = ctx.readonly().state

    # --- INPUT GUARD: when= -------------------------------------------------
    if step.when is not None:
        cond = as_condition(step.when, phase=Phase.INPUT)
        guard = Guard(value=child_input, state=ro, ctx=ctx, phase=Phase.INPUT)
        if not cond.check(guard):
            events.append(
                make_condition_event(
                    event_type=EventType.CONDITION_SKIP,
                    unit=unit_name(step),
                    decision="skip",
                    condition=cond,
                    result=False,
                    operands=cond.operands(guard),
                    status=Status.SKIPPED.value,
                    source="when",
                    pattern=pattern,
                    run_id=run_id,
                    agent_name=agent_name,
                )
            )
            if step.else_ is not None:
                return await _run_fallback(
                    step,
                    child_input,
                    Status.SKIPPED,
                    None,
                    state=state,
                    deps=deps,
                    session_id=session_id,
                    identity=identity,
                    events=events,
                    run_id=run_id,
                    agent_name=agent_name,
                    pattern=pattern,
                    per_step_timeout=per_step_timeout,
                )
            # Shape-preserving pass-through: a skipped step does not shrink the
            # pipeline or inject a sentinel; it hands its input through.
            return RunResult(
                output=child_input, status=Status.SKIPPED.value, run_id=unit_name(step)
            )

    # --- RUN (capture failure/timeout for else=) ---------------------------
    step_timeout = step.timeout if step.timeout is not None else per_step_timeout
    try:
        result = await _invoke(
            step.unit,
            child_input,
            state=state,
            deps=deps,
            session_id=session_id,
            identity=identity,
            timeout=step_timeout,
        )
    except (TimeoutError, RunCancelled) as exc:
        status = Status.TIMEOUT if _is_timeout(exc) else Status.FAILED
        return await _on_failure(
            step,
            child_input,
            status,
            exc,
            state=state,
            deps=deps,
            session_id=session_id,
            identity=identity,
            events=events,
            run_id=run_id,
            agent_name=agent_name,
            pattern=pattern,
            per_step_timeout=per_step_timeout,
        )
    except YaabError as exc:
        return await _on_failure(
            step,
            child_input,
            Status.FAILED,
            exc,
            state=state,
            deps=deps,
            session_id=session_id,
            identity=identity,
            events=events,
            run_id=run_id,
            agent_name=agent_name,
            pattern=pattern,
            per_step_timeout=per_step_timeout,
        )

    _step_writes(state, step, result)
    return result


async def _on_failure(
    step: Step,
    child_input: Any,
    status: Status,
    exc: BaseException | None,
    **kwargs: Any,
) -> RunResult:
    """Route a failed/timed-out step to its ``else_=`` fallback, or re-raise."""
    if step.else_ is not None:
        return await _run_fallback(step, child_input, status, exc, **kwargs)
    # No fallback: propagate so the workflow's run() surfaces the error.
    assert exc is not None
    raise exc


async def _run_fallback(
    step: Step,
    child_input: Any,
    status: Status,
    exc: BaseException | None,
    *,
    state: State,
    deps: Any,
    session_id: str | None,
    identity: str | None,
    events: list[Any],
    run_id: str,
    agent_name: str,
    pattern: str,
    per_step_timeout: float | None,
) -> RunResult:
    """Run a step's ``else_=`` unit, recording a fallback decision event."""
    events.append(
        make_condition_event(
            event_type=EventType.CONDITION_FALLBACK,
            unit=unit_name(step),
            decision="fallback",
            condition=None,
            result=True,
            operands={
                "status": status.value,
                "error": type(exc).__name__ if exc is not None else None,
            },
            status=status.value,
            source="else",
            pattern=pattern,
            run_id=run_id,
            agent_name=agent_name,
        )
    )
    return await _run_guarded(
        as_step(step.else_),
        child_input,
        state=state,
        deps=deps,
        session_id=session_id,
        identity=identity,
        events=events,
        run_id=run_id,
        agent_name=agent_name,
        pattern=pattern,
        per_step_timeout=per_step_timeout,
    )


def _should_stop(
    stop_spec: Any,
    result: RunResult,
    *,
    state: State,
    deps: Any,
    session_id: str | None,
    identity: str | None,
    events: list[Any],
    run_id: str,
    agent_name: str,
    pattern: str,
) -> bool:
    """Evaluate a pattern's ``stop=`` output guard against a child's output."""
    if stop_spec is None:
        return False
    ctx = RunContext(deps=deps, session_id=session_id, identity=identity, state=state)
    cond = as_condition(stop_spec, phase=Phase.OUTPUT)
    guard = Guard(
        value=result.output,
        state=ctx.readonly().state,
        ctx=ctx,
        phase=Phase.OUTPUT,
        status=result.status,
    )
    fired = cond.check(guard)
    if fired:
        events.append(
            make_condition_event(
                event_type=EventType.CONDITION_STOP,
                unit=pattern,
                decision="stop",
                condition=cond,
                result=True,
                operands=cond.operands(guard),
                status=result.status,
                source="stop",
                pattern=pattern,
                run_id=run_id,
                agent_name=agent_name,
            )
        )
    return fired


def _coerce_stop(stop: Any, *legacy: Any, names: tuple[str, ...]) -> Any:
    """Resolve the canonical ``stop=`` from it and any deprecated aliases.

    A non-``None`` legacy alias emits a :class:`DeprecationWarning` and is used
    as the stop spec when ``stop=`` itself was not given.
    """
    chosen = stop
    for value, name in zip(legacy, names, strict=False):
        if value is not None:
            warnings.warn(
                f"{name}= is a deprecated alias of stop=; use stop= instead",
                DeprecationWarning,
                stacklevel=3,
            )
            if chosen is None:
                chosen = value
    return chosen


class _WorkflowBase:
    """Shared run/run_sync/as_tool surface for workflow agents."""

    name: str
    #: The shared-state key this unit captures its final output into (or ``None``
    #: to capture nothing). A prefix on the key selects the scope.
    writes: str | None = None

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        session_id: str | None = None,
        identity: str | None = None,
        state: State | None = None,
        resume: Any = None,
    ) -> RunResult[Any]:
        raise NotImplementedError

    def run_sync(self, prompt: str, **kwargs: Any) -> RunResult[Any]:
        return asyncio.run(self.run(prompt, **kwargs))

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> Any:
        from .tools.agent_tool import AgentTool

        return AgentTool(self, name=name, description=description)


class SequentialAgent(_WorkflowBase):
    """Run sub-agents in sequence over one shared state.

    Each child runs against the same :class:`~yaab.state.State`, so a child that
    declares ``writes="key"`` lands its output where the next child can read it
    (by ``{key}`` injection or a tool). ``pipe_output`` (default ``True``) keeps
    the classic convenience of feeding the prior step's text as the next prompt.

    Any child may be a :class:`~yaab.conditions.Step` with ``when=`` (skip the
    step), ``else_=`` (run a fallback instead), or ``writes=``. The agent-level
    ``stop=`` is an output guard checked after each step; ``stop_when=`` /
    ``early_stop=`` are deprecated aliases that emit a ``DeprecationWarning``.
    """

    def __init__(
        self,
        name: str,
        agents: list[Any],
        *,
        pipe_output: bool = True,
        stop: Any = None,
        stop_when: Callable[[Any], bool] | None = None,
        early_stop: Any = None,
        writes: str | None = None,
        per_step_timeout: float | None = None,
    ) -> None:
        self.name = name
        self.agents = agents
        self.pipe_output = pipe_output
        self.stop = _coerce_stop(stop, stop_when, early_stop, names=("stop_when", "early_stop"))
        self.writes = writes
        self.per_step_timeout = per_step_timeout
        self.instructions = f"Sequential pipeline of {len(agents)} agents."

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        session_id=None,
        identity=None,
        state: State | None = None,
        resume: Any = None,
    ):
        shared = _state_for_run(state)
        usage = Usage()
        events: list[Any] = []
        current_input = prompt
        last: RunResult | None = None
        for entry in self.agents:
            step = as_step(entry)
            last = await _run_guarded(
                step,
                current_input,
                state=shared,
                deps=deps,
                session_id=session_id,
                identity=identity,
                events=events,
                run_id=self.name,
                agent_name=self.name,
                pattern="sequential",
                per_step_timeout=self.per_step_timeout,
            )
            usage.add(last.usage)
            if _should_stop(
                self.stop,
                last,
                state=shared,
                deps=deps,
                session_id=session_id,
                identity=identity,
                events=events,
                run_id=self.name,
                agent_name=self.name,
                pattern="sequential",
            ):
                break
            # A skipped step passes its input through (C6), so piping the prior
            # output keeps the pipeline shape unchanged across the skip.
            if self.pipe_output:
                current_input = _as_text(last.output)
        if self.writes is not None and last is not None and last.status == Status.OK.value:
            shared[self.writes] = last.output
        return RunResult(
            output=last.output if last else None,
            status=last.status if last else Status.OK.value,
            usage=usage,
            events=events,
            run_id=self.name,
        )


class ParallelAgent(_WorkflowBase):
    """Run sub-agents concurrently on the same prompt over one shared state.

    All branches see the *same* state object. Each branch that declares
    ``writes="key"`` lands its result under a key downstream steps read by name —
    two branches writing the *same* session-scoped key without a reducer is a
    declared :class:`~yaab.state.StateConflictError`, not a silent clobber. The
    returned output is a ``name -> result`` map.

    A branch may be a :class:`~yaab.conditions.Step` with ``when=``: its guard is
    evaluated against the shared input *before* scheduling, so an excluded branch
    never runs and is, by default, **absent** from the result map (the map stays
    a clean ``name -> output``). Set ``include_skipped=True`` to add
    ``name -> RunResult(status="skipped")`` entries for a total map.
    """

    def __init__(
        self,
        name: str,
        agents: list[Any],
        *,
        writes: str | None = None,
        include_skipped: bool = False,
        per_step_timeout: float | None = None,
    ) -> None:
        self.name = name
        self.agents = agents
        self.writes = writes
        self.include_skipped = include_skipped
        self.per_step_timeout = per_step_timeout
        self.instructions = f"Parallel fan-out across {len(agents)} agents."

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        session_id=None,
        identity=None,
        state: State | None = None,
        resume: Any = None,
    ):
        shared = _state_for_run(state)
        self._check_write_conflicts()
        events: list[Any] = []
        steps = [as_step(entry) for entry in self.agents]

        async def _branch(step: Step) -> tuple[str, RunResult]:
            result = await _run_guarded(
                step,
                prompt,
                state=shared,
                deps=deps,
                session_id=session_id,
                identity=identity,
                events=events,
                run_id=self.name,
                agent_name=self.name,
                pattern="parallel",
                per_step_timeout=self.per_step_timeout,
            )
            return unit_name(step), result

        results = await asyncio.gather(*(_branch(s) for s in steps))
        usage = Usage()
        output: dict[str, Any] = {}
        for branch_name, result in results:
            usage.add(result.usage)
            if result.status == Status.SKIPPED.value:
                if self.include_skipped:
                    output[branch_name] = result
                continue
            output[branch_name] = result.output
        return RunResult(output=output, usage=usage, events=events, run_id=self.name)

    def _check_write_conflicts(self) -> None:
        seen: set[str] = set()
        for entry in self.agents:
            step = as_step(entry)
            key = step.writes if step.writes is not None else getattr(step.unit, "writes", None)
            # Only session-scoped (unprefixed) concurrent writes collide; scoped
            # writes (temp:/user:/app:) are the caller's explicit choice.
            if key is None or scope_of(key) != "session":
                continue
            if key in seen:
                raise StateConflictError(
                    f"parallel branches both write session-scoped key '{key}'; "
                    f"give each branch a distinct writes= key (or a prefixed scope)"
                )
            seen.add(key)


class MapAgent(_WorkflowBase):
    """Fan one sub-agent out across many inputs concurrently over one shared state.

    Given a list of prompts (or a function that derives them from the incoming
    prompt), run the same agent on each in parallel and return the list of
    outputs. ``max_concurrency`` bounds simultaneous runs. The children share the
    run's state object and its ``session_id`` so they replay the same session.

    Wrap the mapped agent in a :class:`~yaab.conditions.Step` with ``when=`` to
    *filter* inputs: each derived input whose guard is false is dropped (absent
    from the results), so ``when="len(input) > 0"`` is the per-input filter.
    """

    def __init__(
        self,
        name: str,
        agent: Any,
        *,
        map_inputs: Callable[[str], list[str]] | None = None,
        max_concurrency: int = 0,
        writes: str | None = None,
        per_step_timeout: float | None = None,
    ) -> None:
        self.name = name
        self.step = as_step(agent)
        self.agent = self.step.unit
        self.map_inputs = map_inputs
        self.max_concurrency = max_concurrency
        self.writes = writes
        self.per_step_timeout = per_step_timeout
        self.instructions = f"Map {unit_name(self.step)} across N inputs."

    async def run(
        self,
        prompt: str | list[str],
        *,
        deps: Any = None,
        session_id=None,
        identity=None,
        state: State | None = None,
        resume: Any = None,
    ):
        shared = _state_for_run(state)
        if isinstance(prompt, list):
            inputs = prompt
        elif self.map_inputs is not None:
            inputs = self.map_inputs(prompt)
        else:
            inputs = [prompt]

        sem = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency > 0 else None
        events: list[Any] = []
        # A per-input step with no writes= (the map's own writes captures the list).
        item_step = Step(
            self.step.unit,
            when=self.step.when,
            else_=self.step.else_,
            timeout=self.step.timeout,
            name=self.step.name,
        )

        async def _one(p: str) -> RunResult:
            if sem is not None:
                async with sem:
                    return await _run_guarded(
                        item_step,
                        p,
                        state=shared,
                        deps=deps,
                        session_id=session_id,
                        identity=identity,
                        events=events,
                        run_id=self.name,
                        agent_name=self.name,
                        pattern="map",
                        per_step_timeout=self.per_step_timeout,
                    )
            return await _run_guarded(
                item_step,
                p,
                state=shared,
                deps=deps,
                session_id=session_id,
                identity=identity,
                events=events,
                run_id=self.name,
                agent_name=self.name,
                pattern="map",
                per_step_timeout=self.per_step_timeout,
            )

        results = await asyncio.gather(*(_one(p) for p in inputs))
        usage = Usage()
        outputs: list[Any] = []
        for r in results:
            usage.add(r.usage)
            # Filtered (skipped) inputs are absent from the results (C6).
            if r.status == Status.SKIPPED.value:
                continue
            outputs.append(r.output)
        if self.writes is not None:
            shared[self.writes] = outputs
        return RunResult(output=outputs, usage=usage, events=events, run_id=self.name)


class LoopAgent(_WorkflowBase):
    """Re-run a sub-agent over one accumulating shared state until a stop condition.

    Each iteration runs against the same :class:`~yaab.state.State`, so a tool
    that increments ``state["count"]`` accumulates across iterations. ``stop=`` is
    the output guard (state-aware: ``stop="state.score >= 0.9"`` reads what each
    iteration wrote); the loop also stops at ``max_iterations``. ``until=`` is a
    deprecated alias of ``stop=``. ``else_=`` runs when the cap is hit without
    ``stop=`` firing. ``pipe_output`` (default ``True``) feeds the prior output
    as the next prompt for the common refine-in-place case.
    """

    def __init__(
        self,
        name: str,
        agent: Any,
        *,
        max_iterations: int = 5,
        stop: Any = None,
        until: Callable[[Any], bool] | None = None,
        else_: Any = None,
        pipe_output: bool = True,
        writes: str | None = None,
        per_step_timeout: float | None = None,
    ) -> None:
        self.name = name
        self.step = as_step(agent)
        self.agent = self.step.unit
        self.max_iterations = max_iterations
        self.stop = _coerce_stop(stop, until, names=("until",))
        self.else_ = else_
        self.pipe_output = pipe_output
        self.writes = writes
        self.per_step_timeout = per_step_timeout
        self.instructions = f"Loop over {unit_name(self.step)} up to {max_iterations}x."

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        session_id=None,
        identity=None,
        state: State | None = None,
        resume: Any = None,
    ):
        shared = _state_for_run(state)
        usage = Usage()
        events: list[Any] = []
        current_input = prompt
        last: RunResult | None = None
        stopped = False
        item_step = Step(
            self.step.unit,
            when=self.step.when,
            else_=self.step.else_,
            timeout=self.step.timeout,
            name=self.step.name,
        )
        for _ in range(self.max_iterations):
            last = await _run_guarded(
                item_step,
                current_input,
                state=shared,
                deps=deps,
                session_id=session_id,
                identity=identity,
                events=events,
                run_id=self.name,
                agent_name=self.name,
                pattern="loop",
                per_step_timeout=self.per_step_timeout,
            )
            usage.add(last.usage)
            if _should_stop(
                self.stop,
                last,
                state=shared,
                deps=deps,
                session_id=session_id,
                identity=identity,
                events=events,
                run_id=self.name,
                agent_name=self.name,
                pattern="loop",
            ):
                stopped = True
                break
            if self.pipe_output:
                current_input = _as_text(last.output)

        # The cap was reached without stop= firing: run the else= fallback (the
        # loop "timed out" on iterations) if one is configured.
        if not stopped and self.else_ is not None and last is not None:
            fb = await _run_guarded(
                as_step(self.else_),
                current_input,
                state=shared,
                deps=deps,
                session_id=session_id,
                identity=identity,
                events=events,
                run_id=self.name,
                agent_name=self.name,
                pattern="loop",
                per_step_timeout=self.per_step_timeout,
            )
            usage.add(fb.usage)
            last = fb

        if self.writes is not None and last is not None and last.status == Status.OK.value:
            shared[self.writes] = last.output
        return RunResult(
            output=last.output if last else None,
            status=last.status if last else Status.OK.value,
            usage=usage,
            events=events,
            run_id=self.name,
        )


class SwarmState(BaseModel):
    """Shared, mutable state threaded through a swarm via DI.

    ``shared`` is kept for backward compatibility; under the unified model the
    swarm's structured state lives on the run's one :class:`~yaab.state.State`,
    and the handoff target is an ordinary run-local state write — not a fourth
    backing store.
    """

    handoff: str | None = None
    shared: dict[str, Any] = Field(default_factory=dict)


class Swarm(_WorkflowBase):
    """Autonomous hand-off between peer agents (swarm) over one shared state.

    Each member is augmented with ``handoff_to_<peer>`` tools. When an agent
    decides another is better suited, it calls the handoff tool; the swarm then
    continues the task with that agent. Every member runs against the same
    :class:`~yaab.state.State`, and the handoff target is recorded as a run-local
    state write (``temp:__handoff__``) rather than a magic attribute on a separate
    object. ``stop=`` is an output guard checked after each member runs (e.g.
    ``stop="len(state['temp:__handoff_log__']) > 4"`` caps a handoff cycle).
    Runs until no further hand-off (or a cap).
    """

    _HANDOFF_KEY = "temp:__handoff__"
    _HANDOFF_LOG = "temp:__handoff_log__"

    def __init__(
        self,
        name: str,
        agents: list[Any],
        *,
        entry: str | None = None,
        max_handoffs: int = 6,
        stop: Any = None,
        writes: str | None = None,
    ) -> None:
        self.name = name
        self.agents = {a.name: a for a in agents}
        self.entry = entry or agents[0].name
        self.max_handoffs = max_handoffs
        self.stop = stop
        self.writes = writes
        self.instructions = f"Swarm of {len(agents)} agents with autonomous hand-off."
        self._install_handoff_tools()

    def _install_handoff_tools(self) -> None:
        for owner in self.agents.values():
            for peer_name in self.agents:
                if peer_name == owner.name:
                    continue
                owner.tools.append(self._make_handoff_tool(peer_name))

    def _make_handoff_tool(self, target: str) -> Any:
        from .tools.base import FunctionTool

        key = self._HANDOFF_KEY

        async def handoff(ctx: RunContext) -> str:
            ctx.state[key] = target
            if isinstance(ctx.deps, SwarmState):
                ctx.deps.handoff = target
            return f"handing off to {target}"

        tool = FunctionTool(
            handoff,
            name=f"handoff_to_{target}",
            description=f"Delegate the task to the '{target}' agent when it is better suited.",
        )
        return tool

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        session_id=None,
        identity=None,
        state: State | None = None,
        resume: Any = None,
    ):
        shared = _state_for_run(state)
        swarm_deps = deps if isinstance(deps, SwarmState) else SwarmState()
        usage = Usage()
        events: list[Any] = []
        shared.setdefault(self._HANDOFF_LOG, [])
        current = self.entry
        current_input = prompt
        last: RunResult | None = None
        for _ in range(self.max_handoffs + 1):
            shared[self._HANDOFF_KEY] = None
            swarm_deps.handoff = None
            agent = self.agents[current]
            last = await agent.run(
                current_input,
                deps=swarm_deps,
                session_id=session_id,
                identity=identity,
                state=shared,
            )
            usage.add(last.usage)
            if _should_stop(
                self.stop,
                last,
                state=shared,
                deps=swarm_deps,
                session_id=session_id,
                identity=identity,
                events=events,
                run_id=self.name,
                agent_name=self.name,
                pattern="swarm",
            ):
                break
            target = shared.get(self._HANDOFF_KEY)
            if target and target in self.agents and target != current:
                shared[self._HANDOFF_LOG].append(target)
                current = target
                current_input = _as_text(last.output) or prompt
                continue
            break
        if self.writes is not None and last is not None:
            shared[self.writes] = last.output
        return RunResult(
            output=last.output if last else None,
            usage=usage,
            events=events,
            run_id=self.name,
        )


class RouterAgent(_WorkflowBase):
    """Exclusive-choice routing — run exactly one of N branches.

    A peer of Sequential/Parallel/Map/Loop/Swarm that evaluates input guards in
    declared order and runs **exactly one** branch (or the default). The routing
    decision spends **zero model calls** — it is a plain-Python/expression
    picker, so it is deterministic and fully auditable. Because exactly one
    branch runs and its output is returned unmodified, there is no skip-cascade
    and no merge node.

    ``on_no_match`` is ``"default"`` (run the default branch) or ``"error"``
    (raise). A router with no matching branch and no default returns a result
    with status ``"skipped"``. ``writes=`` captures the chosen branch's output
    into shared state. Nests like every workflow agent and works as a tool.
    """

    def __init__(
        self,
        name: str,
        branches: list[Branch],
        *,
        default: Any = None,
        on_no_match: str = "default",
        writes: str | None = None,
    ) -> None:
        if not branches and default is None:
            raise ValueError("RouterAgent needs at least one branch or a default")
        if on_no_match not in ("default", "error"):
            raise ValueError("on_no_match must be 'default' or 'error'")
        self.name = name
        self.branches = list(branches)
        self.default = default
        self.on_no_match = on_no_match
        self.writes = writes
        self.instructions = (
            f"Route to one of {len(branches)} branches "
            f"(default={'yes' if default else 'no'}, on_no_match={on_no_match})."
        )

    @classmethod
    def from_picker(
        cls,
        name: str,
        picker: Callable[[Any, Any], str],
        to: dict[str, Any],
        *,
        default: Any = None,
        on_no_match: str = "default",
        writes: str | None = None,
    ) -> RouterAgent:
        """Build a router from a label-returning picker.

        ``picker(input, ctx) -> label`` classifies the input; ``to`` maps each
        label to an agent. A picker may only return a key present in ``to`` — an
        unknown label raises immediately (a typo is a loud error, never a silent
        fall-through to the default). The ``str``-returning picker is adapted to
        ``bool`` guards here, so each :class:`Branch` is always a boolean guard.
        """
        keys = set(to)

        def _guard_for(key: str) -> Callable[[Any, Any], bool]:
            def _g(value: Any, ctx: Any) -> bool:
                label = picker(value, ctx)
                if label not in keys:
                    raise ValueError(
                        f"router {name!r}: picker returned unknown label {label!r}; "
                        f"expected one of {sorted(keys)}"
                    )
                return label == key

            return _g

        branches = [Branch(when=_guard_for(k), agent=a, name=k) for k, a in to.items()]
        return cls(name, branches, default=default, on_no_match=on_no_match, writes=writes)

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        session_id=None,
        identity=None,
        state: State | None = None,
        resume: Any = None,
    ):
        shared = _state_for_run(state)
        ctx = RunContext(deps=deps, session_id=session_id, identity=identity, state=shared)
        ro = ctx.readonly().state
        usage = Usage()
        events: list[Any] = []
        labels = [b.name or getattr(b.agent, "name", "?") for b in self.branches]
        events.append(
            make_condition_event(
                event_type=EventType.ROUTER_EVALUATED,
                unit=self.name,
                decision="route",
                condition=None,
                result=True,
                operands={"branches": labels, "on_no_match": self.on_no_match},
                status=Status.OK.value,
                source="when",
                pattern="router",
                run_id=self.name,
                agent_name=self.name,
            )
        )

        chosen: Any = None
        chosen_label: str | None = None
        for i, br in enumerate(self.branches):
            cond = as_condition(br.when, phase=Phase.INPUT)
            guard = Guard(value=prompt, state=ro, ctx=ctx, phase=Phase.INPUT)
            if cond.check(guard):
                chosen = br.agent
                chosen_label = br.name or getattr(br.agent, "name", f"branch{i}")
                events.append(
                    make_condition_event(
                        event_type=EventType.ROUTER_MATCHED,
                        unit=self.name,
                        decision="route",
                        condition=cond,
                        result=True,
                        operands={**cond.operands(guard), "branch": chosen_label, "index": i},
                        status=Status.OK.value,
                        source="when",
                        pattern="router",
                        run_id=self.name,
                        agent_name=self.name,
                    )
                )
                break

        if chosen is None:
            if self.on_no_match == "error":
                raise ValueError(f"router {self.name!r}: no branch matched (on_no_match='error')")
            if self.default is None:
                events.append(
                    make_condition_event(
                        event_type=EventType.ROUTER_MATCHED,
                        unit=self.name,
                        decision="route",
                        condition=None,
                        result=False,
                        operands={"branch": None},
                        status=Status.SKIPPED.value,
                        source="when",
                        pattern="router",
                        run_id=self.name,
                        agent_name=self.name,
                    )
                )
                return RunResult(
                    output=None,
                    status=Status.SKIPPED.value,
                    usage=usage,
                    events=events,
                    run_id=f"{self.name}->none",
                )
            chosen = self.default
            chosen_label = "default"
            events.append(
                make_condition_event(
                    event_type=EventType.ROUTER_MATCHED,
                    unit=self.name,
                    decision="route",
                    condition=None,
                    result=True,
                    operands={"branch": "default"},
                    status=Status.OK.value,
                    source="when",
                    pattern="router",
                    run_id=self.name,
                    agent_name=self.name,
                )
            )

        result = await chosen.run(
            prompt, deps=deps, session_id=session_id, identity=identity, state=shared
        )
        usage.add(result.usage)
        if self.writes is not None and result.status == Status.OK.value:
            shared[self.writes] = result.output
        return RunResult(
            output=result.output,
            status=result.status,
            usage=usage,
            events=events + list(result.events),
            run_id=f"{self.name}->{chosen_label}",
        )


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return str(value)


__all__ = [
    "SequentialAgent",
    "ParallelAgent",
    "MapAgent",
    "LoopAgent",
    "Swarm",
    "SwarmState",
    "RouterAgent",
    "Branch",
    "Step",
]
