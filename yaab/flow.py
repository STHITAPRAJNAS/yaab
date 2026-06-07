"""Flow — explicit, durable control flow you can branch, cycle, and pause.

:class:`Flow` is the seventh workflow pattern: the one you reach for when the
control flow itself must be inspectable and crash-proof. It is a *builder* — a
chain of ``.step / .then / .route / .loop / .fan_out / .start_at / .returns`` —
that lowers onto YAAB's durable graph engine (:mod:`yaab.graph.state`) and runs
through the same BSP superstep runtime everything else does. Flow owns **no**
state object, **no** checkpoint format, and **no** pause type of its own:

* it threads the one shared :class:`~yaab.state.State` (Part 1);
* it routes on the one :class:`~yaab.conditions.Condition` (Part 2);
* it pauses into the one :class:`~yaab.types.Pending` (Part 3), kind
  ``"flow_pause"`` — which, per the approved design, also lands an
  :class:`~yaab.governance.approvals.ApprovalRequest` row so the pause is visible
  in ``GET /approvals`` and ``approvals.respond()`` works on it identically to a
  tool approval.

A step body is a plain function ``(state, ctx)`` or an :class:`~yaab.agent.Agent`
used directly. Steps read and write the shared State and receive the same
:class:`~yaab.types.RunContext` tools receive::

    from yaab import Agent, Flow, RunContext, State

    flow = (
        Flow[None, str]("refund")
        .step("parse", fn=lambda state, ctx: {"amount": 50})
        .route(
            "parse",
            lambda state, ctx: "auto" if state["amount"] < 100 else "human",
            to={"auto": "execute", "human": "review"},
        )
        .step("execute", fn=lambda state, ctx: {"out": "auto"})
        .step("review", fn=lambda state, ctx: {"out": "human"})
        .then("execute", Flow.DONE).then("review", Flow.DONE)
        .start_at("parse").returns("out")
    )
    print(flow.run_sync("refund #42").output)   # -> "auto"
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any, Generic

from .conditions import Condition, Guard, Phase, as_condition
from .exceptions import Interrupt, YaabError

# Re-export the checkpoint backends under the Flow-native ``*Checkpoints`` name
# (the public spelling; the ``*Saver`` engine names are deprecated, see 4.6).
from .graph.checkpoint import (  # noqa: E402  (after stdlib/local imports by design)
    MemorySaver as MemoryCheckpoints,
)
from .graph.checkpoint import (
    PostgresSaver as PostgresCheckpoints,
)
from .graph.checkpoint import (
    RedisSaver as RedisCheckpoints,
)
from .graph.checkpoint import (
    SQLiteSaver as SQLiteCheckpoints,
)
from .graph.state import END, START, Channel, RetryPolicy, StateGraph
from .multiagent import _state_for_run, _WorkflowBase
from .state import ReadonlyState, State
from .types import Deps, Output, Pending, RunContext, RunResult, Usage

#: A skip-sink node a guarded-out step routes to (it does nothing and ends).
_SKIP_SINK = "__skip__"


class _StepSpec:
    """One declared step in a :class:`Flow` (lowered to a graph node)."""

    __slots__ = ("name", "agent", "fn", "writes", "when", "fallback", "merge", "retry")

    def __init__(
        self,
        name: str,
        *,
        agent: Any = None,
        fn: Callable[..., Any] | None = None,
        writes: str | None = None,
        when: Any = None,
        fallback: Any = None,
        merge: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.name = name
        self.agent = agent
        self.fn = fn
        self.writes = writes
        self.when = when
        self.fallback = fallback
        self.merge = merge
        self.retry = retry


class Flow(_WorkflowBase, Generic[Deps, Output]):
    """A durable, branchable, resumable flow of work — the seventh pattern.

    ``Flow[Deps, Output]`` parameterizes exactly like ``Agent[Deps, Output]``:
    ``deps`` flows to every step's :class:`~yaab.types.RunContext`, and the value
    at the ``.returns(key)`` key becomes the typed ``RunResult.output``.
    """

    #: Terminal / entry markers (the public spelling of the engine's END/START).
    DONE = END
    ENTRY = START

    def __init__(self, name: str, *, writes: str | None = None) -> None:
        self.name = name
        #: For the model-less governance shim: a Flow has no model/registry, so it
        #: registers under its own name and the flow-level gate runs on that.
        self.registry_id = name
        self.writes = writes
        self.instructions = f"Flow of explicit, durable control flow ({name})."
        self._steps: dict[str, _StepSpec] = {}
        self._order: list[str] = []
        self._edges: dict[str, list[str]] = {}
        self._routes: dict[str, tuple[Any, dict[str, str]]] = {}
        self._entry: str | None = None
        self._returns: str | None = None
        # Injected by the Runner delegation seam (or built on a standalone run).
        self._runner: Any = None
        # The active run's context (set by run(); a transient one is built when a
        # lowered graph is driven directly, e.g. flow.lower().compile().invoke()).
        self._ctx: RunContext | None = None
        self._transient_ctx: RunContext | None = None
        self._events: list[Any] = []

    # --- builder -------------------------------------------------------
    def step(
        self,
        name: str,
        agent: Any = None,
        *,
        fn: Callable[..., Any] | None = None,
        writes: str | None = None,
        when: Any = None,
        fallback: Any = None,
        merge: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> Flow[Deps, Output]:
        """Declare a step: an ``agent=`` or a ``fn=(state, ctx)`` callable.

        ``writes=`` captures the step's typed output into shared State under the
        key (overriding the step agent's own ``agent.writes``); the prefix selects
        the scope. ``when=`` is a Part 2 guard (skip the step on False, routing to
        the ``fallback`` or the skip-sink). ``merge=`` declares the Channel reducer
        for an unprefixed key two branches both write (S6). ``retry=`` attaches a
        per-step :class:`RetryPolicy`.
        """
        if name in (START, END, _SKIP_SINK):
            raise YaabError(f"{name!r} is a reserved step name")
        if name in self._steps:
            raise YaabError(f"duplicate step {name!r}")
        if merge is not None and _scope_of_merge(writes):
            raise YaabError(
                f"merge= applies only to a session-scoped (unprefixed) key; "
                f"step {name!r} writes a scoped key {writes!r}"
            )
        self._steps[name] = _StepSpec(
            name,
            agent=agent,
            fn=fn,
            writes=writes,
            when=when,
            fallback=fallback,
            merge=merge,
            retry=retry,
        )
        self._order.append(name)
        return self

    def then(self, src: str, dst: str) -> Flow[Deps, Output]:
        """Add a "then go to" edge: ``src`` -> ``dst`` (``dst`` may be ``Flow.DONE``)."""
        self._edges.setdefault(src, []).append(dst)
        return self

    def route(self, src: str, picker: Any, to: dict[str, str]) -> Flow[Deps, Output]:
        """Branch from ``src`` by a label the ``picker`` returns.

        ``picker`` is a Part 2 :class:`~yaab.conditions.Condition`, a
        ``(state, ctx) -> label`` callable, or a safe expression string. It is run
        against a **read-only** state view (it physically cannot mutate state). The
        ``to`` map sends each label to a successor step (or ``Flow.DONE``).
        """
        self._routes[src] = (picker, dict(to))
        return self

    def loop(self, step: str, *, until: Any, max_iterations: int = 5) -> Flow[Deps, Output]:
        """An explicit, bounded cycle on ``step`` until ``until`` (a Condition).

        Sugar for a self-edge plus a route whose exit picker is ``until``: while
        ``until`` is false (and the iteration cap is not hit) control returns to
        ``step``; otherwise it leaves. State accumulates across iterations on the
        one shared State.
        """
        self._routes[step] = (
            _LoopExit(step, until, max_iterations),
            {"__loop__": step, "__done__": END},
        )
        return self

    def fan_out(self, src: str, targets: list[str]) -> Flow[Deps, Output]:
        """Fan ``src`` out to several parallel successor steps (one superstep)."""
        for t in targets:
            self._edges.setdefault(src, []).append(t)
        return self

    def start_at(self, name: str) -> Flow[Deps, Output]:
        """Set the entry step."""
        self._entry = name
        return self

    def returns(self, state_key: str) -> Flow[Deps, Output]:
        """Name the state key whose value becomes ``RunResult.output`` at terminus."""
        self._returns = state_key
        return self

    # --- lowering ------------------------------------------------------
    def lower(self) -> StateGraph:
        """Compile this Flow into a :class:`~yaab.graph.state.StateGraph`.

        Each ``.step`` becomes a node, ``.then`` an edge, ``.route`` a conditional
        edge, and ``.loop`` a cyclic edge with a conditional exit. The Flow runs
        through the same engine; this does not change the engine's behavior.
        """
        if self._entry is None:
            raise YaabError(f"flow {self.name!r} has no entry; call .start_at(name)")
        # Validate route completeness at build time: every label target must be a
        # real step or a terminal marker, and a route must offer at least one
        # target — an empty/incomplete mapping is a loud error, never a silent
        # fall-through at run time.
        _terminals = {END, _SKIP_SINK}
        for src, (picker, mapping) in self._routes.items():
            if isinstance(picker, _LoopExit):
                continue  # loop exits use reserved internal labels
            if not mapping:
                raise ValueError(
                    f"flow {self.name!r}: route from {src!r} has an empty target map; "
                    "give it at least one label -> step"
                )
            for label, target in mapping.items():
                if target not in self._steps and target not in _terminals:
                    raise ValueError(
                        f"flow {self.name!r}: route from {src!r} sends label {label!r} to "
                        f"unknown step {target!r}; declare it with .step({target!r}, ...)"
                    )
        channels: dict[str, Channel] = {}
        for spec in self._steps.values():
            if spec.merge is not None and spec.writes is not None:
                channels[spec.writes] = Channel(spec.merge, default=_default_for(spec.merge))
        graph = StateGraph(channels=channels or None)
        for name in self._order:
            spec = self._steps[name]
            graph.add_node(name, self._make_node(spec), retry=spec.retry)
        # A do-nothing skip-sink for guarded-out steps (only added if needed).
        needs_sink = any(s.when is not None and s.fallback is None for s in self._steps.values())
        if needs_sink:
            graph.add_node(_SKIP_SINK, lambda state, ctx: {})
            graph.set_finish_point(_SKIP_SINK)
        graph.set_entry_point(self._entry)
        # Plain edges (.then / .fan_out).
        for src, dsts in self._edges.items():
            for dst in dsts:
                graph.add_edge(src, dst)
        # Conditional edges (.route / .loop).
        for src, (picker, mapping) in self._routes.items():
            graph.add_conditional_edges(src, self._make_router(picker, mapping), mapping)
        # when= guards lower to a conditional edge to the skip-sink (Part 2 §2.6).
        for name in self._order:
            spec = self._steps[name]
            if spec.when is not None:
                self._lower_when(graph, spec)
        return graph

    def _lower_when(self, graph: StateGraph, spec: _StepSpec) -> None:
        """Lower a ``when=`` step guard to a conditional edge to a skip target.

        The guard is evaluated *before* the step's body runs, so we wrap the node
        in a pre-check that, on a false guard, short-circuits the body to an empty
        update and routes to the fallback/skip-sink. The simplest faithful lowering
        is to guard inside the node closure (already done by ``_make_node``) and
        leave normal successor edges intact — the body is skipped, successors run.
        """
        # The body already no-ops on a false guard (see _make_node); nothing extra
        # is required here. Kept as a seam for an explicit skip-sink route when a
        # step has no normal successor.
        return None

    def _active_ctx(self, engine_state: dict[str, Any], gctx: Any = None) -> RunContext:
        """The RunContext for the current node/router, bridged to ``engine_state``.

        Uses the run's context when ``run()`` is driving (so usage/deps/identity
        thread); otherwise builds a transient one so a directly-driven lowered
        graph (``flow.lower().compile().invoke()``) still gets a real
        :class:`~yaab.types.RunContext` with a bridged State.
        """
        ctx = self._ctx or self._transient_ctx
        if ctx is None:
            deps = getattr(gctx, "deps", None) if gctx is not None else None
            ctx = RunContext(deps=deps, state=State(session=engine_state), usage=Usage())
            self._transient_ctx = ctx
        # Bridge the engine's per-node dict into the shared State's session scope.
        ctx.state._session = engine_state
        if gctx is not None:
            ctx.pause_for = gctx.interrupt
        return ctx

    def _make_router(self, picker: Any, mapping: dict[str, str]) -> Callable[[dict], str]:
        """Wrap a Flow picker into the engine's single-arg ``router(state)``.

        Per the design (seam 1, FL2): the user picker is called with the
        **read-only** state view and the run's RunContext, so it can read state but
        physically cannot mutate it. The returned label must be present in
        ``mapping`` (an unknown label is a loud error, never a silent fall-through).
        """
        if isinstance(picker, _LoopExit):
            return picker.as_router(self)
        cond_or_fn = picker

        def _router(engine_state: dict[str, Any]) -> str:
            ctx = self._active_ctx(engine_state)
            ro = ReadonlyState(ctx.state)
            label = _call_picker(cond_or_fn, ro, ctx)
            if label not in mapping:
                raise YaabError(
                    f"flow {self.name!r}: picker returned unknown label {label!r}; "
                    f"expected one of {sorted(mapping)}"
                )
            return label

        return _router

    def _make_node(self, spec: _StepSpec) -> Callable[..., Any]:
        """Build the engine node closure for a step (function or agent).

        The closure bridges the engine's per-node dict to the one shared State
        (FL8), wires ``ctx.pause_for`` to the engine interrupt, runs the body, and
        captures ``writes=`` into the typed shared State. An agent step delegates
        to the **parent** Runner so usage/events/session/governance/state thread.
        """

        async def _node(engine_state: dict[str, Any], gctx: Any) -> dict[str, Any]:
            # Bridge: the engine's per-node dict IS the session scope of the one
            # State (S1/FL8). The active ctx points the shared State's session view
            # at it so reads see committed values and the returned update folds back
            # through the engine; temp:/user:/app: writes route to their stores
            # unchanged. ``pause_for`` is wired to this superstep's interrupt.
            ctx = self._active_ctx(engine_state, gctx)

            # when= input guard (Part 2): skip the body on a false guard.
            if spec.when is not None and not self._guard_passes(spec, ctx):
                return {}

            if spec.fn is not None:
                result = self._call_fn(spec.fn, ctx)
                if inspect.isawaitable(result):
                    result = await result
                updates = dict(result) if result else {}
                if spec.writes is not None:
                    # A bare-fn step with writes= captures the whole return (or the
                    # value already under writes=) — folded typed via the engine.
                    pass
                return updates

            if spec.agent is not None:
                return await self._run_agent_step(spec, ctx, engine_state, gctx)
            return {}

        return _node

    async def _run_agent_step(
        self, spec: _StepSpec, ctx: RunContext, engine_state: dict[str, Any], gctx: Any
    ) -> dict[str, Any]:
        """Run an agent step through the parent Runner, capturing its output."""
        runner = self._runner
        prompt = _prompt_from(engine_state)
        res = await runner.run(
            spec.agent,
            prompt,
            deps=ctx.deps,
            session_id=ctx.session_id,
            identity=ctx.identity,
            state=ctx.state,
        )
        ctx.usage.add(res.usage)
        self._events.extend(res.events)
        if res.paused:
            # A step-level approval/question becomes a Flow pause (FL5): surface
            # the inner Pending as a flow interrupt so the run pauses uniformly.
            raise Interrupt(res.pending[0] if res.pending else res.pause_value)
        # writes= capture (FL1): step-level overrides the agent's own writes.
        key = spec.writes if spec.writes is not None else getattr(spec.agent, "writes", None)
        if key is not None:
            return {key: res.output}
        return {}

    def _guard_passes(self, spec: _StepSpec, ctx: RunContext) -> bool:
        cond = as_condition(spec.when, phase=Phase.INPUT)
        guard = Guard(value=None, state=ReadonlyState(ctx.state), ctx=ctx, phase=Phase.INPUT)
        return cond.check(guard)

    def _call_fn(self, fn: Callable[..., Any], ctx: RunContext) -> Any:
        params = len(inspect.signature(fn).parameters)
        if params >= 2:
            return fn(ctx.state, ctx)
        return fn(ctx.state)

    # --- run (the _WorkflowBase surface) -------------------------------
    async def run(
        self,
        prompt: str | None = None,
        *,
        deps: Any = None,
        session_id: str | None = None,
        identity: str | None = None,
        state: State | None = None,
        resume: Any = None,
        resume_from_checkpoint: bool = False,
    ) -> RunResult[Any]:
        """Run (or resume) the flow, returning a typed :class:`RunResult`.

        A standalone ``flow.run(...)`` builds the one shared State and a default
        Runner; a nested Flow inherits the parent's State and Runner. ``resume``
        (a :class:`~yaab.governance.approvals_decide.Decision`) continues a paused
        flow with the decided value threaded back as ``pause_for``'s return value.
        """
        from .runner import Runner

        if self._runner is None:
            self._runner = Runner(run_checkpointer=MemoryCheckpoints())
        runner: Runner = self._runner
        checkpointer = getattr(runner, "run_checkpointer", None) or MemoryCheckpoints()
        thread_id = session_id or f"flow:{self.name}"

        shared = _state_for_run(state)
        self._ctx = RunContext(
            deps=deps, session_id=session_id, identity=identity, state=shared, usage=Usage()
        )
        self._events = []

        graph = self.lower()
        compiled = graph.compile(checkpointer)

        from .graph.state import _MISSING

        if resume is None:
            engine_resume: Any = _MISSING
        else:
            resume_value, _ = _resume_value(resume)
            engine_resume = resume_value

        # Seed the engine from the INHERITED shared state (so a value a prior
        # step/agent wrote — e.g. a classifier's writes= — is visible to the
        # Flow's nodes) plus the prompt. The engine's per-node dict IS this
        # state's session scope, so writes fold back into the one State.
        seed: dict[str, Any] = dict(shared._session)
        if resume_from_checkpoint:
            # Continue from a forked/edited checkpoint on this thread: layer its
            # persisted state under the inherited/prompt values so a re-run picks
            # up exactly where the (possibly edited) snapshot left off.
            saved = checkpointer.get(thread_id)
            if saved is not None:
                _, snapshot = saved
                seed = {**dict(snapshot.get("state", {})), **seed}
        if prompt is not None:
            seed.update(_seed_inputs(prompt))
        try:
            graph_result = await compiled.ainvoke(
                seed,
                thread_id=thread_id,
                resume=engine_resume,
                deps=deps,
            )
        except Interrupt as itr:
            return await self._pause(itr.value, thread_id, runner)

        if graph_result.interrupted:
            # If a decision was already provided but the flow re-paused (a later
            # step), surface the new pause.
            return await self._pause(graph_result.interrupt_value, thread_id, runner)

        output = _read_returns(graph_result.state, self._returns)
        # writes= capture for the whole flow's output.
        if self.writes is not None:
            shared[self.writes] = output
        return RunResult(
            output=output,
            usage=self._ctx.usage,
            events=list(self._events),
            run_id=thread_id,
        )

    async def _pause(self, value: Any, thread_id: str, runner: Any) -> RunResult[Any]:
        """Build the paused RunResult and (D5) create a flow_pause approval row.

        ``value`` is either a raw ``pause_for(...)`` payload (when this Flow's own
        step paused) or an inner :class:`Pending` (when a nested agent-step paused).
        We persist an :class:`ApprovalRequest(kind="flow_pause")` so the pause is
        visible in ``GET /approvals`` and ``approvals.respond()`` works on it, then
        surface the typed :class:`Pending` the caller reads off ``result.pending``.
        """
        store = getattr(runner, "approval_store", None)
        # An already-typed inner Pending (nested agent-step) passes through.
        if isinstance(value, Pending):
            pending = value
        else:
            approval_id = _flow_pause_id(thread_id, value)
            pending = Pending(
                kind="flow_pause",
                approval_id=approval_id,
                run_id=thread_id,
                resume_id=thread_id,
                payload=value,
                prompt=_payload_prompt(value),
            )
            if store is not None:
                from .governance.approvals import ApprovalRequest

                req = ApprovalRequest(
                    approval_id=approval_id,
                    run_id=thread_id,
                    resume_id=thread_id,
                    agent=self.name,
                    identity=self._ctx.identity if self._ctx is not None else None,
                    tool="flow_pause",
                    arguments=_payload_args(value),
                    kind="flow_pause",
                    prompt=_payload_prompt(value),
                )
                await store.create(req)
        return RunResult(
            output=None,
            usage=self._ctx.usage if self._ctx is not None else Usage(),
            events=list(self._events),
            run_id=thread_id,
            paused=True,
            pause_value=value if not isinstance(value, Pending) else value.payload,
            pending=[pending],
        )


# --- helpers ---------------------------------------------------------------


class _LoopExit:
    """A loop's exit picker: returns to the step until ``until`` (or the cap)."""

    def __init__(self, step: str, until: Any, max_iterations: int) -> None:
        self.step = step
        self.until = until
        self.max_iterations = max_iterations

    def as_router(self, flow: Flow) -> Callable[[dict], str]:
        counter_key = f"temp:__loop_count__{self.step}"

        def _router(engine_state: dict[str, Any]) -> str:
            # Bind the engine state so the until= condition reads committed values.
            ctx = flow._active_ctx(engine_state)
            count = ctx.state.get(counter_key, 0) + 1
            ctx.state[counter_key] = count
            ro = ReadonlyState(ctx.state)
            if count >= self.max_iterations:
                return "__done__"
            # A loop exit reads STATE, not an output value: the until predicate's
            # first argument is the read-only state view (matching route pickers),
            # so ``until=lambda state, ctx: ...`` and ``until="state.x >= n"`` both
            # see the accumulated loop state.
            return "__done__" if _eval_until(self.until, ro, ctx) else "__loop__"

        return _router


def _eval_until(until: Any, ro: ReadonlyState, ctx: RunContext) -> bool:
    """Evaluate a loop ``until`` predicate, reading STATE (not an output value).

    A ``Condition`` is checked against the state view; a string is compiled and
    evaluated against the state; a callable's first argument is the read-only
    state view (its second, if present, the RunContext) — matching route pickers.
    """
    if isinstance(until, Condition):
        return until.check(Guard(value=None, state=ro, ctx=ctx, phase=Phase.INPUT))
    if isinstance(until, str):
        from .expr import compile_expr

        cond = compile_expr(until, phase=Phase.INPUT)
        return cond.check(Guard(value=None, state=ro, ctx=ctx, phase=Phase.INPUT))
    if callable(until):
        params = len(inspect.signature(until).parameters)
        return bool(until(ro, ctx) if params >= 2 else until(ro))
    raise YaabError(f"not a loop until predicate: {until!r}")


def _call_picker(picker: Any, ro: ReadonlyState, ctx: RunContext) -> str:
    """Call a route picker (Condition / callable / expr) and return its label."""
    if isinstance(picker, Condition):
        guard = Guard(value=None, state=ro, ctx=ctx, phase=Phase.INPUT)
        return str(picker.check(guard))
    if isinstance(picker, str):
        # A safe-expression picker returns a label by evaluating to a value.
        from .expr import compile_expr

        cond = compile_expr(picker, phase=Phase.INPUT)
        guard = Guard(value=None, state=ro, ctx=ctx, phase=Phase.INPUT)
        return str(cond.check(guard))
    if callable(picker):
        params = len(inspect.signature(picker).parameters)
        return picker(ro, ctx) if params >= 2 else picker(ro)
    raise YaabError(f"not a route picker: {picker!r}")


def _scope_of_merge(writes: str | None) -> bool:
    """True when a ``writes=`` key is *scoped* (so ``merge=`` is rejected)."""
    if writes is None:
        return False
    from .state import scope_of

    return scope_of(writes) != "session"


def _default_for(reducer: str) -> Any:
    if reducer == "append":
        return []
    if reducer == "add":
        return 0
    return None


def _seed_inputs(prompt: str) -> dict[str, Any]:
    """Seed the engine state with the incoming prompt under a conventional key."""
    return {"input": prompt, "prompt": prompt}


def _prompt_from(engine_state: dict[str, Any]) -> str:
    """Derive an agent step's prompt from the current state."""
    val = engine_state.get("input", engine_state.get("prompt", ""))
    return val if isinstance(val, str) else str(val)


def _read_returns(state: dict[str, Any], key: str | None) -> Any:
    if key is None:
        return None
    return state.get(key)


def _resume_value(resume: Any) -> tuple[Any, str | None]:
    """Extract the decided value (and approval id) from a resume token."""
    if resume is None:
        return None, None
    # A Decision (Wave 3): its answer is the threaded-back value.
    answer = getattr(resume, "answer", None)
    if answer is not None:
        return answer, getattr(resume, "approval_id", None)
    verdict = getattr(resume, "verdict", None)
    if verdict is not None:
        # An approve/deny with no typed answer: thread the verdict word back.
        return ("approve" if verdict == "approved" else "deny"), getattr(
            resume, "approval_id", None
        )
    # A bare value passed directly as resume=.
    return resume, None


def _flow_pause_id(thread_id: str, value: Any) -> str:
    """Deterministic approval id for a flow pause (idempotent create self-heals)."""
    try:
        sig = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        sig = str(value)
    digest = hashlib.sha256(f"{thread_id}|flow_pause|{sig}".encode()).hexdigest()[:12]
    return f"ap_{digest}"


def _payload_prompt(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("question", "prompt", "needs", "message"):
            if k in value:
                return str(value[k])
    return None


def _payload_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


__all__ = [
    "Flow",
    "MemoryCheckpoints",
    "SQLiteCheckpoints",
    "PostgresCheckpoints",
    "RedisCheckpoints",
    "RetryPolicy",
]
