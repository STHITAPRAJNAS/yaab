"""Multi-agent orchestration patterns over the one runtime.

These *workflow agents* compose other agents and expose the same ``run`` /
``run_sync`` / ``as_tool`` surface as a plain :class:`~yaab.agent.Agent`, so they
nest arbitrarily and drop into tools, graphs, and servers:

* :class:`SequentialAgent` — run sub-agents in order, sharing one state;
* :class:`ParallelAgent`   — run sub-agents concurrently on the same input;
* :class:`MapAgent`        — fan one sub-agent across many inputs;
* :class:`LoopAgent`       — re-run a sub-agent until a condition or a cap;
* :class:`Swarm`           — autonomous hand-off between peer agents.

Every pattern shares **one** :class:`~yaab.state.State` object across all its
children for a run, so a value written by one step is read by the next by key.
A step can declare ``writes="key"`` to capture its (typed) output into that
shared state; the next step reads it via ``{key}`` instruction injection or a
tool. Usage is rolled up across all sub-agents so cost/token accounting stays
whole.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from .state import State, StateConflictError, scope_of
from .types import RunResult, Usage


def _capture(state: State, child: Any, result: RunResult) -> None:
    """Land a child's output into shared state under its ``writes=`` key.

    The *typed* ``result.output`` is stored exactly as produced (a model stays a
    model, a list stays a list) — it never round-trips through text. The key's
    prefix chooses the scope (``temp:``/``user:``/``app:``/session) for free.
    """
    key = getattr(child, "writes", None)
    if key is not None and result is not None:
        state[key] = result.output


def _state_for_run(state: State | None) -> State:
    """Inherit the parent's State, or build a run-local one for a top-level run.

    A workflow agent invoked as a child is handed the parent's State; only the
    outermost entity builds one. A workflow run with a ``session_id`` still lets
    each child's Runner reuse the session — the shared State is the in-run
    communication medium, and child runs persist through their own session seam.
    """
    return state if state is not None else State()


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
    ``stop_when`` receives each step's output and returns ``True`` to stop early.
    """

    def __init__(
        self,
        name: str,
        agents: list[Any],
        *,
        pipe_output: bool = True,
        stop_when: Callable[[Any], bool] | None = None,
        writes: str | None = None,
    ) -> None:
        self.name = name
        self.agents = agents
        self.pipe_output = pipe_output
        self.stop_when = stop_when
        self.writes = writes
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
        current_input = prompt
        last: RunResult | None = None
        for agent in self.agents:
            last = await agent.run(
                current_input,
                deps=deps,
                session_id=session_id,
                identity=identity,
                state=shared,
            )
            usage.add(last.usage)
            _capture(shared, agent, last)
            if self.stop_when and self.stop_when(last.output):
                break
            if self.pipe_output:
                current_input = _as_text(last.output)
        return RunResult(output=last.output if last else None, usage=usage, run_id=self.name)


class ParallelAgent(_WorkflowBase):
    """Run sub-agents concurrently on the same prompt over one shared state.

    All branches see the *same* state object. Each branch that declares
    ``writes="key"`` lands its result under a key downstream steps read by name —
    two branches writing the *same* session-scoped key without a reducer is a
    declared :class:`~yaab.state.StateConflictError`, not a silent clobber. The
    returned output is a ``name -> result`` map for convenience.
    """

    def __init__(self, name: str, agents: list[Any], *, writes: str | None = None) -> None:
        self.name = name
        self.agents = agents
        self.writes = writes
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
        # Detect colliding session-scoped writes= keys *before* running, so a
        # clobber surfaces as a clear error rather than last-writer-wins.
        self._check_write_conflicts()
        # All branches share one state object (the in-run communication medium).
        results = await asyncio.gather(
            *(
                a.run(
                    prompt,
                    deps=deps,
                    session_id=session_id,
                    identity=identity,
                    state=shared,
                )
                for a in self.agents
            )
        )
        usage = Usage()
        output: dict[str, Any] = {}
        for agent, result in zip(self.agents, results, strict=False):
            usage.add(result.usage)
            _capture(shared, agent, result)
            # Read-by-key: the fan-in map is keyed by each branch's name, and the
            # branch's writes= value is already in shared state for downstream
            # steps to read by name (never reconstructed by positional zip).
            output[agent.name] = result.output
        return RunResult(output=output, usage=usage, run_id=self.name)

    def _check_write_conflicts(self) -> None:
        seen: set[str] = set()
        for agent in self.agents:
            key = getattr(agent, "writes", None)
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
    """

    def __init__(
        self,
        name: str,
        agent: Any,
        *,
        map_inputs: Callable[[str], list[str]] | None = None,
        max_concurrency: int = 0,
        writes: str | None = None,
    ) -> None:
        self.name = name
        self.agent = agent
        self.map_inputs = map_inputs
        self.max_concurrency = max_concurrency
        self.writes = writes
        self.instructions = f"Map {agent.name} across N inputs."

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
        # Accept an explicit list of prompts, or derive them via map_inputs.
        if isinstance(prompt, list):
            inputs = prompt
        elif self.map_inputs is not None:
            inputs = self.map_inputs(prompt)
        else:
            inputs = [prompt]

        sem = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency > 0 else None

        async def _one(p: str) -> RunResult:
            # Thread session_id AND the shared state to every map child — both
            # were previously dropped, so a map child could not see the run's
            # session or its shared state.
            if sem is not None:
                async with sem:
                    return await self.agent.run(
                        p,
                        deps=deps,
                        session_id=session_id,
                        identity=identity,
                        state=shared,
                    )
            return await self.agent.run(
                p,
                deps=deps,
                session_id=session_id,
                identity=identity,
                state=shared,
            )

        results = await asyncio.gather(*(_one(p) for p in inputs))
        usage = Usage()
        for r in results:
            usage.add(r.usage)
        outputs = [r.output for r in results]
        if self.writes is not None:
            shared[self.writes] = outputs
        return RunResult(output=outputs, usage=usage, run_id=self.name)


class LoopAgent(_WorkflowBase):
    """Re-run a sub-agent over one accumulating shared state until a stop condition.

    Each iteration runs against the same :class:`~yaab.state.State`, so a tool
    that increments ``state["count"]`` accumulates across iterations. ``until``
    receives the latest output and returns ``True`` to stop; the loop also stops
    at ``max_iterations``. ``pipe_output`` (default ``True``) keeps feeding the
    prior output as the next prompt for the common refine-in-place case.
    """

    def __init__(
        self,
        name: str,
        agent: Any,
        *,
        max_iterations: int = 5,
        until: Callable[[Any], bool] | None = None,
        pipe_output: bool = True,
        writes: str | None = None,
    ) -> None:
        self.name = name
        self.agent = agent
        self.max_iterations = max_iterations
        self.until = until
        self.pipe_output = pipe_output
        self.writes = writes
        self.instructions = f"Loop over {agent.name} up to {max_iterations}x."

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
        current_input = prompt
        last: RunResult | None = None
        for _ in range(self.max_iterations):
            last = await self.agent.run(
                current_input,
                deps=deps,
                session_id=session_id,
                identity=identity,
                state=shared,
            )
            usage.add(last.usage)
            _capture(shared, self.agent, last)
            if self.until and self.until(last.output):
                break
            if self.pipe_output:
                current_input = _as_text(last.output)
        if self.writes is not None and last is not None:
            shared[self.writes] = last.output
        return RunResult(output=last.output if last else None, usage=usage, run_id=self.name)


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
    object. Runs until no further hand-off (or a cap).
    """

    _HANDOFF_KEY = "temp:__handoff__"

    def __init__(
        self,
        name: str,
        agents: list[Any],
        *,
        entry: str | None = None,
        max_handoffs: int = 6,
        writes: str | None = None,
    ) -> None:
        self.name = name
        self.agents = {a.name: a for a in agents}
        self.entry = entry or agents[0].name
        self.max_handoffs = max_handoffs
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
        from .types import RunContext

        key = self._HANDOFF_KEY

        async def handoff(ctx: RunContext) -> str:
            # Record the handoff on the shared state (run-local). Keep the legacy
            # SwarmState.handoff in sync when one was passed as deps so existing
            # code observing deps keeps working.
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
            target = shared.get(self._HANDOFF_KEY)
            if target and target in self.agents and target != current:
                current = target
                current_input = _as_text(last.output) or prompt
                continue
            break
        if self.writes is not None and last is not None:
            shared[self.writes] = last.output
        return RunResult(output=last.output if last else None, usage=usage, run_id=self.name)


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
]
