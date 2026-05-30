"""Durable, checkpointed graph orchestration (LangGraph-style).

A :class:`StateGraph` has nodes (functions), edges (incl. conditional edges and
cycles), and typed state channels with reducers. It compiles to a runtime that
executes in BSP supersteps (planned by the Rust core), checkpoints state at
every step, supports human-in-the-loop via :func:`interrupt`, and can resume by
``thread_id`` after a crash or an interrupt.

This is the deterministic, inspectable counterpart to the model-driven fast
path — the one you reach for when an auditor or SLA needs explicit control flow.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .. import _core
from ..exceptions import Interrupt, YaabError
from .checkpoint import Checkpointer, MemorySaver

START = "__start__"
END = "__end__"

_MISSING = object()

NodeFn = Callable[..., Any]
Router = Callable[[dict[str, Any]], str]


class Channel:
    """Declares how writes to a state key are reduced.

    ``reducer`` is one of ``"last_value"`` (overwrite), ``"append"``
    (accumulate into a list), or ``"add"`` (numeric sum) — matching the Rust
    reducer in :mod:`yaab._core`.
    """

    def __init__(self, reducer: str = "last_value", default: Any = None) -> None:
        self.reducer = reducer
        self.default = default


class GraphContext:
    """Per-invocation context handed to each node as its second argument."""

    def __init__(self, thread_id: str, deps: Any, resume: Any = _MISSING) -> None:
        self.thread_id = thread_id
        self.deps = deps
        self._resume = resume
        self._resume_used = False

    def interrupt(self, value: Any) -> Any:
        """Pause for human input, or return the resumed value on continuation.

        On the first pass this raises :class:`Interrupt`; the runtime
        checkpoints and surfaces ``value`` to the caller. When the caller
        resumes the thread, the same call returns the supplied resume value.
        """
        if self._resume is not _MISSING and not self._resume_used:
            self._resume_used = True
            return self._resume
        raise Interrupt(value)


def interrupt(ctx: GraphContext, value: Any) -> Any:
    """Module-level alias for :meth:`GraphContext.interrupt`."""
    return ctx.interrupt(value)


class GraphResult(BaseModel):
    """The outcome of a graph invocation."""

    state: dict[str, Any]
    interrupted: bool = False
    interrupt_value: Any = None
    steps: int = 0


class StateGraph:
    """Builder for a stateful, durable graph."""

    def __init__(
        self,
        state_schema: Optional[type] = None,
        *,
        channels: Optional[dict[str, Channel]] = None,
    ) -> None:
        self.state_schema = state_schema
        self.channels: dict[str, Channel] = channels or {}
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, list[str]] = {}
        self.conditional: dict[str, tuple[Router, dict[str, str]]] = {}
        self.entry: Optional[str] = None

    # --- construction --------------------------------------------------
    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in (START, END):
            raise YaabError(f"'{name}' is a reserved node name")
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges.setdefault(src, []).append(dst)
        if src == START:
            self.entry = dst
        return self

    def add_conditional_edges(
        self, src: str, router: Router, mapping: dict[str, str]
    ) -> "StateGraph":
        self.conditional[src] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def set_finish_point(self, name: str) -> "StateGraph":
        self.edges.setdefault(name, []).append(END)
        return self

    def add_channel(self, key: str, channel: Channel) -> "StateGraph":
        self.channels[key] = channel
        return self

    def compile(self, checkpointer: Optional[Checkpointer] = None) -> "CompiledGraph":
        if self.entry is None:
            raise YaabError("graph has no entry point; call set_entry_point or add_edge(START, ...)")
        # Plan supersteps via the Rust core (informational / parallel grouping).
        edge_pairs = [
            (s, d)
            for s, dsts in self.edges.items()
            for d in dsts
            if s not in (START, END) and d not in (START, END)
        ]
        supersteps = _core.plan_supersteps(list(self.nodes.keys()), edge_pairs)
        return CompiledGraph(self, checkpointer or MemorySaver(), supersteps)


class CompiledGraph:
    """An executable graph with checkpointing and HITL resume."""

    def __init__(
        self, graph: StateGraph, checkpointer: Checkpointer, supersteps: list[list[str]]
    ) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self.supersteps = supersteps

    def _init_state(self, inputs: dict[str, Any]) -> dict[str, Any]:
        state: dict[str, Any] = {k: ch.default for k, ch in self.graph.channels.items()}
        state.update(inputs or {})
        return state

    def _apply(self, state: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            channel = self.graph.channels.get(key)
            if channel is None:
                state[key] = value  # untyped key: last-value
            else:
                current = state.get(key, channel.default)
                state[key] = _core.reduce_channel(channel.reducer, current, value)

    def _successors(self, node: str, state: dict[str, Any]) -> list[str]:
        if node in self.graph.conditional:
            router, mapping = self.graph.conditional[node]
            key = router(state)
            target = mapping.get(key, key)
            return [target]
        return list(self.graph.edges.get(node, []))

    async def ainvoke(
        self,
        inputs: Optional[dict[str, Any]] = None,
        *,
        thread_id: str = "default",
        resume: Any = _MISSING,
        deps: Any = None,
        max_supersteps: int = 100,
    ) -> GraphResult:
        # Resume from the latest checkpoint if one exists, else initialize.
        saved = self.checkpointer.get(thread_id) if resume is not _MISSING else None
        if saved is not None:
            step, snapshot = saved
            state = snapshot["state"]
            frontier = snapshot.get("frontier", [self.graph.entry])
        else:
            step = 0
            state = self._init_state(inputs or {})
            frontier = [self.graph.entry]

        resume_for_first = resume

        for _ in range(max_supersteps):
            active = [n for n in frontier if n not in (START, END)]
            if not active:
                break

            next_frontier: list[str] = []
            for node in active:
                fn = self.graph.nodes[node]
                ctx = GraphContext(thread_id, deps, resume=resume_for_first)
                resume_for_first = _MISSING  # only the first resumed node consumes it
                try:
                    updates = await _maybe_await(fn, state, ctx)
                except Interrupt as itr:
                    # Park this node (and the rest of the frontier) and surface.
                    self.checkpointer.put(
                        thread_id,
                        step,
                        {"state": state, "frontier": [node, *[n for n in active if n != node]]},
                    )
                    return GraphResult(
                        state=state, interrupted=True, interrupt_value=itr.value, steps=step
                    )
                if updates:
                    self._apply(state, updates)
                for succ in self._successors(node, state):
                    if succ == END:
                        continue
                    if succ not in next_frontier:
                        next_frontier.append(succ)

            step += 1
            self.checkpointer.put(thread_id, step, {"state": state, "frontier": next_frontier})
            frontier = next_frontier
            if not frontier:
                break

        return GraphResult(state=state, interrupted=False, steps=step)

    def invoke(self, inputs: Optional[dict[str, Any]] = None, **kwargs: Any) -> GraphResult:
        import asyncio

        return asyncio.run(self.ainvoke(inputs, **kwargs))


async def _maybe_await(fn: NodeFn, state: dict[str, Any], ctx: GraphContext) -> Any:
    # Nodes may be sync or async, and may accept (state) or (state, ctx).
    params = len(inspect.signature(fn).parameters)
    result = fn(state, ctx) if params >= 2 else fn(state)
    if inspect.isawaitable(result):
        result = await result
    return result


__all__ = [
    "StateGraph",
    "CompiledGraph",
    "GraphResult",
    "GraphContext",
    "Channel",
    "interrupt",
    "START",
    "END",
]
