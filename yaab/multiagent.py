"""Multi-agent orchestration patterns over the one runtime.

These *workflow agents* compose other agents and expose the same ``run`` /
``run_sync`` / ``as_tool`` surface as a plain :class:`~yaab.agent.Agent`, so they
nest arbitrarily and drop into tools, graphs, and servers:

* :class:`SequentialAgent` — run sub-agents in order, piping output to input;
* :class:`ParallelAgent`   — run sub-agents concurrently on the same input;
* :class:`LoopAgent`       — re-run a sub-agent until a condition or a cap;
* :class:`Swarm`           — autonomous hand-off between peer agents.

Usage is rolled up across all sub-agents so cost/token accounting stays whole.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from .types import RunResult, Usage


class _WorkflowBase:
    """Shared run/run_sync/as_tool surface for workflow agents."""

    name: str

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        session_id: Optional[str] = None,
        identity: Optional[str] = None,
    ) -> RunResult[Any]:
        raise NotImplementedError

    def run_sync(self, prompt: str, **kwargs: Any) -> RunResult[Any]:
        return asyncio.run(self.run(prompt, **kwargs))

    def as_tool(self, *, name: Optional[str] = None, description: Optional[str] = None) -> Any:
        from .tools.agent_tool import AgentTool

        return AgentTool(self, name=name, description=description)


class SequentialAgent(_WorkflowBase):
    """Run sub-agents in sequence, feeding each output into the next prompt."""

    def __init__(self, name: str, agents: list[Any], *, pipe_output: bool = True) -> None:
        self.name = name
        self.agents = agents
        self.pipe_output = pipe_output
        self.instructions = f"Sequential pipeline of {len(agents)} agents."

    async def run(self, prompt: str, *, deps: Any = None, session_id=None, identity=None):
        usage = Usage()
        current_input = prompt
        last: Optional[RunResult] = None
        for agent in self.agents:
            last = await agent.run(
                current_input, deps=deps, session_id=session_id, identity=identity
            )
            usage.add(last.usage)
            if self.pipe_output:
                current_input = _as_text(last.output)
        return RunResult(output=last.output if last else None, usage=usage, run_id=self.name)


class ParallelAgent(_WorkflowBase):
    """Run sub-agents concurrently on the same prompt; output is a name→result map."""

    def __init__(self, name: str, agents: list[Any]) -> None:
        self.name = name
        self.agents = agents
        self.instructions = f"Parallel fan-out across {len(agents)} agents."

    async def run(self, prompt: str, *, deps: Any = None, session_id=None, identity=None):
        results = await asyncio.gather(
            *(a.run(prompt, deps=deps, identity=identity) for a in self.agents)
        )
        usage = Usage()
        output: dict[str, Any] = {}
        for agent, result in zip(self.agents, results, strict=False):
            usage.add(result.usage)
            output[agent.name] = result.output
        return RunResult(output=output, usage=usage, run_id=self.name)


class LoopAgent(_WorkflowBase):
    """Re-run a sub-agent, feeding its output back, until a stop condition.

    ``until`` receives the latest output and returns ``True`` to stop; the loop
    also stops at ``max_iterations``.
    """

    def __init__(
        self,
        name: str,
        agent: Any,
        *,
        max_iterations: int = 5,
        until: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        self.name = name
        self.agent = agent
        self.max_iterations = max_iterations
        self.until = until
        self.instructions = f"Loop over {agent.name} up to {max_iterations}x."

    async def run(self, prompt: str, *, deps: Any = None, session_id=None, identity=None):
        usage = Usage()
        current_input = prompt
        last: Optional[RunResult] = None
        for _ in range(self.max_iterations):
            last = await self.agent.run(
                current_input, deps=deps, session_id=session_id, identity=identity
            )
            usage.add(last.usage)
            if self.until and self.until(last.output):
                break
            current_input = _as_text(last.output)
        return RunResult(output=last.output if last else None, usage=usage, run_id=self.name)


class SwarmState(BaseModel):
    """Shared, mutable state threaded through a swarm via DI."""

    handoff: Optional[str] = None
    shared: dict[str, Any] = Field(default_factory=dict)


class Swarm(_WorkflowBase):
    """Autonomous hand-off between peer agents (Strands-style swarm).

    Each member is augmented with ``handoff_to_<peer>`` tools. When an agent
    decides another is better suited, it calls the handoff tool; the swarm then
    continues the task with that agent. Runs until no further hand-off (or a cap).
    """

    def __init__(
        self,
        name: str,
        agents: list[Any],
        *,
        entry: Optional[str] = None,
        max_handoffs: int = 6,
    ) -> None:
        self.name = name
        self.agents = {a.name: a for a in agents}
        self.entry = entry or agents[0].name
        self.max_handoffs = max_handoffs
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

        async def handoff(ctx: RunContext) -> str:
            if isinstance(ctx.deps, SwarmState):
                ctx.deps.handoff = target
            return f"handing off to {target}"

        tool = FunctionTool(
            handoff,
            name=f"handoff_to_{target}",
            description=f"Delegate the task to the '{target}' agent when it is better suited.",
        )
        return tool

    async def run(self, prompt: str, *, deps: Any = None, session_id=None, identity=None):
        state = deps if isinstance(deps, SwarmState) else SwarmState()
        usage = Usage()
        current = self.entry
        current_input = prompt
        last: Optional[RunResult] = None
        for _ in range(self.max_handoffs + 1):
            state.handoff = None
            agent = self.agents[current]
            last = await agent.run(
                current_input, deps=state, session_id=session_id, identity=identity
            )
            usage.add(last.usage)
            if state.handoff and state.handoff in self.agents and state.handoff != current:
                current = state.handoff
                current_input = _as_text(last.output) or prompt
                continue
            break
        return RunResult(output=last.output if last else None, usage=usage, run_id=self.name)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return str(value)


__all__ = ["SequentialAgent", "ParallelAgent", "LoopAgent", "Swarm", "SwarmState"]
