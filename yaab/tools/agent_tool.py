"""Agent-as-Tool: expose one agent as a callable tool of another.

This is the building block for hierarchical delegation and the
"agent as tool" multi-agent pattern. The sub-agent runs with its own loop and
returns its final output as the tool result.
"""

from __future__ import annotations

from typing import Any

from ..types import RunContext


class AgentTool:
    """Adapt an :class:`~yaab.agent.Agent` into a :class:`~yaab.tools.base.Tool`."""

    def __init__(self, agent: Any, *, name: str | None = None, description: str | None = None):
        self.agent = agent
        self.name = name or f"call_{agent.name}"
        self.description = description or (
            agent.instructions
            if isinstance(agent.instructions, str)
            else f"Delegate to the {agent.name} agent."
        )

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The task or question to delegate to the sub-agent.",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        }

    async def execute(self, ctx: RunContext, *, prompt: str) -> Any:
        # Share the caller's one state object so the delegated agent reads/writes
        # the same shared state as its parent (S0: one State per run).
        result = await self.agent.run(
            prompt,
            deps=ctx.deps,
            session_id=ctx.session_id,
            identity=ctx.identity,
            state=getattr(ctx, "state", None),
        )
        # Roll the sub-agent's usage up into the parent run.
        ctx.usage.add(result.usage)
        return result.output
