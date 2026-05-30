"""Plugin system — cross-cutting extension, ADK-style.

Plugins register on the :class:`~yaab.runner.Runner` and fire on lifecycle
callbacks that apply globally across every agent the runner drives. A hook can:

* **observe** — return ``None`` (the default no-op);
* **intervene** — return a value to short-circuit (e.g. a cached model
  response, or a blocked tool result);
* **amend** — mutate the :class:`RunContext` in place.

Built-ins (audit, governance enforcement, cost budget) live in
:mod:`yaab.plugins.builtins`.
"""

from __future__ import annotations

from typing import Any, Optional

from ..models.base import ModelResponse
from ..types import Message, RunContext


class Plugin:
    """Base plugin. Override the hooks you care about; the rest are no-ops.

    Hooks are async so plugins can do I/O (audit sinks, remote policy checks).
    """

    name: str = "plugin"

    async def before_run(self, ctx: RunContext, agent: str, prompt: str) -> None:
        ...

    async def after_run(self, ctx: RunContext, agent: str, output: Any) -> None:
        ...

    async def on_user_message(self, ctx: RunContext, agent: str, message: Message) -> None:
        ...

    async def before_model(
        self, ctx: RunContext, agent: str, messages: list[Message]
    ) -> Optional[ModelResponse]:
        """Return a :class:`ModelResponse` to short-circuit the model call."""
        return None

    async def after_model(
        self, ctx: RunContext, agent: str, response: ModelResponse
    ) -> Optional[ModelResponse]:
        """Return a replacement response to amend the model output."""
        return None

    async def repair_tool_args(
        self, ctx: RunContext, agent: str, tool: str, args: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Pre-process/repair raw tool-call args before they are validated.

        Return a replacement args dict to use instead, or ``None`` to leave the
        args unchanged. Runs before :meth:`before_tool` and tool execution — the
        seam for coercing malformed model output (Pydantic AI #3008).
        """
        return None

    async def before_tool(
        self, ctx: RunContext, agent: str, tool: str, args: dict[str, Any]
    ) -> Any:
        """Return a non-``None`` value to short-circuit the tool execution."""
        return None

    async def after_tool(self, ctx: RunContext, agent: str, tool: str, result: Any) -> Any:
        """Return a non-``None`` value to replace the tool result."""
        return None


__all__ = ["Plugin"]
