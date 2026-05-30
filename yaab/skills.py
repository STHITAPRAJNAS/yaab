"""Skills — reusable, shareable bundles of tools + instructions + prompts.

A :class:`Skill` packages a capability (its tools, an instruction fragment, an
optional prompt, and declared permissions) so it can be attached to any agent
and discovered/loaded via entry points. This is YAAB's analog of ADK's tool
bundles and Pydantic AI's "capabilities", made first-class and governable: a
skill's ``permissions`` feed the registry's action-scope, and its tools appear
in the agent card.

    from yaab import Agent
    from yaab.skills import Skill

    research = Skill(
        name="research",
        instructions="Use search before answering factual questions.",
        tools=[web_search],
        permissions=["net:read"],
    )
    agent = Agent("analyst", skills=[research])
"""

from __future__ import annotations

from typing import Any

from .tools.base import Tool, coerce_tools


class Skill:
    """A named bundle of tools, instructions, and an optional prompt."""

    def __init__(
        self,
        name: str,
        *,
        instructions: str = "",
        tools: list[Any] | None = None,
        prompt: str | None = None,
        permissions: list[str] | None = None,
        version: str = "0.1.0",
        description: str = "",
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: list[Tool] = coerce_tools(tools or [])
        self.prompt = prompt
        self.permissions = permissions or []
        self.version = version
        self.description = description or instructions

    def card_skill(self) -> dict[str, Any]:
        """Render an A2A agent-card ``skills[]`` entry."""
        return {
            "id": self.name,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.permissions,
        }

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, tools={len(self.tools)}, perms={self.permissions})"


def load_skills() -> dict[str, Skill]:
    """Discover third-party skills registered via the ``yaab.skills`` entry point."""
    out: dict[str, Skill] = {}
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="yaab.skills"):
            try:
                obj = ep.load()
                skill = obj() if callable(obj) and not isinstance(obj, Skill) else obj
                if isinstance(skill, Skill):
                    out[ep.name] = skill
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return out


__all__ = ["Skill", "load_skills"]
