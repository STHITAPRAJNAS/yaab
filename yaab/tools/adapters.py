"""Adapt tools from other ecosystems into native YAAB tools.

A tool you already have — from another agent library — can be reused as-is: the
adapters here wrap a foreign tool object in a :class:`~yaab.tools.base.FunctionTool`
so it carries its name/description/arguments and runs through YAAB's normal tool
loop (timeouts, parallelism, governance, tracing). The foreign interface is
*duck-typed*, so neither library needs to be installed for the adapter to work —
only the tool object you pass in.

    from yaab.tools.adapters import adapt_tool
    agent = Agent("a", model=..., tools=[adapt_tool(my_existing_tool)])
"""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import ConfigDict, create_model

from .base import FunctionTool


def _coerce(result: Any) -> Any:
    """Normalize a foreign tool's return into something the model can read."""
    return result


def _carry_args(tool: FunctionTool, foreign: Any) -> FunctionTool:
    """Let a duck-typed wrapper accept the foreign tool's arguments.

    A ``**kwargs`` wrapper yields an empty argument model, which would drop the
    model's tool arguments before they reach the foreign tool. Replace it with a
    model that *allows extra fields* so every argument passes through, and surface
    the foreign tool's own parameter schema (LangChain ``.args`` / a pydantic
    ``args_schema``) to the model when it is discoverable.
    """
    permissive = create_model(f"{tool.name}_Args", __config__=ConfigDict(extra="allow"))
    # Best-effort schema fidelity: expose the foreign tool's parameter shape so a
    # real model knows what to pass. Falls back to an open object when absent.
    args_schema = getattr(foreign, "args", None)
    if isinstance(args_schema, dict) and args_schema:
        props = {k: (v if isinstance(v, dict) else {}) for k, v in args_schema.items()}
        permissive.model_json_schema = lambda *a, **k: {  # type: ignore[method-assign]
            "type": "object",
            "properties": props,
        }
    tool._arg_model = permissive
    return tool


def from_langchain_tool(tool: Any) -> FunctionTool:
    """Wrap a LangChain ``BaseTool`` (or anything with ``.name``/``.invoke``).

    The wrapped callable accepts arbitrary keyword arguments and forwards them to
    the foreign tool's ``invoke`` (preferred) / ``run`` / ``_run`` method, passing
    a single positional value when the tool takes one input and a dict otherwise.
    """
    name = getattr(tool, "name", None)
    description = getattr(tool, "description", "") or ""
    runner = (
        getattr(tool, "invoke", None) or getattr(tool, "run", None) or getattr(tool, "_run", None)
    )
    if name is None or runner is None:
        raise TypeError(
            "not a LangChain-style tool: expected a `.name` and an `.invoke`/`.run`/`._run` method"
        )

    async def _call(**kwargs: Any) -> Any:
        # A single-input tool wants the bare value; a structured one wants the map.
        payload: Any = next(iter(kwargs.values())) if len(kwargs) == 1 else kwargs
        result = runner(payload)
        if inspect.isawaitable(result):
            result = await result
        return _coerce(result)

    return _carry_args(FunctionTool(_call, name=name, description=description.strip()), tool)


def from_crewai_tool(tool: Any) -> FunctionTool:
    """Wrap a CrewAI ``BaseTool`` (or anything with ``.name``/``.run``).

    CrewAI tools take keyword arguments, so the wrapper forwards ``**kwargs``
    straight through to the foreign tool's ``run`` / ``_run`` method.
    """
    name = getattr(tool, "name", None)
    description = getattr(tool, "description", "") or ""
    runner = getattr(tool, "run", None) or getattr(tool, "_run", None)
    if name is None or runner is None:
        raise TypeError("not a CrewAI-style tool: expected a `.name` and a `.run`/`._run` method")

    async def _call(**kwargs: Any) -> Any:
        result = runner(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return _coerce(result)

    return _carry_args(FunctionTool(_call, name=name, description=description.strip()), tool)


def adapt_tool(tool: Any) -> FunctionTool:
    """Adapt a foreign tool, auto-detecting its ecosystem by its interface.

    A tool exposing ``.invoke`` is treated as LangChain-style; one exposing only
    ``.run``/``._run`` as CrewAI-style. Anything that matches neither raises
    ``TypeError`` so a misuse fails loudly rather than silently.
    """
    if hasattr(tool, "invoke") and hasattr(tool, "name"):
        return from_langchain_tool(tool)
    if (hasattr(tool, "run") or hasattr(tool, "_run")) and hasattr(tool, "name"):
        return from_crewai_tool(tool)
    raise TypeError(
        f"cannot adapt {type(tool).__name__}: expected a tool with a `.name` and an "
        "`.invoke` (LangChain-style) or `.run`/`._run` (CrewAI-style) method"
    )
