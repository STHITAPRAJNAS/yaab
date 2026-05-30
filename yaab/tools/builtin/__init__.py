"""Built-in starter tools — a robust agent toolbox out of the box.

These are ready-to-use :func:`~yaab.tool`-decorated functions covering the
capabilities every agent reaches for first: time, math, HTTP fetch, web search,
and sandboxed Python execution. They are dependency-light (web/HTTP tools import
``httpx`` lazily; code exec uses the stdlib) and governed by the same
authorization / approval / guardrail layer as any other tool.

    from yaab import Agent
    from yaab.tools.builtin import calculator, current_time, http_get

    agent = Agent("a", model="openai/gpt-4o", tools=[calculator, current_time, http_get])

Or grab the safe default set:

    from yaab.tools.builtin import default_toolset
    agent = Agent("a", model="openai/gpt-4o", tools=default_toolset())
"""

from __future__ import annotations

from .calculator import calculator
from .code import python_exec
from .datetime_tool import current_time
from .http import http_get
from .search import web_search


def default_toolset() -> list:
    """A safe, read-only default set (no code exec, no network writes)."""
    return [calculator, current_time, http_get, web_search]


__all__ = [
    "calculator",
    "current_time",
    "http_get",
    "web_search",
    "python_exec",
    "default_toolset",
]
