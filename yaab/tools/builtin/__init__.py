"""Built-in starter tools — a robust agent toolbox out of the box.

These are ready-to-use :func:`~yaab.tool`-decorated functions covering the
capabilities every agent reaches for first: time, math, HTTP fetch, URL-to-text
(``fetch_url``), web search (with a keyless DuckDuckGo option), sandboxed Python
execution, and sandboxed file read/write/list. They are dependency-light
(web/HTTP tools import ``httpx`` lazily; code exec uses the stdlib) and governed
by the same authorization / approval / guardrail layer as any other tool.

    from yaab import Agent
    from yaab.tools.builtin import calculator, current_time, http_get

    agent = Agent("a", model="openai/gpt-4o", tools=[calculator, current_time, http_get])

Or grab the safe default set:

    from yaab.tools.builtin import default_toolset
    agent = Agent("a", model="openai/gpt-4o", tools=default_toolset())

Every built-in is also registered in the component registry under the ``"tool"``
kind, so they're discoverable via ``yaab.available_components("tool")`` and
buildable via ``yaab.get_component("tool", "calculator")``.
"""

from __future__ import annotations

from ...extensions import register
from .ask_user import ask_user
from .calculator import calculator
from .code import python_exec
from .datetime_tool import current_time
from .files import file_toolset, make_file_tools
from .grounding import grounding_settings
from .http import http_get
from .search import duckduckgo_provider, set_search_provider, web_search
from .url_context import fetch_url, make_fetch_url


def default_toolset() -> list:
    """A safe, read-only default set (no code exec, no network writes)."""
    return [calculator, current_time, http_get, web_search, fetch_url]


# --- component-registry catalog -----------------------------------------
# Register every built-in under the ``"tool"`` kind so they show up in
# ``available_components("tool")`` and can be built by name. Simple tools are
# singletons (the factory just returns the shared instance); the file tools take
# a ``root=`` kwarg so each call yields a sandbox-bound instance.
def _register_builtin_tools() -> None:
    for t in (
        calculator,
        current_time,
        http_get,
        web_search,
        python_exec,
        fetch_url,
        ask_user,
    ):
        register("tool", t.name, lambda _t=t, **_kw: _t)
    register("tool", "file_read", lambda *, root=".", **_kw: make_file_tools(root=root)[0])
    register("tool", "file_write", lambda *, root=".", **_kw: make_file_tools(root=root)[1])
    register("tool", "file_list", lambda *, root=".", **_kw: make_file_tools(root=root)[2])


_register_builtin_tools()


__all__ = [
    "calculator",
    "current_time",
    "http_get",
    "web_search",
    "python_exec",
    "fetch_url",
    "make_fetch_url",
    "duckduckgo_provider",
    "set_search_provider",
    "grounding_settings",
    "make_file_tools",
    "file_toolset",
    "ask_user",
    "default_toolset",
]
