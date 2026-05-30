"""Web search tool — provider-pluggable, with a clear "configure me" default.

Real search needs an API key (Tavily, Brave, SerpAPI, ...). Rather than bake in
one vendor, :func:`set_search_provider` registers an async ``(query, k) -> list``
callable; the :func:`web_search` tool calls it. Without a provider configured,
the tool returns a helpful message instead of failing — so an agent wired with
the default toolset degrades gracefully offline.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..base import tool

SearchProvider = Callable[[str, int], Awaitable[list[dict]]]
_provider: SearchProvider | None = None


def set_search_provider(provider: SearchProvider | None) -> None:
    """Register the backend used by :func:`web_search` (or clear it with None)."""
    global _provider
    _provider = provider


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return titled result snippets.

    Requires a search provider configured via
    ``yaab.tools.builtin.search.set_search_provider``; otherwise returns a
    configuration hint so the agent can proceed without crashing.
    """
    if _provider is None:
        return (
            "error: no web search provider configured. Call "
            "yaab.tools.builtin.search.set_search_provider(fn) with a Tavily/"
            "Brave/SerpAPI-backed async function."
        )
    try:
        results = await _provider(query, max_results)
    except Exception as exc:  # noqa: BLE001
        return f"error: search failed: {exc}"
    lines = []
    for r in results[:max_results]:
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("snippet") or r.get("content", "")
        lines.append(f"- {title} ({url})\n  {snippet}")
    return "\n".join(lines) if lines else "No results."
