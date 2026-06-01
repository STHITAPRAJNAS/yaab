"""Web search tool — provider-pluggable, with a clear "configure me" default.

Real search needs an API key (Tavily, Brave, SerpAPI, ...). Rather than bake in
one vendor, :func:`set_search_provider` registers an async ``(query, k) -> list``
callable; the :func:`web_search` tool calls it. Without a provider configured,
the tool returns a helpful message instead of failing — so an agent wired with
the default toolset degrades gracefully offline.

For a zero-config option, :func:`duckduckgo_provider` builds a provider that
hits DuckDuckGo's keyless HTML endpoint and parses the top results, needing no
API key and no extra dependencies (just ``httpx``, lazy-imported). Its httpx
client is injectable so tests drive it with ``httpx.MockTransport``.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..base import tool

SearchProvider = Callable[[str, int], Awaitable[list[dict]]]
_provider: SearchProvider | None = None

#: DuckDuckGo's no-JS HTML endpoint — returns plain server-rendered results.
_DDG_URL = "https://html.duckduckgo.com/html/"

# Parse the result blocks out of the HTML the no-JS endpoint returns. We pull the
# anchor href + text for the title/url, then the snippet anchor. Kept as regexes
# so the provider stays dependency-free (no bs4 required).
_DDG_RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip(fragment: str) -> str:
    """Drop inline tags and unescape entities from a result fragment."""
    return _html.unescape(_TAG_RE.sub("", fragment)).strip()


def _default_ddg_client_factory() -> Any:
    import httpx

    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; yaab/1.0)"},
    )


def duckduckgo_provider(*, client_factory: Callable[[], Any] | None = None) -> SearchProvider:
    """Build a keyless web-search provider backed by DuckDuckGo's HTML endpoint.

    Needs no API key and no dependency beyond ``httpx`` (lazy-imported). Pass a
    ``client_factory`` returning an ``httpx.AsyncClient`` to inject a mock
    transport in tests; production uses the real client.

    Register it with :func:`set_search_provider` to power :func:`web_search`.
    """
    factory = client_factory or _default_ddg_client_factory

    async def _search(query: str, k: int) -> list[dict]:
        client = factory()
        async with client:
            resp = await client.post(_DDG_URL, data={"q": query})
            resp.raise_for_status()
            body = resp.text
        snippets = [_strip(m.group("snippet")) for m in _DDG_SNIPPET_RE.finditer(body)]
        results: list[dict] = []
        for i, m in enumerate(_DDG_RESULT_RE.finditer(body)):
            results.append(
                {
                    "title": _strip(m.group("title")),
                    "url": _html.unescape(m.group("url")).strip(),
                    "snippet": snippets[i] if i < len(snippets) else "",
                }
            )
            if len(results) >= k:
                break
        return results

    return _search


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
