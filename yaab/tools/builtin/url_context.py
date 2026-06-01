"""url_context tool — fetch a URL and hand the model readable page text.

``url_context`` lets a model pull a web page into its context window. It
fetches via ``httpx`` (lazy import), strips HTML down to
readable text reusing the RAG loader's BeautifulSoup-or-regex extractor (so the
behaviour is identical whether or not ``bs4`` is installed), and truncates to
``max_chars`` so a single huge page can't blow the context window.

Errors surface as ``error: ...`` strings rather than exceptions: the agent loop
feeds tool results back to the model, so a fetch failure becomes something the
model can read and react to instead of aborting the run.

The httpx client is injectable via :func:`make_fetch_url` so tests can drive it
with ``httpx.MockTransport`` and never touch the network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...rag.loaders import html_to_text
from ..base import FunctionTool, tool

#: Hard cap on bytes pulled into text extraction regardless of ``max_chars`` —
#: keeps a pathological response from ballooning memory before truncation.
_MAX_BYTES = 200_000


def _default_client_factory() -> Any:
    import httpx

    return httpx.AsyncClient(follow_redirects=True, timeout=15)


async def _fetch(url: str, max_chars: int, client_factory: Callable[[], Any]) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        return "error: only http/https URLs are allowed"
    try:
        import httpx  # noqa: F401  (lazy import so the core stays light)
    except ImportError:
        return "error: httpx is not installed (`pip install httpx`)"
    try:
        client = client_factory()
        async with client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw = resp.text[:_MAX_BYTES]
    except Exception as exc:  # noqa: BLE001 - report fetch failures to the model
        return f"error: failed to fetch {url}: {exc}"
    text = html_to_text(raw)
    return text[:max_chars]


def make_fetch_url(
    *, client_factory: Callable[[], Any] | None = None, name: str = "fetch_url"
) -> FunctionTool:
    """Build a ``fetch_url`` tool, optionally with an injected httpx client.

    ``client_factory`` returns a fresh ``httpx.AsyncClient`` per call (the tool
    uses it as an async context manager). Tests pass a factory wired to
    ``httpx.MockTransport``; production passes ``None`` to get the real client.
    """
    factory = client_factory or _default_client_factory

    @tool(name=name)
    async def fetch_url(url: str, max_chars: int = 8000) -> str:
        """Fetch a web page and return its readable text (HTML stripped).

        Use this to bring the contents of a specific URL into context. Returns
        up to ``max_chars`` characters of plain text; only http/https URLs are
        allowed. Fetch failures are returned as ``error: ...`` strings.
        """
        return await _fetch(url, max_chars, factory)

    return fetch_url


#: Default tool instance using the real httpx client (lazy-imported on first use).
fetch_url = make_fetch_url()

__all__ = ["fetch_url", "make_fetch_url"]
