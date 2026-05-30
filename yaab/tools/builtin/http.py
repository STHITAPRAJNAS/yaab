"""HTTP GET tool (read-only fetch). ``httpx`` is imported lazily."""

from __future__ import annotations

from ..base import tool

_MAX_BYTES = 50_000


@tool
async def http_get(url: str, max_chars: int = 10_000) -> str:
    """Fetch the text body of an HTTP(S) URL (GET only, read-only).

    Returns up to ``max_chars`` characters of the response body. Only http/https
    schemes are allowed.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return "error: only http/https URLs are allowed"
    try:
        import httpx
    except ImportError:
        return "error: httpx is not installed (`pip install httpx`)"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text[: min(max_chars, _MAX_BYTES)]
            return text
    except Exception as exc:  # noqa: BLE001 - report fetch failures to the model
        return f"error: failed to fetch {url}: {exc}"
