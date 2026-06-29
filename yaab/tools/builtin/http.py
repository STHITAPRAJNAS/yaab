"""HTTP GET tool (read-only fetch). ``httpx`` is imported lazily."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from ...capabilities import Capability
from ..base import tool

_MAX_BYTES = 50_000


def _host_is_blocked(host: str) -> bool:
    """True if ``host`` resolves to a loopback/private/link-local/metadata IP.

    Blocks SSRF to cloud metadata (``169.254.169.254``), internal services, and
    localhost, and refuses unresolvable hosts. Validation is post-DNS so a public
    name that resolves to a private IP (rebinding) is still caught.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # unresolvable -> refuse
    for *_h, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


@tool(capabilities={Capability.NET_EGRESS})
async def http_get(url: str, max_chars: int = 10_000) -> str:
    """Fetch the text body of an HTTP(S) URL (GET only, read-only).

    Returns up to ``max_chars`` characters of the response body. Only http/https
    schemes are allowed. Hosts resolving to private/loopback/metadata IPs are
    refused (SSRF guard), and each redirect hop is re-validated.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return "error: only http/https URLs are allowed"
    try:
        import httpx
    except ImportError:
        return "error: httpx is not installed (`pip install httpx`)"
    host = urlparse(url).hostname or ""
    if _host_is_blocked(host):
        return f"error: host {host!r} is blocked (private/loopback/metadata)"
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=15) as client:
            for _hop in range(5):
                resp = await client.get(url)
                if resp.is_redirect and resp.next_request is not None:
                    url = str(resp.next_request.url)
                    nhost = urlparse(url).hostname or ""
                    if _host_is_blocked(nhost):
                        return f"error: redirect to blocked host {nhost!r}"
                    continue
                resp.raise_for_status()
                return resp.text[: min(max_chars, _MAX_BYTES)]
            return "error: too many redirects"
    except Exception as exc:  # noqa: BLE001 - report fetch failures to the model
        return f"error: failed to fetch {url}: {exc}"
