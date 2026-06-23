"""Recipe: browser use — drive a real browser, safely.

``browser_toolset()`` exposes navigate/click/type/extract/screenshot tools bound
to a headless Playwright session, gated by a domain allowlist. This recipe shows
the wiring and the allowlist guard using an injected fake page (so it runs with
no browser); the flagship research app drives real Chromium.

    python -m cookbook.recipes.browser
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab.exceptions import ToolError
from yaab.tools.builtin.browser import BrowserSession, browser_toolset


class _FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"

    async def goto(self, url: str) -> None:
        self.url = url

    async def title(self) -> str:
        return "Example Domain"


async def run() -> dict:
    session = BrowserSession(allow_domains=["example.com"], page=_FakePage())
    tools = browser_toolset(session=session)
    names = sorted(t.name for t in tools)
    expect("browser_navigate" in names, "expected a navigate tool")

    # Allowed host works...
    ok = await session.navigate("https://example.com/")
    expect("example.com" in ok.lower(), "navigation to an allowed host should work")

    # ...off-allowlist navigation is blocked.
    blocked = False
    try:
        await session.navigate("https://evil.example.org/")
    except ToolError:
        blocked = True
    expect(blocked, "off-allowlist navigation must be blocked")
    return {"tools": names, "blocked_offsite": blocked}


if __name__ == "__main__":
    print(asyncio.run(run()))
