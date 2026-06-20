"""Browser-use tools — drive a real (headless) browser via Playwright.

``browser_toolset()`` returns navigate/click/type/extract/screenshot/back tools
bound to a lazily-started :class:`BrowserSession`. Playwright is an optional
extra (``pip install 'yaab-sdk[browser]' && playwright install chromium``),
imported lazily, so the SDK never needs it unless a browser tool actually runs.

Safety: a domain allowlist gates navigation; because these are ordinary tools,
the existing ``ToolApprovalPlugin`` / guardrails apply unchanged (e.g.
``ToolApprovalPlugin(tools=["browser_navigate"])`` gates navigation through HITL).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ...exceptions import ToolError
from ..base import FunctionTool, Tool

_INSTALL_HINT = (
    "browser tools need Playwright: pip install 'yaab-sdk[browser]' && playwright install chromium"
)


def _host_allowed(url: str, allow_domains: list[str] | None) -> bool:
    """True if ``url``'s host is permitted. ``None`` allows all; ``[]`` blocks all.

    Matching is suffix-on-label: ``example.com`` permits ``example.com`` and any
    subdomain (``www.example.com``) but not a look-alike (``notexample.com``).
    """
    if allow_domains is None:
        return True
    host = (urlparse(url).hostname or "").lower()
    for allowed in allow_domains:
        a = allowed.lower().lstrip(".")
        if host == a or host.endswith("." + a):
            return True
    return False


class BrowserSession:
    """A lazily-started browser page with safety-gated actions.

    Pass ``page`` to drive a pre-built page (used in tests); otherwise the first
    action launches headless Chromium via Playwright.
    """

    def __init__(
        self,
        *,
        allow_domains: list[str] | None = None,
        headless: bool = True,
        nav_timeout_s: float = 30.0,
        page: Any | None = None,
    ) -> None:
        self.allow_domains = allow_domains
        self.headless = headless
        self.nav_timeout_s = nav_timeout_s
        self._page = page
        self._pw: Any = None
        self._browser: Any = None

    async def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ToolError(_INSTALL_HINT) from exc
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page()
        return self._page

    async def navigate(self, url: str) -> str:
        if not _host_allowed(url, self.allow_domains):
            raise ToolError(f"navigation to {url!r} is blocked by the domain allowlist")
        page = await self._ensure_page()
        await page.goto(url)
        return f"navigated to {page.url}; title: {await page.title()}"

    async def click(self, selector: str) -> str:
        page = await self._ensure_page()
        try:
            await page.click(selector)
        except Exception as exc:  # noqa: BLE001 - report to the model, don't crash the run
            return f"error: could not click {selector!r}: {exc}"
        return f"clicked {selector!r}; now at {page.url}"

    async def type(self, selector: str, text: str, submit: bool = False) -> str:
        page = await self._ensure_page()
        try:
            await page.fill(selector, text)
            if submit:
                await page.press(selector, "Enter")
        except Exception as exc:  # noqa: BLE001
            return f"error: could not type into {selector!r}: {exc}"
        return f"typed into {selector!r}" + (" and submitted" if submit else "")

    async def extract(self, selector: str | None = None, *, max_chars: int = 5_000) -> str:
        page = await self._ensure_page()
        try:
            text = await page.inner_text(selector or "body")
        except Exception as exc:  # noqa: BLE001
            return f"error: could not extract {selector or 'body'!r}: {exc}"
        return text[:max_chars]

    async def screenshot(self, path: str = "screenshot.png") -> str:
        # The model controls ``path``; reduce to a bare filename so it can never
        # traverse out of the working directory (no ``../`` or absolute paths).
        import os

        safe = os.path.basename(path) or "screenshot.png"
        page = await self._ensure_page()
        data = await page.screenshot()
        with open(safe, "wb") as fh:
            fh.write(data)
        return f"saved screenshot to {safe} ({len(data)} bytes)"

    async def back(self) -> str:
        page = await self._ensure_page()
        await page.go_back()
        return f"went back; now at {page.url}"

    async def aclose(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            finally:
                if self._pw is not None:
                    await self._pw.stop()


class _BrowserToolList(list):
    """A tool list that also carries the bound :class:`BrowserSession`."""

    session: BrowserSession


def browser_toolset(
    *,
    allow_domains: list[str] | None = None,
    headless: bool = True,
    nav_timeout_s: float = 30.0,
    session: BrowserSession | None = None,
) -> list[Tool]:
    """Build browser tools bound to one :class:`BrowserSession`.

    The returned list carries the session on ``.session`` so the caller can
    ``await tools.session.aclose()`` at the end of a run.
    """
    sess = session or BrowserSession(
        allow_domains=allow_domains, headless=headless, nav_timeout_s=nav_timeout_s
    )

    async def browser_navigate(url: str) -> str:
        """Navigate the browser to ``url`` (subject to the domain allowlist)."""
        return await sess.navigate(url)

    async def browser_click(selector: str) -> str:
        """Click the element matching ``selector`` (CSS or ``text=...``)."""
        return await sess.click(selector)

    async def browser_type(selector: str, text: str, submit: bool = False) -> str:
        """Type ``text`` into ``selector``; set ``submit=True`` to press Enter."""
        return await sess.type(selector, text, submit=submit)

    async def browser_extract(selector: str | None = None) -> str:
        """Return the readable text of the page (or of ``selector`` if given)."""
        return await sess.extract(selector)

    async def browser_screenshot(path: str = "screenshot.png") -> str:
        """Save a PNG screenshot of the current page to ``path``."""
        return await sess.screenshot(path)

    async def browser_back() -> str:
        """Go back one entry in the browser history."""
        return await sess.back()

    tools = _BrowserToolList(
        [
            FunctionTool(browser_navigate, name="browser_navigate"),
            FunctionTool(browser_click, name="browser_click"),
            FunctionTool(browser_type, name="browser_type"),
            FunctionTool(browser_extract, name="browser_extract"),
            FunctionTool(browser_screenshot, name="browser_screenshot"),
            FunctionTool(browser_back, name="browser_back"),
        ]
    )
    tools.session = sess
    return tools


__all__ = ["BrowserSession", "browser_toolset"]
