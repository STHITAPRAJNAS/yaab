from __future__ import annotations

import pytest


def test_host_allowed_suffix_matching():
    from yaab.tools.builtin.browser import _host_allowed

    # None = allow all (dev only); [] = block all.
    assert _host_allowed("https://anything.com/x", None) is True
    assert _host_allowed("https://anything.com/x", []) is False
    # Suffix match: allowing example.com permits its subdomains.
    assert _host_allowed("https://example.com/p", ["example.com"]) is True
    assert _host_allowed("https://www.example.com/p", ["example.com"]) is True
    assert _host_allowed("https://evil.com/p", ["example.com"]) is False
    # A look-alike suffix is not allowed (notexample.com != example.com).
    assert _host_allowed("https://notexample.com", ["example.com"]) is False


class _FakePage:
    """A minimal stand-in for a Playwright page for offline tests."""

    def __init__(self) -> None:
        self.url = "about:blank"
        self._title = "Blank"
        self.typed: dict[str, str] = {}
        self.clicked: list[str] = []
        self.pressed: list[tuple[str, str]] = []

    async def goto(self, url: str) -> None:
        self.url = url
        self._title = f"Title of {url}"

    async def title(self) -> str:
        return self._title

    async def click(self, selector: str) -> None:
        self.clicked.append(selector)

    async def fill(self, selector: str, text: str) -> None:
        self.typed[selector] = text

    async def press(self, selector: str, key: str) -> None:
        self.pressed.append((selector, key))

    async def inner_text(self, selector: str) -> str:
        return f"text[{selector}]"

    async def screenshot(self) -> bytes:
        return b"\x89PNG-fake-bytes"

    async def go_back(self) -> None:
        self.url = "about:blank"


@pytest.mark.asyncio
async def test_browser_session_navigate_respects_allowlist():
    from yaab.exceptions import ToolError
    from yaab.tools.builtin.browser import BrowserSession

    page = _FakePage()
    session = BrowserSession(allow_domains=["example.com"], page=page)

    out = await session.navigate("https://example.com/start")
    assert "example.com/start" in out
    assert page.url == "https://example.com/start"

    with pytest.raises(ToolError):
        await session.navigate("https://evil.com/")


@pytest.mark.asyncio
async def test_browser_session_actions():
    from yaab.tools.builtin.browser import BrowserSession

    page = _FakePage()
    session = BrowserSession(page=page)  # allow all (dev)

    await session.navigate("https://example.com")
    await session.type("#q", "hello", submit=True)
    assert page.typed["#q"] == "hello"
    assert ("#q", "Enter") in page.pressed

    await session.click("#go")
    assert "#go" in page.clicked

    text = await session.extract("#main")
    assert text == "text[#main]"

    out = await session.screenshot(str_path := "shot.png")
    assert str_path in out
    import os

    assert os.path.exists("shot.png")
    os.remove("shot.png")


@pytest.mark.asyncio
async def test_browser_toolset_wires_to_session():
    from yaab.tools.builtin.browser import BrowserSession, browser_toolset

    page = _FakePage()
    session = BrowserSession(allow_domains=None, page=page)
    tools = browser_toolset(session=session)

    names = {t.name for t in tools}
    assert names == {
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_extract",
        "browser_screenshot",
        "browser_back",
    }
    assert tools.session is session

    # Execute a tool the way the runner does (FunctionTool.execute(ctx, **kwargs)).
    from yaab.types import RunContext

    nav = next(t for t in tools if t.name == "browser_navigate")
    out = await nav.execute(RunContext(), url="https://example.com")
    assert "example.com" in out
    assert page.url == "https://example.com"


@pytest.mark.asyncio
async def test_missing_playwright_raises_with_hint(monkeypatch):
    import builtins

    from yaab.exceptions import ToolError
    from yaab.tools.builtin.browser import BrowserSession

    real_import = builtins.__import__

    def _no_playwright(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("no playwright")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_playwright)

    session = BrowserSession()  # no injected page -> must launch -> import fails
    with pytest.raises(ToolError, match="Playwright"):
        await session.navigate("https://example.com")


def test_browser_not_in_default_toolset():
    from yaab.tools.builtin import default_toolset

    names = {t.name for t in default_toolset()}
    assert not any(n.startswith("browser_") for n in names)
