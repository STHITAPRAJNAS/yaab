"""Tests for the expanded built-in tool catalog.

Covers url_context (fetch_url), grounded_search/grounding_settings, the
DuckDuckGo provider for web_search, sandboxed file tools, and registration in
the component registry. All network is faked with ``httpx.MockTransport`` so the
suite stays offline and deterministic.
"""

from __future__ import annotations

import httpx
import pytest

from yaab.extensions import available as available_components
from yaab.extensions import get as get_component
from yaab.tools.builtin import default_toolset
from yaab.tools.builtin.files import file_toolset, make_file_tools
from yaab.tools.builtin.grounding import grounding_settings
from yaab.tools.builtin.search import duckduckgo_provider, set_search_provider, web_search
from yaab.tools.builtin.url_context import fetch_url, make_fetch_url
from yaab.types import RunContext


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- url_context --------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_url_strips_html_to_readable_text():
    html = (
        "<html><head><style>.x{color:red}</style>"
        "<script>evil()</script></head>"
        "<body><h1>Title</h1><p>Hello &amp; welcome</p></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/page"
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    tool = make_fetch_url(client_factory=lambda: _client(handler))
    out = await tool.execute(RunContext(), url="https://example.com/page")
    assert "Title" in out
    assert "Hello & welcome" in out
    # script/style bodies must be gone
    assert "evil()" not in out
    assert "color:red" not in out


@pytest.mark.asyncio
async def test_fetch_url_truncates_to_max_chars():
    body = "<p>" + ("A" * 5000) + "</p>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    tool = make_fetch_url(client_factory=lambda: _client(handler))
    out = await tool.execute(RunContext(), url="https://example.com", max_chars=100)
    assert len(out) <= 100


@pytest.mark.asyncio
async def test_fetch_url_rejects_non_http_scheme():
    out = await fetch_url.execute(RunContext(), url="ftp://example.com")
    assert out.startswith("error")


@pytest.mark.asyncio
async def test_fetch_url_reports_http_error_as_string():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    tool = make_fetch_url(client_factory=lambda: _client(handler))
    out = await tool.execute(RunContext(), url="https://example.com")
    assert out.startswith("error")


@pytest.mark.asyncio
async def test_fetch_url_reports_transport_error_as_string():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    tool = make_fetch_url(client_factory=lambda: _client(handler))
    out = await tool.execute(RunContext(), url="https://example.com")
    assert out.startswith("error")


# --- grounding_settings -------------------------------------------------
def test_grounding_settings_gemini_fragment():
    frag = grounding_settings(provider="gemini")
    assert frag == {"tools": [{"googleSearch": {}}]}


def test_grounding_settings_default_is_gemini():
    assert grounding_settings() == {"tools": [{"googleSearch": {}}]}


def test_grounding_settings_unknown_provider_raises():
    with pytest.raises(ValueError):
        grounding_settings(provider="nope")


def test_grounding_settings_is_pure_and_independent():
    a = grounding_settings(provider="gemini")
    b = grounding_settings(provider="gemini")
    a["tools"].append("mutated")
    # second call must not see the mutation of the first
    assert b == {"tools": [{"googleSearch": {}}]}


# --- duckduckgo provider for web_search ---------------------------------
_DDG_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="https://a.example/one">First Result</a>
  <a class="result__snippet">Snippet one text.</a>
</div>
<div class="result">
  <a class="result__a" href="https://b.example/two">Second Result</a>
  <a class="result__snippet">Snippet two text.</a>
</div>
</body></html>
"""


@pytest.mark.asyncio
async def test_duckduckgo_provider_parses_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "duckduckgo.com" in str(request.url)
        return httpx.Response(200, text=_DDG_HTML)

    provider = duckduckgo_provider(client_factory=lambda: _client(handler))
    results = await provider("python", 5)
    assert len(results) == 2
    assert results[0]["title"] == "First Result"
    assert results[0]["url"] == "https://a.example/one"
    assert "Snippet one" in results[0]["snippet"]


@pytest.mark.asyncio
async def test_web_search_uses_duckduckgo_provider_end_to_end():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_DDG_HTML)

    set_search_provider(duckduckgo_provider(client_factory=lambda: _client(handler)))
    try:
        out = await web_search.execute(RunContext(), query="python", max_results=5)
        assert "First Result" in out
        assert "https://a.example/one" in out
    finally:
        set_search_provider(None)


@pytest.mark.asyncio
async def test_duckduckgo_provider_needs_no_api_key():
    # Construction must not require any credential/env var.
    provider = duckduckgo_provider(
        client_factory=lambda: _client(lambda r: httpx.Response(200, text=""))
    )
    out = await provider("x", 3)
    assert out == []


# --- file tools (sandboxed) ---------------------------------------------
@pytest.mark.asyncio
async def test_file_tools_round_trip_read_write_list(tmp_path):
    read_file, write_file, list_directory = make_file_tools(root=str(tmp_path))

    w = await write_file.execute(RunContext(), path="notes.txt", content="hello world")
    assert "error" not in w.lower()

    r = await read_file.execute(RunContext(), path="notes.txt")
    assert r == "hello world"

    listed = await list_directory.execute(RunContext(), path=".")
    assert "notes.txt" in listed


@pytest.mark.asyncio
async def test_file_read_truncates_to_max_chars(tmp_path):
    read_file, write_file, _ = make_file_tools(root=str(tmp_path))
    await write_file.execute(RunContext(), path="big.txt", content="Z" * 5000)
    out = await read_file.execute(RunContext(), path="big.txt", max_chars=50)
    assert len(out) <= 50


@pytest.mark.asyncio
async def test_file_list_respects_glob(tmp_path):
    read_file, write_file, list_directory = make_file_tools(root=str(tmp_path))
    await write_file.execute(RunContext(), path="a.txt", content="1")
    await write_file.execute(RunContext(), path="b.md", content="2")
    out = await list_directory.execute(RunContext(), path=".", glob="*.txt")
    assert "a.txt" in out
    assert "b.md" not in out


@pytest.mark.asyncio
async def test_file_read_rejects_traversal(tmp_path):
    read_file, _, _ = make_file_tools(root=str(tmp_path))
    out = await read_file.execute(RunContext(), path="../../etc/passwd")
    assert out.startswith("error")


@pytest.mark.asyncio
async def test_file_write_rejects_traversal(tmp_path):
    _, write_file, _ = make_file_tools(root=str(tmp_path))
    out = await write_file.execute(RunContext(), path="../escape.txt", content="x")
    assert out.startswith("error")


@pytest.mark.asyncio
async def test_file_list_rejects_traversal(tmp_path):
    _, _, list_directory = make_file_tools(root=str(tmp_path))
    out = await list_directory.execute(RunContext(), path="..")
    assert out.startswith("error")


@pytest.mark.asyncio
async def test_file_read_missing_file_reports_error(tmp_path):
    read_file, _, _ = make_file_tools(root=str(tmp_path))
    out = await read_file.execute(RunContext(), path="ghost.txt")
    assert out.startswith("error")


@pytest.mark.asyncio
async def test_file_toolset_returns_three_tools(tmp_path):
    tools = file_toolset(root=str(tmp_path))
    names = {t.name for t in tools}
    assert names == {"file_read", "file_write", "file_list"}


# --- registration / discovery -------------------------------------------
def test_builtin_tools_registered_in_catalog():
    names = set(available_components("tool"))
    for expected in (
        "calculator",
        "current_time",
        "http_get",
        "web_search",
        "python_exec",
        "fetch_url",
        "file_read",
        "file_write",
        "file_list",
    ):
        assert expected in names, f"{expected} missing from tool catalog: {sorted(names)}"


def test_get_component_returns_a_tool():
    calc = get_component("tool", "calculator")
    assert calc.name == "calculator"
    assert hasattr(calc, "execute")


def test_get_file_tool_component_accepts_root(tmp_path):
    tool = get_component("tool", "file_read", root=str(tmp_path))
    assert tool.name == "file_read"


def test_default_toolset_unchanged():
    names = {t.name for t in default_toolset()}
    assert "python_exec" not in names
    assert "calculator" in names
