"""Explicit prompt/context caching write-side.

YAAB already READS cached-token counts; these tests pin the WRITE side: when
``cache_system_prompt`` / ``cache_tools`` are enabled, ``LiteLLMModel`` must
inject the provider-specific cache directives litellm understands. Anthropic
models get ``cache_control`` blocks; OpenAI (and other unsupported) providers
are left byte-for-byte untouched; Gemini passes a user-supplied
``cached_content`` through.

All exercised against a monkeypatched fake litellm (no network) following the
fake-litellm pattern from ``test_model_resilience_depth.py``.
"""

from __future__ import annotations

import pytest

from yaab.models.litellm_provider import LiteLLMModel
from yaab.types import Message, Role


class _Capture:
    """A fake litellm that records the kwargs passed to acompletion."""

    def __init__(self) -> None:
        self.captured: dict | None = None
        self.stream_captured: dict | None = None

    async def acompletion(self, **kw):
        if kw.get("stream"):
            self.stream_captured = kw
            return _stream_resp()
        self.captured = kw
        return _Resp()

    def completion_cost(self, **kw):
        return 0.0


class _Msg:
    content = "ok"
    tool_calls = None
    reasoning_content = None


class _Choice:
    finish_reason = "stop"
    message = _Msg()


class _Resp:
    choices = [_Choice()]
    usage = None


async def _stream_resp():
    class _Delta:
        content = "hi"
        tool_calls = None

    class _SChoice:
        delta = _Delta()

    class _Chunk:
        choices = [_SChoice()]

    yield _Chunk()


def _patch(monkeypatch, fake):
    monkeypatch.setattr("yaab.models.litellm_provider._require_litellm", lambda: fake)


SYS = Message(role=Role.SYSTEM, content="You are a careful assistant.")
USER = Message(role=Role.USER, content="hi")
TOOLS = [
    {"type": "function", "function": {"name": "a", "parameters": {}}},
    {"type": "function", "function": {"name": "b", "parameters": {}}},
]


def _system_block(captured: dict) -> dict:
    """Return the system message litellm received."""
    return next(m for m in captured["messages"] if m["role"] == "system")


# --- Anthropic: system-prompt caching ---------------------------------------
@pytest.mark.asyncio
async def test_anthropic_caches_system_prompt(monkeypatch):
    fake = _Capture()
    _patch(monkeypatch, fake)
    model = LiteLLMModel("anthropic/claude-3-5-sonnet", cache_system_prompt=True)

    await model.complete([SYS, USER])

    sys_msg = _system_block(fake.captured)
    # The system content is rewritten into the list-block form.
    assert isinstance(sys_msg["content"], list)
    last_block = sys_msg["content"][-1]
    assert last_block["type"] == "text"
    assert last_block["text"] == "You are a careful assistant."
    assert last_block["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_claude_name_also_triggers_caching(monkeypatch):
    # The model-name heuristic accepts 'claude' as well as 'anthropic'.
    fake = _Capture()
    _patch(monkeypatch, fake)
    model = LiteLLMModel("bedrock/claude-3-haiku", cache_system_prompt=True)

    await model.complete([SYS, USER])

    sys_msg = _system_block(fake.captured)
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}


# --- Anthropic: tool caching ------------------------------------------------
@pytest.mark.asyncio
async def test_anthropic_caches_last_tool(monkeypatch):
    fake = _Capture()
    _patch(monkeypatch, fake)
    model = LiteLLMModel("anthropic/claude-3-5-sonnet", cache_tools=True)

    await model.complete([USER], tools=TOOLS)

    sent_tools = fake.captured["tools"]
    # Only the LAST tool carries the breakpoint (caches everything up to it).
    assert "cache_control" not in sent_tools[0]
    assert sent_tools[-1]["cache_control"] == {"type": "ephemeral"}
    # The original tool list passed in is not mutated.
    assert "cache_control" not in TOOLS[-1]


# --- OpenAI / unsupported providers: untouched ------------------------------
@pytest.mark.asyncio
async def test_openai_payload_not_modified(monkeypatch):
    fake = _Capture()
    _patch(monkeypatch, fake)
    model = LiteLLMModel("openai/gpt-4o", cache_system_prompt=True, cache_tools=True)

    await model.complete([SYS, USER], tools=TOOLS)

    sys_msg = _system_block(fake.captured)
    # Plain string content, no cache_control anywhere.
    assert sys_msg["content"] == "You are a careful assistant."
    assert all("cache_control" not in t for t in fake.captured["tools"])


@pytest.mark.asyncio
async def test_anthropic_no_caching_when_disabled(monkeypatch):
    fake = _Capture()
    _patch(monkeypatch, fake)
    model = LiteLLMModel("anthropic/claude-3-5-sonnet")  # flags default False

    await model.complete([SYS, USER], tools=TOOLS)

    sys_msg = _system_block(fake.captured)
    assert sys_msg["content"] == "You are a careful assistant."
    assert all("cache_control" not in t for t in fake.captured["tools"])


# --- Gemini: pass-through cached_content ------------------------------------
@pytest.mark.asyncio
async def test_gemini_passes_cached_content(monkeypatch):
    fake = _Capture()
    _patch(monkeypatch, fake)
    # cached_content supplied via model settings flows through untouched; no
    # cache_control blocks are injected (Gemini uses its own kwarg / implicit
    # caching).
    model = LiteLLMModel("gemini/gemini-1.5-pro", cached_content="cached/ctx/123")

    await model.complete([SYS, USER])

    assert fake.captured["cached_content"] == "cached/ctx/123"
    sys_msg = _system_block(fake.captured)
    assert sys_msg["content"] == "You are a careful assistant."


# --- streaming path also caches ---------------------------------------------
@pytest.mark.asyncio
async def test_stream_also_injects_cache_control(monkeypatch):
    fake = _Capture()
    _patch(monkeypatch, fake)
    model = LiteLLMModel("anthropic/claude-3-5-sonnet", cache_system_prompt=True)

    chunks = [c async for c in model.stream([SYS, USER])]
    assert chunks  # consumed the stream

    sys_msg = _system_block(fake.stream_captured)
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_no_system_message_is_safe(monkeypatch):
    # With caching on but no system message, nothing blows up and the payload is
    # left as-is (no spurious system block created).
    fake = _Capture()
    _patch(monkeypatch, fake)
    model = LiteLLMModel("anthropic/claude-3-5-sonnet", cache_system_prompt=True)

    await model.complete([USER])

    assert all(m["role"] != "system" for m in fake.captured["messages"])
