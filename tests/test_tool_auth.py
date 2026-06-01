"""Tests for tool-level authentication (credentials + OAuth2 consent surface).

Covers the YAAB equivalent of ADK's tool ``auth_scheme``/``auth_credential``:

* :func:`as_headers` renders all four credential kinds to HTTP headers.
* :meth:`ToolAuth.resolve` returns a static credential, calls a sync/async
  ``credential_provider`` (passing ``ctx`` so it can look up per-identity
  tokens), or raises :class:`ToolAuthRequired` when nothing can be resolved.
* An expired static credential (``expires_at`` in the past) is ignored and the
  provider is consulted instead — so short-lived OAuth tokens get refreshed.
* :class:`FunctionTool` injects the resolved credential into the wrapped
  function via an ``auth_headers`` param, a ``credential`` param, or ``ctx.state``.
* A missing credential surfaces as a model-visible ``error: ... requires
  authorization`` string (never a crash) — proven through a full ``Agent.run``
  with ``TestModel`` driving the tool call and the loop continuing afterward.
* The ``@tool(auth=...)`` decorator form wires auth the same way.
"""

from __future__ import annotations

import base64
import time

import pytest

from yaab import Agent, RunContext, tool
from yaab.models.test_model import TestModel
from yaab.tools.auth import ToolAuth, ToolAuthRequired, ToolCredential, as_headers
from yaab.tools.base import FunctionTool

# --------------------------------------------------------------------------- #
# as_headers — all four credential kinds                                       #
# --------------------------------------------------------------------------- #


def test_as_headers_api_key_uses_named_header():
    cred = ToolCredential(kind="api_key", value="secret", header="X-API-Key")
    assert as_headers(cred) == {"X-API-Key": "secret"}


def test_as_headers_api_key_defaults_header():
    cred = ToolCredential(kind="api_key", value="secret")
    assert as_headers(cred) == {"x-api-key": "secret"}


def test_as_headers_bearer():
    cred = ToolCredential(kind="bearer", token="tok123")
    assert as_headers(cred) == {"Authorization": "Bearer tok123"}


def test_as_headers_oauth2_uses_bearer_scheme():
    cred = ToolCredential(kind="oauth2", token="oauthtok")
    assert as_headers(cred) == {"Authorization": "Bearer oauthtok"}


def test_as_headers_basic_base64_encodes():
    cred = ToolCredential(kind="basic", value="alice", token="pw")
    expected = base64.b64encode(b"alice:pw").decode()
    assert as_headers(cred) == {"Authorization": f"Basic {expected}"}


# --------------------------------------------------------------------------- #
# ToolAuth.resolve — static, provider (sync/async/per-identity), required      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolve_returns_static_credential():
    cred = ToolCredential(kind="api_key", value="k")
    auth = ToolAuth(scheme="api_key", credential=cred)
    assert await auth.resolve(RunContext()) is cred


@pytest.mark.asyncio
async def test_resolve_calls_sync_provider():
    def provider(ctx: RunContext) -> ToolCredential:
        return ToolCredential(kind="bearer", token="from-sync")

    auth = ToolAuth(scheme="bearer", credential_provider=provider)
    cred = await auth.resolve(RunContext())
    assert cred.token == "from-sync"


@pytest.mark.asyncio
async def test_resolve_calls_async_provider_with_identity():
    tokens = {"alice": "alice-tok", "bob": "bob-tok"}

    async def provider(ctx: RunContext) -> ToolCredential:
        return ToolCredential(kind="bearer", token=tokens[ctx.identity])

    auth = ToolAuth(scheme="bearer", credential_provider=provider)
    cred = await auth.resolve(RunContext(identity="bob"))
    assert cred.token == "bob-tok"


@pytest.mark.asyncio
async def test_resolve_raises_when_nothing_resolvable():
    auth = ToolAuth(
        scheme="oauth2",
        consent_url="https://idp.example/consent",
        scopes=["read", "write"],
    )
    with pytest.raises(ToolAuthRequired) as exc:
        await auth.resolve(RunContext(), tool_name="search")
    assert exc.value.consent_url == "https://idp.example/consent"
    assert exc.value.scopes == ["read", "write"]
    assert exc.value.tool == "search"


@pytest.mark.asyncio
async def test_resolve_expired_static_credential_falls_back_to_provider():
    expired = ToolCredential(kind="bearer", token="stale", expires_at=time.time() - 60)
    calls = {"n": 0}

    async def provider(ctx: RunContext) -> ToolCredential:
        calls["n"] += 1
        return ToolCredential(kind="bearer", token="fresh")

    auth = ToolAuth(scheme="bearer", credential=expired, credential_provider=provider)
    cred = await auth.resolve(RunContext())
    assert cred.token == "fresh"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_resolve_expired_without_provider_raises():
    expired = ToolCredential(kind="bearer", token="stale", expires_at=time.time() - 60)
    auth = ToolAuth(scheme="bearer", credential=expired, consent_url="https://c")
    with pytest.raises(ToolAuthRequired):
        await auth.resolve(RunContext(), tool_name="t")


def test_credential_is_expired_helper():
    assert ToolCredential(kind="bearer", token="x", expires_at=time.time() - 1).is_expired()
    assert not ToolCredential(kind="bearer", token="x", expires_at=time.time() + 60).is_expired()
    assert not ToolCredential(kind="bearer", token="x").is_expired()  # no expiry -> never


# --------------------------------------------------------------------------- #
# FunctionTool credential injection                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_api_key_injected_as_auth_headers_param():
    seen: dict = {}

    def call_api(query: str, auth_headers: dict) -> str:
        seen.update(auth_headers)
        return "ok"

    auth = ToolAuth(
        scheme="api_key",
        credential=ToolCredential(kind="api_key", value="K", header="X-Key"),
    )
    t = FunctionTool(call_api, auth=auth)
    # auth_headers must not leak into the model-facing schema.
    props = t.schema()["function"]["parameters"]["properties"]
    assert "auth_headers" not in props
    assert "query" in props

    out = await t.execute(RunContext(), query="hi")
    assert out == "ok"
    assert seen == {"X-Key": "K"}


@pytest.mark.asyncio
async def test_credential_injected_as_credential_param():
    seen: dict = {}

    def call_api(credential: ToolCredential) -> str:
        seen["token"] = credential.token
        return "ok"

    auth = ToolAuth(
        scheme="bearer",
        credential=ToolCredential(kind="bearer", token="tok"),
    )
    t = FunctionTool(call_api, auth=auth)
    assert "credential" not in t.schema()["function"]["parameters"]["properties"]
    await t.execute(RunContext())
    assert seen["token"] == "tok"


@pytest.mark.asyncio
async def test_credential_stashed_on_ctx_state_when_no_param():
    auth = ToolAuth(scheme="bearer", credential=ToolCredential(kind="bearer", token="tok"))

    def plain() -> str:
        return "ran"

    t = FunctionTool(plain, auth=auth)
    ctx = RunContext()
    out = await t.execute(ctx)
    assert out == "ran"
    stashed = ctx.state["__tool_credential__"]
    assert isinstance(stashed, ToolCredential)
    assert stashed.token == "tok"


@pytest.mark.asyncio
async def test_provider_receives_identity_through_execute():
    captured: dict = {}

    async def provider(ctx: RunContext) -> ToolCredential:
        captured["identity"] = ctx.identity
        return ToolCredential(kind="bearer", token=f"tok-{ctx.identity}")

    def call_api(auth_headers: dict) -> str:
        return auth_headers["Authorization"]

    auth = ToolAuth(scheme="bearer", credential_provider=provider)
    t = FunctionTool(call_api, auth=auth)
    out = await t.execute(RunContext(identity="carol"))
    assert out == "Bearer tok-carol"
    assert captured["identity"] == "carol"


@pytest.mark.asyncio
async def test_expired_credential_reresolves_via_provider_in_execute():
    expired = ToolCredential(kind="bearer", token="stale", expires_at=time.time() - 5)

    async def provider(ctx: RunContext) -> ToolCredential:
        return ToolCredential(kind="bearer", token="fresh")

    def call_api(auth_headers: dict) -> str:
        return auth_headers["Authorization"]

    auth = ToolAuth(scheme="bearer", credential=expired, credential_provider=provider)
    t = FunctionTool(call_api, auth=auth)
    out = await t.execute(RunContext())
    assert out == "Bearer fresh"


# --------------------------------------------------------------------------- #
# Missing credential -> model-visible error string (no crash)                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_credential_returns_error_string_not_raise():
    auth = ToolAuth(
        scheme="oauth2",
        consent_url="https://idp.example/oauth/consent",
        scopes=["calendar.read"],
    )

    def call_api(auth_headers: dict) -> str:  # never reached
        return "should not run"

    t = FunctionTool(call_api, name="calendar", auth=auth)
    out = await t.execute(RunContext())
    assert isinstance(out, str)
    assert out.startswith("error: tool calendar requires authorization")
    assert "https://idp.example/oauth/consent" in out
    assert "calendar.read" in out


@pytest.mark.asyncio
async def test_missing_credential_through_full_agent_run_does_not_crash():
    auth = ToolAuth(
        scheme="oauth2",
        consent_url="https://idp.example/oauth/consent",
        scopes=["mail.send"],
    )

    @tool(name="send_mail", auth=auth)
    def send_mail(auth_headers: dict) -> str:
        """Send mail (requires OAuth)."""
        return "sent"

    model = TestModel(custom_output="told the user to authorize", call_tools=["send_mail"])
    agent = Agent("a", model=model, tools=[send_mail])
    result = await agent.run("send an email")
    # The loop must complete normally; the model produced a final answer after
    # seeing the authorization error in the tool result.
    assert result.output == "told the user to authorize"


# --------------------------------------------------------------------------- #
# @tool(auth=...) decorator form                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_decorator_passes_auth_through():
    auth = ToolAuth(scheme="api_key", credential=ToolCredential(kind="api_key", value="K"))

    @tool(auth=auth)
    def fetch(auth_headers: dict) -> str:
        """Fetch with an injected API key."""
        return auth_headers["x-api-key"]

    assert isinstance(fetch, FunctionTool)
    assert fetch.auth is auth
    out = await fetch.execute(RunContext())
    assert out == "K"


def test_tool_without_auth_has_none():
    @tool
    def plain() -> str:
        """No auth."""
        return "x"

    assert plain.auth is None
