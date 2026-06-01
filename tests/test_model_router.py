"""Dynamic model routing (ADK model-routing parity).

A :class:`ModelRouter` is a ``ModelProvider`` that picks a downstream model per
request via a classifier (built-in ``'length'`` or any sync/async callable),
resolving string route specs lazily. These tests pin: length-based routing,
tool-presence routing, custom callables (sync + async), lazy string resolution,
``last_route`` observability, and registry registration.
"""

from __future__ import annotations

import pytest

from yaab.models import ModelProvider
from yaab.models.router import ModelRouter
from yaab.models.test_model import TestModel
from yaab.types import Message, Role


def _msgs(text: str) -> list[Message]:
    return [Message(role=Role.USER, content=text)]


# --- length classifier ------------------------------------------------------
@pytest.mark.asyncio
async def test_length_routes_short_to_simple():
    simple = TestModel("simple-answer")
    complex_ = TestModel("complex-answer")
    router = ModelRouter({"simple": simple, "complex": complex_}, classifier="length")

    resp = await router.complete(_msgs("hello"))
    assert resp.content == "simple-answer"
    assert router.last_route == "simple"


@pytest.mark.asyncio
async def test_length_routes_long_to_complex():
    simple = TestModel("simple-answer")
    complex_ = TestModel("complex-answer")
    router = ModelRouter(
        {"simple": simple, "complex": complex_},
        classifier="length",
        complexity_threshold=20,
    )

    resp = await router.complete(_msgs("x" * 50))
    assert resp.content == "complex-answer"
    assert router.last_route == "complex"


@pytest.mark.asyncio
async def test_length_routes_to_complex_when_tools_present():
    simple = TestModel("simple-answer")
    complex_ = TestModel("complex-answer")
    router = ModelRouter({"simple": simple, "complex": complex_}, classifier="length")

    tools = [{"type": "function", "function": {"name": "a", "parameters": {}}}]
    resp = await router.complete(_msgs("hi"), tools=tools)
    assert resp.content == "complex-answer"
    assert router.last_route == "complex"


@pytest.mark.asyncio
async def test_default_classifier_is_length():
    # No classifier given -> defaults to the built-in length classifier.
    router = ModelRouter({"simple": TestModel("s"), "complex": TestModel("c")})
    await router.complete(_msgs("short"))
    assert router.last_route == "simple"


# --- custom callable classifiers --------------------------------------------
@pytest.mark.asyncio
async def test_custom_sync_classifier():
    def pick(messages, tools):
        return "complex" if "urgent" in messages[-1].content else "simple"

    router = ModelRouter({"simple": TestModel("s"), "complex": TestModel("c")}, classifier=pick)
    assert (await router.complete(_msgs("urgent please"))).content == "c"
    assert router.last_route == "complex"
    assert (await router.complete(_msgs("relax"))).content == "s"
    assert router.last_route == "simple"


@pytest.mark.asyncio
async def test_custom_async_classifier():
    async def pick(messages, tools):
        return "complex"

    router = ModelRouter({"simple": TestModel("s"), "complex": TestModel("c")}, classifier=pick)
    assert (await router.complete(_msgs("anything"))).content == "c"
    assert router.last_route == "complex"


# --- unknown route falls back to default ------------------------------------
@pytest.mark.asyncio
async def test_unknown_route_falls_back_to_default():
    router = ModelRouter(
        {"simple": TestModel("s"), "complex": TestModel("c")},
        classifier=lambda m, t: "nonexistent",
    )
    resp = await router.complete(_msgs("hi"))
    # default is the first route key ("simple").
    assert resp.content == "s"
    assert router.last_route == "simple"


@pytest.mark.asyncio
async def test_explicit_default_route():
    router = ModelRouter(
        {"simple": TestModel("s"), "complex": TestModel("c")},
        classifier=lambda m, t: "bogus",
        default="complex",
    )
    resp = await router.complete(_msgs("hi"))
    assert resp.content == "c"
    assert router.last_route == "complex"


# --- lazy string route resolution -------------------------------------------
@pytest.mark.asyncio
async def test_string_route_specs_resolve_lazily(monkeypatch):
    # A string route spec is resolved through resolve_model only when chosen.
    resolved: list[str] = []

    sentinel = TestModel("from-spec")

    def fake_resolve(spec):
        resolved.append(spec)
        return sentinel

    monkeypatch.setattr("yaab.models.router.resolve_model", fake_resolve)

    router = ModelRouter(
        {"simple": "openai/gpt-4o-mini", "complex": "openai/gpt-4o"},
        classifier="length",
    )
    # Nothing resolved at construction time.
    assert resolved == []

    resp = await router.complete(_msgs("short"))
    assert resp.content == "from-spec"
    # Only the chosen route was resolved.
    assert resolved == ["openai/gpt-4o-mini"]
    assert router.last_route == "simple"


@pytest.mark.asyncio
async def test_resolved_string_route_is_cached(monkeypatch):
    calls: list[str] = []

    def fake_resolve(spec):
        calls.append(spec)
        return TestModel("x")

    monkeypatch.setattr("yaab.models.router.resolve_model", fake_resolve)

    router = ModelRouter({"simple": "openai/gpt-4o-mini"}, classifier="length")
    await router.complete(_msgs("a"))
    await router.complete(_msgs("b"))
    # Resolved once, then reused.
    assert calls == ["openai/gpt-4o-mini"]


# --- streaming delegates too ------------------------------------------------
@pytest.mark.asyncio
async def test_stream_delegates_and_records_route():
    router = ModelRouter(
        {"simple": TestModel("hello world"), "complex": TestModel("c")},
        classifier="length",
    )
    chunks = [c async for c in router.stream(_msgs("hi"))]
    text = "".join(c.delta for c in chunks)
    assert text == "hello world"
    assert router.last_route == "simple"


# --- kwargs pass through ----------------------------------------------------
@pytest.mark.asyncio
async def test_kwargs_pass_through_to_delegate():
    simple = TestModel("s")
    router = ModelRouter({"simple": simple, "complex": TestModel("c")}, classifier="length")
    await router.complete(_msgs("hi"), temperature=0.3)
    assert simple.call_kwargs[-1].get("temperature") == 0.3


# --- protocol conformance + registry ----------------------------------------
def test_router_satisfies_model_provider_protocol():
    router = ModelRouter({"simple": TestModel("s")})
    assert isinstance(router, ModelProvider)
    assert router.name == "router"


def test_router_registered_in_component_registry():
    from yaab.extensions import available, get

    assert "router" in available("model")
    built = get(
        "model",
        "router",
        routes={"simple": TestModel("s"), "complex": TestModel("c")},
    )
    assert isinstance(built, ModelRouter)


def test_empty_routes_rejected():
    with pytest.raises(ValueError):
        ModelRouter({})
