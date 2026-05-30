"""Tests for the session/memory/artifact managers and extensibility registry."""

from __future__ import annotations

import pytest

from yaab import ArtifactManager, MemoryManager, SessionManager
from yaab.extensions import ComponentError, available, get, register
from yaab.types import Role


@pytest.mark.asyncio
async def test_session_manager_scoping_and_listing():
    mgr = SessionManager()
    s1 = await mgr.create_session(app_name="bank", user_id="alice", state={"tier": "gold"})
    await mgr.create_session(app_name="bank", user_id="alice")
    await mgr.append_text(s1.id, Role.USER, "hello")
    sessions = await mgr.list_sessions(app_name="bank", user_id="alice")
    assert len(sessions) == 2
    state = await mgr.get_state(s1.id)
    assert state["tier"] == "gold"


@pytest.mark.asyncio
async def test_memory_manager_namespacing_and_ingestion():
    mgr = MemoryManager()
    await mgr.add("Alice prefers email", app_name="bank", user_id="alice")
    await mgr.add("Bob prefers phone", app_name="bank", user_id="bob")
    hits = await mgr.search("how to contact", app_name="bank", user_id="alice", k=5)
    assert all(rec.metadata["user_id"] == "alice" for rec, _ in hits)


@pytest.mark.asyncio
async def test_memory_manager_add_session():
    from yaab.sessions.base import Session
    from yaab.types import Message

    session = Session()
    session.messages = [
        Message(role=Role.USER, content="What's my balance?"),
        Message(role=Role.ASSISTANT, content="Your balance is $100."),
    ]
    mgr = MemoryManager()
    records = await mgr.add_session_to_memory(session, app_name="bank", user_id="alice")
    assert len(records) == 2


@pytest.mark.asyncio
async def test_artifact_manager_versioning():
    mgr = ArtifactManager()
    v1 = await mgr.save("report.txt", b"v1 content", session_id="s1")
    v2 = await mgr.save("report.txt", b"v2 content", session_id="s1")
    assert (v1, v2) == (1, 2)
    assert await mgr.load("report.txt", session_id="s1") == b"v2 content"
    assert await mgr.load("report.txt", version=1, session_id="s1") == b"v1 content"
    assert await mgr.list_versions("report.txt", session_id="s1") == 2
    assert "report.txt" in await mgr.list_artifacts(session_id="s1")


def test_extension_registry_register_and_get():
    @register("model", "echo")
    def _make(**kw):
        return {"kind": "echo", **kw}

    assert "echo" in available("model")
    assert get("model", "echo", greeting="hi")["greeting"] == "hi"
    with pytest.raises(ComponentError):
        get("model", "does-not-exist")


def test_builtin_embedders_registered():
    assert "hashing" in available("embedder")
    assert "litellm" in available("embedder")
    embedder = get("embedder", "hashing")
    vec = embedder("hello world")
    assert isinstance(vec, list) and len(vec) > 0
