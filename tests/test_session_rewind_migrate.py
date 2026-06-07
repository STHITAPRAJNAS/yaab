"""Session rewind (roll back to a prior turn) and migration (move backends).

Rewind truncates a conversation back to an earlier point — useful for "undo the
last exchange and try again" — keeping its structured state. Migration copies a
session (messages + state) from one backend into another, so you can move a
conversation across stores or schema versions without losing it.
"""

from __future__ import annotations

import pytest

from yaab.sessions.manager import SessionManager
from yaab.sessions.memory import InMemorySessionService
from yaab.types import Message, Role


async def _seed(mgr: SessionManager, sid: str) -> None:
    await mgr.create_session(session_id=sid)
    for i in range(1, 4):
        await mgr.append_message(sid, Message(role=Role.USER, content=f"q{i}"))
        await mgr.append_message(sid, Message(role=Role.ASSISTANT, content=f"a{i}"))


@pytest.mark.asyncio
async def test_rewind_keeps_first_n_turns():
    mgr = SessionManager()
    await _seed(mgr, "s1")
    # Keep the first 2 user turns (q1/a1, q2/a2); drop q3/a3.
    session = await mgr.rewind("s1", keep_turns=2)
    contents = [m.content for m in session.messages]
    assert contents == ["q1", "a1", "q2", "a2"]


@pytest.mark.asyncio
async def test_rewind_last_drops_recent_turns():
    mgr = SessionManager()
    await _seed(mgr, "s2")
    session = await mgr.rewind_last("s2", turns=1)  # drop the most recent turn
    contents = [m.content for m in session.messages]
    assert contents == ["q1", "a1", "q2", "a2"]


@pytest.mark.asyncio
async def test_rewind_preserves_state():
    mgr = SessionManager()
    await _seed(mgr, "s3")
    await mgr.update_state("s3", topic="refunds")
    session = await mgr.rewind("s3", keep_turns=1)
    assert session.state["topic"] == "refunds"
    assert [m.content for m in session.messages] == ["q1", "a1"]


@pytest.mark.asyncio
async def test_rewind_is_persisted():
    mgr = SessionManager()
    await _seed(mgr, "s4")
    await mgr.rewind("s4", keep_turns=1)
    reloaded = await mgr.get_session(session_id="s4")
    assert [m.content for m in reloaded.messages] == ["q1", "a1"]


@pytest.mark.asyncio
async def test_migrate_session_to_another_backend():
    src = SessionManager(InMemorySessionService())
    await _seed(src, "m1")
    await src.update_state("m1", k="v")

    dst_service = InMemorySessionService()
    dst = SessionManager(dst_service)
    await src.migrate_session("m1", to_service=dst_service)

    moved = await dst.get_session(session_id="m1")
    assert [m.content for m in moved.messages] == ["q1", "a1", "q2", "a2", "q3", "a3"]
    assert moved.state["k"] == "v"
