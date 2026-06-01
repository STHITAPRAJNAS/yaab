"""ParallelAgent must propagate session_id/identity to its children (Phase A3)."""

from __future__ import annotations

import pytest

from yaab import Agent, ParallelAgent
from yaab.models.test_model import FunctionModel


@pytest.mark.asyncio
async def test_parallel_agent_threads_session_id_to_children():
    seen: dict[str, object] = {}

    def make_model(key):
        def fn(messages):
            # Record how many messages the child saw (proxy for session replay).
            seen[key] = len(messages)
            return f"{key}-done"

        return FunctionModel(fn)

    a = Agent("a", model=make_model("a"))
    b = Agent("b", model=make_model("b"))

    from yaab import Runner
    from yaab.sessions.memory import InMemorySessionService
    from yaab.types import Message, Role

    sessions = InMemorySessionService()
    # Seed a session with prior history both children should replay.
    sid = "shared"
    await sessions.append(sid, Message(role=Role.USER, content="earlier turn"))

    # Children must use the shared session service + id. ParallelAgent builds its
    # children with their own runners, so the test verifies the kwarg is passed
    # through by giving each child a runner bound to the same session service.
    a._runner = Runner(session_service=sessions)
    b._runner = Runner(session_service=sessions)

    par = ParallelAgent("fan", [a, b])
    result = await par.run("q", session_id=sid)
    assert set(result.output.keys()) == {"a", "b"}
    # Both children replayed the seeded session history (saw >1 message).
    assert seen["a"] > 1 and seen["b"] > 1
