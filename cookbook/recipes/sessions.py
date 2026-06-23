"""Recipe: sessions — durable, multi-turn conversations.

Pass a ``session_id`` and the agent replays prior history automatically, so a
later turn remembers an earlier one.

    python -m cookbook.recipes.sessions
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect, resolve_model
from yaab import Agent, Runner
from yaab.models.test_model import TestModel
from yaab.sessions import InMemorySessionService


async def run() -> dict:
    # Offline: the model "remembers" by echoing a fixed answer; the point of the
    # recipe is that both turns share one session and the history grows.
    model = resolve_model(offline_default=TestModel(custom_output="Your name is Alice."))
    runner = Runner(session_service=InMemorySessionService())
    agent: Agent = Agent("assistant", model=model, instructions="Be friendly and concise.")

    sid = "conv-1"
    await runner.run(agent, "Hi, my name is Alice.", session_id=sid)
    second = await runner.run(agent, "What is my name?", session_id=sid)

    session = await runner.session_service.get(sid)
    history = len(session.messages)
    expect(history >= 4, f"expected >=4 messages persisted across two turns, got {history}")
    expect("alice" in str(second.output).lower(), "expected the assistant to recall the name")
    return {"answer": str(second.output), "messages_persisted": history}


if __name__ == "__main__":
    print(asyncio.run(run()))
