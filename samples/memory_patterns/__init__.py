"""Memory patterns — episodic (in-session) vs long-term (cross-session) memory.

Agents have two distinct kinds of memory, and YAAB models them with two distinct
services. This sample shows both, end to end, and how you move information from
one to the other (the session-to-memory consolidation step):

* **Episodic memory** — the turns of *the current conversation*. Held by the
  ``SessionService`` (here SQLite-backed, so it survives a restart) and keyed by
  ``session_id``. It is naturally short-lived and scoped to one conversation: a
  brand-new session starts with an empty history.
* **Long-term memory** — durable facts the assistant should remember *across*
  conversations. Held by the ``MemoryService`` / :class:`MemoryManager`, scoped
  by ``(app_name, user_id)`` and retrieved by semantic search. It is not tied to
  any single session.

The bridge between them is :meth:`MemoryManager.add_session_to_memory`: when a
conversation ends, you consolidate its salient turns from episodic storage into
long-term memory so a *future* session can recall them.

    python -m samples.memory_patterns
    YAAB_SAMPLE_MODEL=ollama/llama3 python -m samples.memory_patterns
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from yaab import Agent, MemoryManager, Runner
from yaab.memory import InMemoryVectorMemory
from yaab.models.test_model import TestModel
from yaab.sessions import SQLiteSessionService
from yaab.types import Role

from .._common import resolve_model

APP = "travel_concierge"


def build(model: Any = None, *, db_path: str | None = None):
    """Build a concierge agent + a Runner with SQLite episodic sessions.

    Returns ``(agent, runner, memory)`` where ``memory`` is the long-term store.
    """
    db_path = db_path or os.path.join(tempfile.mkdtemp(), "memory.db")
    memory = MemoryManager(InMemoryVectorMemory())  # swap for a vector-store backend
    runner = Runner(
        session_service=SQLiteSessionService(db_path),  # episodic memory (per session)
        memory_service=memory.service,  # long-term recall into prompt
    )
    agent = Agent(
        "concierge",
        model=resolve_model(model, offline_default=TestModel("Noted — happy to help plan that.")),
        instructions="You are a travel concierge. Use recalled memory about the traveler.",
    )
    return agent, runner, memory


async def run(model: Any = None, *, user_id: str = "sam") -> dict[str, Any]:
    """Walk through episodic vs long-term memory and the consolidation step."""
    db_path = os.path.join(tempfile.mkdtemp(), "memory.db")
    agent, runner, memory = build(model, db_path=db_path)

    # --- A conversation (episode): several turns in ONE session ----------
    trip = "trip-tokyo-2026"
    for turn in (
        "I'm planning a trip to Tokyo in April.",
        "I'm vegetarian, so I need veggie-friendly spots.",
        "I prefer boutique hotels over big chains.",
    ):
        await runner.run(agent, turn, session_id=trip, identity=user_id)

    # Episodic memory == this session's stored turns (short-term, per-conversation).
    episode = await runner.session_service.get(trip)
    episodic_turns = len(episode.messages)

    # --- Consolidate the episode into long-term memory -------------------
    # When the conversation is "done", lift its turns into durable, cross-session
    # memory scoped to this traveler. This is the episodic -> long-term bridge.
    # We keep only the traveler's own statements (USER turns) — those are the
    # facts worth remembering, not the assistant's acknowledgments.
    consolidated = await memory.add_session_to_memory(
        episode, app_name=APP, user_id=user_id, roles=(Role.USER,)
    )

    # --- A NEW conversation, days later (simulated restart) ---------------
    # A fresh session has NO episodic history of the Tokyo trip...
    followup = "trip-followup"
    fresh = await runner.session_service.get_or_create(followup)
    new_session_episodic_turns = len(fresh.messages)  # 0 — episodic memory is per-session

    # ...but long-term memory still recalls the traveler's preferences.
    dietary = await memory.search(
        "what are the traveler's food preferences", app_name=APP, user_id=user_id, k=3
    )
    lodging = await memory.search(
        "what kind of hotels does the traveler like", app_name=APP, user_id=user_id, k=3
    )

    # Scoping check: a *different* user shares neither episodic nor long-term memory.
    other_user = await memory.search("food preferences", app_name=APP, user_id="someone_else", k=3)

    return {
        "episodic_turns_in_trip_session": episodic_turns,
        "new_session_episodic_turns": new_session_episodic_turns,
        "consolidated_to_long_term": len(consolidated),
        "recalled_dietary": [r.text for r, _ in dietary],
        "recalled_lodging": [r.text for r, _ in lodging],
        "other_user_recall_count": len(other_user),
    }
