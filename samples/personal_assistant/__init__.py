"""Personal assistant — durable SQLite sessions + long-term memory + callbacks.

A realistic chat assistant that demonstrates, in one app, the state machinery you
need in production:

* **SQLite session service** — conversation history persists across process
  restarts, keyed by ``session_id``. Restart the program and the assistant still
  remembers what was said earlier in the same session.
* **Long-term memory service** — durable facts the assistant learns ("Alice
  prefers email") are stored in a vector memory and recalled across *different*
  sessions, folded into the system prompt automatically by the Runner.
* **Callbacks (Plugin hooks)** — a `MemoryWritebackPlugin` observes each run and
  writes salient user statements into long-term memory after the turn, and a
  `UsageLogPlugin` records token usage. This is how ADK-style
  before/after callbacks map onto YAAB.

Run it offline (deterministic) or against a real model:

    python -m samples.personal_assistant
    YAAB_SAMPLE_MODEL=ollama/llama3 python -m samples.personal_assistant
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from yaab import Agent, MemoryManager, Runner
from yaab.memory import InMemoryVectorMemory
from yaab.models.test_model import TestModel
from yaab.plugins import Plugin
from yaab.sessions import SQLiteSessionService

from .._common import resolve_model


class MemoryWritebackPlugin(Plugin):
    """After each run, persist the user's message into long-term memory.

    A real assistant would extract only *salient* facts (via an LLM or rules);
    here we store any user statement that looks like a preference/fact so the
    pattern is clear and deterministic.
    """

    name = "memory_writeback"

    def __init__(self, memory: MemoryManager, *, app_name: str, user_id: str) -> None:
        self.memory = memory
        self.app_name = app_name
        self.user_id = user_id
        self._last_user_msg: str | None = None

    async def on_user_message(self, ctx: Any, agent: str, message: Any) -> None:
        self._last_user_msg = message.content

    async def after_run(self, ctx: Any, agent: str, output: Any) -> None:
        text = self._last_user_msg or ""
        # Heuristic "is this a durable fact?" — keep the sample deterministic.
        keywords = ("my name is", "i prefer", "i like", "i live", "remember")
        if any(kw in text.lower() for kw in keywords):
            await self.memory.add(text, app_name=self.app_name, user_id=self.user_id)


class UsageLogPlugin(Plugin):
    """Record cumulative token usage across runs (an observability callback)."""

    name = "usage_log"

    def __init__(self) -> None:
        self.total_tokens = 0

    async def after_run(self, ctx: Any, agent: str, output: Any) -> None:
        self.total_tokens += ctx.usage.total_tokens


def build(model: Any = None, *, db_path: str | None = None, user_id: str = "alice"):
    """Build the assistant + a Runner wired with SQLite sessions, memory, callbacks.

    Returns ``(agent, runner, memory, usage_log)``.
    """
    db_path = db_path or os.path.join(tempfile.mkdtemp(), "assistant.db")

    # Long-term memory (durable facts across sessions). Swap InMemoryVectorMemory
    # for a vector-store-backed MemoryService in production.
    memory = MemoryManager(InMemoryVectorMemory())

    runner = Runner(
        session_service=SQLiteSessionService(db_path),  # durable conversation history
        memory_service=memory.service,  # recalled into the prompt
        plugins=[
            MemoryWritebackPlugin(memory, app_name="assistant", user_id=user_id),
            UsageLogPlugin(),
        ],
    )
    agent = Agent(
        "assistant",
        model=resolve_model(model, offline_default=TestModel("Got it — I'll remember that.")),
        instructions="You are a helpful personal assistant. Use any recalled memory about you.",
    )
    usage_log = next(p for p in runner.plugins if isinstance(p, UsageLogPlugin))
    return agent, runner, memory, usage_log


async def run(model: Any = None) -> dict[str, Any]:
    """Demonstrate session persistence + cross-session memory recall."""
    db_path = os.path.join(tempfile.mkdtemp(), "assistant.db")
    user = "alice"

    # --- Session 1: the user tells the assistant a durable fact -----------
    agent, runner, memory, usage = build(model, db_path=db_path, user_id=user)
    await runner.run(
        agent, "My name is Alice and I prefer email.", session_id="day1", identity=user
    )
    await runner.run(agent, "What's the weather?", session_id="day1", identity=user)

    # Same session remembers earlier turns (SQLite-persisted history).
    sess = await runner.session_service.get("day1")
    history_len = len(sess.messages)

    # The fact was written to long-term memory by the callback.
    learned = await memory.search(
        "how does the user want to be contacted", app_name="assistant", user_id=user, k=3
    )

    # --- Session 2 (a brand-new conversation, simulated restart) ----------
    # Rebuild from the same DB + memory: a new session has no shared history,
    # but long-term memory still recalls the fact learned in session 1.
    agent2, runner2, _, _ = build(model, db_path=db_path, user_id=user)
    runner2.memory_service = memory.service  # same durable memory store
    recall_hits = await runner2.memory_service.search("contact preference", k=3)

    return {
        "session1_history_messages": history_len,
        "learned_facts": [r.text for r, _ in learned],
        "recalled_in_session2": [r.text for r, _ in recall_hits],
        "total_tokens": usage.total_tokens,
    }
