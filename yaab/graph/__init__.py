"""Durable graph orchestration."""

from __future__ import annotations

from .checkpoint import (
    Checkpointer,
    MemorySaver,
    PostgresSaver,
    RedisSaver,
    SQLiteSaver,
)
from .state import (
    END,
    START,
    Channel,
    CompiledGraph,
    GraphContext,
    GraphResult,
    RetryPolicy,
    StateGraph,
    interrupt,
)

__all__ = [
    "StateGraph",
    "CompiledGraph",
    "GraphResult",
    "GraphContext",
    "Channel",
    "RetryPolicy",
    "interrupt",
    "START",
    "END",
    "Checkpointer",
    "MemorySaver",
    "SQLiteSaver",
    "PostgresSaver",
    "RedisSaver",
]


def _register_checkpointers() -> None:
    """Register checkpointers as ``checkpointer`` components (selectable by name)."""
    from ..extensions import register

    register("checkpointer", "memory", lambda **kw: MemorySaver())
    register("checkpointer", "sqlite", lambda **kw: SQLiteSaver(**kw))
    register("checkpointer", "postgres", lambda **kw: PostgresSaver(**kw))
    register("checkpointer", "aurora", lambda **kw: PostgresSaver(**kw))
    register("checkpointer", "redis", lambda **kw: RedisSaver(**kw))


_register_checkpointers()
