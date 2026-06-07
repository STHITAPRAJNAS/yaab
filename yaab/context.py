"""Context-window management — keep conversations within the model's budget.

Long conversations silently overflow a model's context window. A
:class:`ContextStrategy` decides how to trim the message list before each model
call. Three strategies ship:

* :class:`KeepAll` — no-op (the default; matches prior behavior);
* :class:`TruncateMessages` — keep the system message(s) + the most recent N
  messages (cheap, deterministic, no extra model calls);
* :class:`SummarizeHistory` — when the conversation exceeds a token budget,
  fold the oldest messages into a running summary via a model, preserving the
  system prompt and the most recent turns.

Token counting is approximate by default (chars/4) and pluggable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .types import Message, Role

# An estimator maps a message list to an approximate token count.
TokenCounter = Callable[[list[Message]], int]


def approx_tokens(messages: list[Message]) -> int:
    """Rough token estimate: ~4 characters per token."""
    chars = sum(len(m.content or "") for m in messages)
    return chars // 4


@runtime_checkable
class ContextStrategy(Protocol):
    async def apply(self, messages: list[Message], *, model: Any = None) -> list[Message]: ...


class KeepAll:
    """Pass messages through unchanged (default)."""

    async def apply(self, messages: list[Message], *, model: Any = None) -> list[Message]:
        return messages


def _split_system(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    system = [m for m in messages if m.role is Role.SYSTEM]
    rest = [m for m in messages if m.role is not Role.SYSTEM]
    return system, rest


class TruncateMessages:
    """Keep the system message(s) plus the most recent ``max_messages`` turns."""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages

    async def apply(self, messages: list[Message], *, model: Any = None) -> list[Message]:
        system, rest = _split_system(messages)
        if len(rest) <= self.max_messages:
            return messages
        return system + rest[-self.max_messages :]


class SummarizeHistory:
    """Summarize the oldest history once a token budget is exceeded.

    Keeps system message(s) and the last ``keep_recent`` messages verbatim;
    everything older is condensed into a single system "conversation summary"
    message by ``model``. Only triggers above ``max_tokens`` so short
    conversations pay nothing.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 6000,
        keep_recent: int = 6,
        token_counter: TokenCounter | None = None,
        model: Any = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.token_counter = token_counter or approx_tokens
        self.model = model

    async def apply(self, messages: list[Message], *, model: Any = None) -> list[Message]:
        if self.token_counter(messages) <= self.max_tokens:
            return messages
        model = self.model or model
        if model is None:
            # No model to summarize with — fall back to truncation.
            return await TruncateMessages(self.keep_recent).apply(messages)

        system, rest = _split_system(messages)
        if len(rest) <= self.keep_recent:
            return messages
        to_summarize = rest[: -self.keep_recent]
        recent = rest[-self.keep_recent :]

        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in to_summarize)
        prompt = (
            "Summarize the following conversation excerpt concisely, preserving "
            "facts, decisions, and any open questions:\n\n" + transcript
        )
        resp = await model.complete([Message(role=Role.USER, content=prompt)])
        summary = Message(
            role=Role.SYSTEM, content=f"Summary of earlier conversation:\n{resp.content}"
        )
        return system + [summary] + recent


#: Common words that carry no topical signal, excluded from overlap scoring.
_STOPWORDS = frozenset(
    "the a an and or but is are was were be been being to of in on at for with "
    "how what when where why who which that this these those it its as by from "
    "do does did can could would should will i you he she they we me my your".split()
)


def _content_words(text: str) -> set[str]:
    return {w.strip(".,!?;:'\"") for w in text.lower().split()} - _STOPWORDS - {""}


def _keyword_overlap(query: str, text: str) -> float:
    """Default relevance score: overlap of topical (non-stopword) word sets."""
    q = _content_words(query)
    t = _content_words(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


class RelevanceFilter:
    """Drop prior turns that aren't relevant to the latest user message.

    Where :class:`TruncateMessages` drops by age and :class:`SummarizeHistory`
    compresses, this keeps only history that earns its place: each non-system,
    non-latest message is scored against the latest user message, and anything
    below ``min_score`` is dropped. The system message(s) and the latest user
    message are *always* kept — you cannot drop the question you must answer.

    ``scorer(query, text) -> float`` is injectable (defaults to keyword overlap),
    so the filter runs offline; pass an embedding-similarity scorer for semantic
    relevance.
    """

    def __init__(
        self,
        *,
        min_score: float = 0.15,
        scorer: Callable[[str, str], float] | None = None,
    ) -> None:
        self.min_score = min_score
        self.scorer = scorer or _keyword_overlap

    async def apply(self, messages: list[Message], *, model: Any = None) -> list[Message]:
        system, rest = _split_system(messages)
        if not rest:
            return messages
        # The latest user message is the relevance anchor and is always kept.
        last_user_idx = next(
            (i for i in range(len(rest) - 1, -1, -1) if rest[i].role is Role.USER), None
        )
        if last_user_idx is None:
            return messages
        query = str(rest[last_user_idx].content)
        kept: list[Message] = []
        for i, msg in enumerate(rest):
            if i == last_user_idx or self.scorer(query, str(msg.content)) >= self.min_score:
                kept.append(msg)
        return system + kept


__all__ = [
    "ContextStrategy",
    "KeepAll",
    "TruncateMessages",
    "SummarizeHistory",
    "RelevanceFilter",
    "approx_tokens",
]
