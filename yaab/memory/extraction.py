"""LLM-based memory extraction — distill durable memories from a conversation.

Raw transcripts make poor long-term memory: they are verbose, redundant, and
recall the *question* rather than the *fact*. :class:`MemoryExtractor` uses an
LLM to extract consolidated memories, backend-agnostically: one model call turns
a session's messages into a short JSON array of durable statements (facts,
preferences, decisions) that recall cleanly later.

The extractor only *produces* candidate memories; consolidation/dedup against
the existing store lives in :class:`yaab.memory.manager.MemoryManager`, which
owns the (app, user) namespace and the embedder.
"""

from __future__ import annotations

from typing import Any

from ..types import Message, Role

#: System prompt steering the model to emit ONLY a JSON array of short strings.
#: Kept strict because downstream parsing wants an array of memory statements;
#: the markdown-fence tolerance in the parser is a safety net, not a license to
#: wrap output. We ask for distilled facts (not Q&A) so recall returns knowledge.
_EXTRACTION_SYSTEM = """\
You distill durable long-term memories from a conversation.

Extract only information worth remembering across future sessions: stable user \
facts, preferences, decisions, goals, and important context. Ignore small talk, \
transient details, and anything specific to only this conversation.

Rewrite each memory as a short, self-contained third-person statement (e.g. \
"User prefers dark mode."), not a verbatim copy of a message and not a question.

Respond with ONLY a JSON array of strings, e.g. ["...", "..."]. If there is \
nothing worth remembering, respond with []. Do not add commentary or keys."""


class MemoryExtractor:
    """Distill durable memories from a session's messages via one LLM call.

    Parameters
    ----------
    model:
        Any :class:`~yaab.models.base.ModelProvider` (or a model-name string,
        resolved lazily via ``yaab.models.resolve_model``). Used once per
        :meth:`extract` to produce the JSON array of memory statements.
    roles:
        Which message roles to include in the transcript handed to the model.
        Defaults to user + assistant turns (system/tool noise is dropped).
    max_memories:
        Safety cap on the number of memories returned, so a misbehaving model
        cannot flood the store. ``None`` disables the cap.
    """

    def __init__(
        self,
        model: Any,
        *,
        roles: tuple[Role, ...] = (Role.USER, Role.ASSISTANT),
        max_memories: int | None = 20,
    ) -> None:
        # Resolve strings to a provider lazily so importing this module never
        # pulls in litellm; instances pass through unchanged.
        if isinstance(model, str):
            from ..models import resolve_model

            model = resolve_model(model)
        self.model = model
        self.roles = roles
        self.max_memories = max_memories

    async def extract(self, messages: list[Message]) -> list[str]:
        """Return a list of distilled memory statements for ``messages``.

        Makes a single model call and tolerantly parses the response (markdown
        fences allowed). Non-string items and a non-list payload are dropped so
        a malformed response degrades to ``[]`` rather than raising — ingestion
        should never crash a run.
        """
        transcript = self._render(messages)
        if not transcript.strip():
            return []
        prompt = [
            Message(role=Role.SYSTEM, content=_EXTRACTION_SYSTEM),
            Message(
                role=Role.USER,
                content=f"Conversation:\n{transcript}\n\nReturn the JSON array of memories.",
            ),
        ]
        response = await self.model.complete(prompt)
        return self._parse(response.content)

    def _render(self, messages: list[Message]) -> str:
        """Flatten the selected turns into a compact, role-labeled transcript."""
        lines = [
            f"{msg.role.value}: {msg.content}"
            for msg in messages
            if msg.role in self.roles and msg.content
        ]
        return "\n".join(lines)

    def _parse(self, content: str) -> list[str]:
        """Tolerantly parse the model's JSON array of memory strings.

        Reuses :func:`yaab.streaming.parse_partial_json` (lazy import) so a
        markdown-fenced or slightly-truncated response still yields the array.
        """
        from ..streaming import parse_partial_json

        parsed = parse_partial_json(content)
        if not isinstance(parsed, list):
            return []
        memories = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
        if self.max_memories is not None:
            memories = memories[: self.max_memories]
        return memories


__all__ = ["MemoryExtractor"]
