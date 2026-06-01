"""Memory manager — scoped long-term memory with session ingestion.

Adds ``(app_name, user_id)`` namespacing on top of a :class:`MemoryService`,
plus :meth:`add_session_to_memory` which ingests a finished conversation into
long-term memory so future runs can recall it — the session-to-memory
consolidation step, made backend-agnostic.
"""

from __future__ import annotations

from typing import Any

from .. import _core
from ..types import Role
from . import InMemoryVectorMemory, MemoryRecord, MemoryService, resolve_embedder

DEFAULT_APP = "default"
DEFAULT_USER = "default"

#: Cosine-similarity threshold above which an extracted memory is treated as a
#: near-duplicate of an existing one and skipped (consolidation/dedup). High by
#: default so only genuine restatements collapse.
DEFAULT_DEDUP_THRESHOLD = 0.92


class MemoryManager:
    """Manage long-term memory scoped by app and user."""

    def __init__(
        self,
        service: MemoryService | None = None,
        *,
        embedder: Any | None = None,
    ) -> None:
        # ``embedder`` (callable or LiteLLM model-name string) is a convenience for
        # the default in-memory backend; ignored when an explicit service is given.
        self.service = service or InMemoryVectorMemory(embedder=embedder)
        # Embedder used for consolidation/dedup of extracted memories. Reuse the
        # service's embedder when it exposes one (so similarity is computed in the
        # same space the store retrieves in); otherwise resolve the given/default
        # embedder. Construction is cheap and offline.
        self._embedder = getattr(self.service, "embedder", None) or resolve_embedder(embedder)

    @staticmethod
    def _ns(app_name: str, user_id: str) -> dict[str, str]:
        return {"app_name": app_name, "user_id": user_id}

    async def add(
        self,
        text: str,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        meta = {**self._ns(app_name, user_id), **(metadata or {})}
        return await self.service.add(text, metadata=meta)

    async def search(
        self,
        query: str,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        k: int = 5,
    ) -> list[tuple[MemoryRecord, float]]:
        # Over-fetch, then filter to the requested namespace.
        hits = await self.service.search(query, k=k * 4)
        scoped = [
            (rec, score)
            for rec, score in hits
            if rec.metadata.get("app_name", app_name) == app_name
            and rec.metadata.get("user_id", user_id) == user_id
        ]
        return scoped[:k]

    async def add_session_to_memory(
        self,
        session: Any,
        *,
        app_name: str = DEFAULT_APP,
        user_id: str = DEFAULT_USER,
        roles: tuple[Role, ...] = (Role.USER, Role.ASSISTANT),
        extract: bool = False,
        extractor: Any | None = None,
        model: Any | None = None,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    ) -> list[MemoryRecord]:
        """Ingest a session's messages into long-term memory.

        By default (``extract=False`` and no ``extractor``) this preserves the
        original behavior: raw user/assistant lines are copied verbatim into the
        store — kept for backward compatibility.

        When ``extract=True`` (or an ``extractor`` is supplied), the messages are
        instead distilled by a :class:`~yaab.memory.extraction.MemoryExtractor`
        into a handful of durable memory statements (the extraction
        workflow). Each candidate is then *consolidated*: if an existing memory in
        the same ``(app_name, user_id)`` namespace is within ``dedup_threshold``
        cosine similarity, it is skipped — so re-ingesting restated facts does not
        bloat the store.

        ``extractor`` takes precedence; otherwise an extractor is built from
        ``model`` (a ``ModelProvider`` or model-name string).
        """
        messages = getattr(session, "messages", [])

        if extractor is not None or extract:
            extractor = extractor or self._build_extractor(model, roles)
            return await self._ingest_extracted(
                extractor,
                messages,
                session=session,
                app_name=app_name,
                user_id=user_id,
                dedup_threshold=dedup_threshold,
            )

        records: list[MemoryRecord] = []
        for msg in messages:
            if msg.role in roles and msg.content:
                records.append(
                    await self.add(
                        msg.content,
                        app_name=app_name,
                        user_id=user_id,
                        metadata={"session_id": session.id, "role": msg.role.value},
                    )
                )
        return records

    @staticmethod
    def _build_extractor(model: Any | None, roles: tuple[Role, ...]) -> Any:
        # Lazy import keeps the extractor (and any model deps) out of the hot path
        # for the common raw-copy case.
        from .extraction import MemoryExtractor

        if model is None:
            raise ValueError(
                "extract=True requires a `model` (or pass an `extractor=`). "
                "Give a ModelProvider or model-name string."
            )
        return MemoryExtractor(model, roles=roles)

    async def _ingest_extracted(
        self,
        extractor: Any,
        messages: list[Any],
        *,
        session: Any,
        app_name: str,
        user_id: str,
        dedup_threshold: float,
    ) -> list[MemoryRecord]:
        """Distill memories, consolidate against the store, and persist the rest."""
        memories = await extractor.extract(messages)
        stored: list[MemoryRecord] = []
        for text in memories:
            if await self._is_duplicate(
                text, app_name=app_name, user_id=user_id, threshold=dedup_threshold
            ):
                continue
            record = await self.add(
                text,
                app_name=app_name,
                user_id=user_id,
                metadata={"session_id": session.id, "extracted": True},
            )
            stored.append(record)
        return stored

    async def _is_duplicate(
        self, text: str, *, app_name: str, user_id: str, threshold: float
    ) -> bool:
        """True when an existing memory in the namespace is ~identical to ``text``.

        Compares the candidate's embedding against the embeddings of the nearest
        existing memories (same app/user) and treats anything at or above
        ``threshold`` cosine similarity as a restatement to skip.
        """
        existing = await self.search(text, app_name=app_name, user_id=user_id, k=5)
        if not existing:
            return False
        query_vec = self._embedder(text)
        for rec, _score in existing:
            # Prefer the stored embedding; fall back to re-embedding the text so
            # backends that don't surface embeddings still dedup correctly.
            other_vec = rec.embedding or self._embedder(rec.text)
            if _core.cosine_similarity(query_vec, other_vec) >= threshold:
                return True
        return False


__all__ = ["MemoryManager"]
