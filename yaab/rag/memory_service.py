"""Durable, vector-store-backed long-term memory.

The only built-in :class:`~yaab.memory.MemoryService` is
``InMemoryVectorMemory`` — process-local, lost on restart.
:class:`KnowledgeBaseMemory` adds durability, backend-agnostically: it
implements the ``MemoryService`` protocol on top of a
:class:`~yaab.rag.knowledge.KnowledgeBase`, so memory persists in *any* of the
eight vector-store backends (pgvector, Chroma, Qdrant, Pinecone, Weaviate,
OpenSearch, Oracle, Aurora) the KnowledgeBase already supports.

It lives in ``yaab.rag`` (not ``yaab.memory``) on purpose: ``yaab.rag`` already
imports ``yaab.memory`` (for the embedder), so putting it here avoids a circular
import.
"""

from __future__ import annotations

from typing import Any

from ..memory import MemoryRecord
from .knowledge import KnowledgeBase
from .types import Document


class KnowledgeBaseMemory:
    """A :class:`~yaab.memory.MemoryService` backed by a :class:`KnowledgeBase`.

    ``add`` ingests a memory as a single-chunk document (no splitting — a memory
    statement is already atomic); ``search`` retrieves the most similar memories
    and adapts the RAG ``RetrievedChunk`` results back into ``MemoryRecord`` /
    score tuples so it is a drop-in for ``InMemoryVectorMemory``.

    ``search`` also accepts ``app_name`` / ``user_id`` as named parameters so the
    :class:`~yaab.runner.Runner` (which *inspects the signature* to thread the
    run's identity/app scope) and :class:`~yaab.memory.manager.MemoryManager`
    can scope retrieval — namespace filtering pushes down to the store's metadata
    ``where`` filter, keeping per-user/app isolation cheap even at scale.
    """

    def __init__(self, knowledge_base: KnowledgeBase | None = None, **kb_kwargs: Any) -> None:
        # Accept a ready KnowledgeBase, or build one from passthrough kwargs
        # (embedder=, store=, ...) for the common one-liner construction.
        self.kb = knowledge_base or KnowledgeBase(**kb_kwargs)

    async def add(self, text: str, *, metadata: dict | None = None) -> MemoryRecord:
        """Store a memory statement. Returns the corresponding ``MemoryRecord``.

        The memory is indexed as a one-chunk document so it survives wherever the
        KnowledgeBase's vector store lives. ``dedup=False`` because consolidation
        is the caller's job (``MemoryManager``); we never silently drop a write.
        """
        meta = dict(metadata or {})
        doc = Document(text=text, metadata=meta)
        self.kb.add(doc, dedup=False)
        # Mirror the KnowledgeBase record back as a MemoryRecord so callers get
        # the same shape as InMemoryVectorMemory (id + text + embedding + meta).
        return MemoryRecord(id=doc.id, text=text, embedding=self.kb.embedder(text), metadata=meta)

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        app_name: str | None = None,
        user_id: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Retrieve up to ``k`` memories most similar to ``query``.

        ``app_name`` / ``user_id`` (when given) become a metadata filter so only
        memories in that namespace are returned. Results are adapted from RAG
        ``RetrievedChunk``s into ``(MemoryRecord, score)`` tuples.
        """
        where: dict[str, Any] = {}
        if app_name is not None:
            where["app_name"] = app_name
        if user_id is not None:
            where["user_id"] = user_id
        results = await self.kb.retrieve(query, k=k, where=where or None)
        out: list[tuple[MemoryRecord, float]] = []
        for r in results:
            chunk = r.chunk
            record = MemoryRecord(
                id=chunk.id,
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
            )
            out.append((record, r.score))
        return out


__all__ = ["KnowledgeBaseMemory"]
