"""KnowledgeBase — the one object that makes RAG built-in, not bolted on.

Ties together a chunker, an embedder, a vector store, and an optional reranker:
``add()`` ingests documents (chunk → embed → store); ``retrieve()`` embeds the
query, recalls candidates, optionally reranks, and returns scored chunks with
source attribution. Per-tenant isolation rides on a metadata ``where`` filter.

A KnowledgeBase exposes :meth:`as_tool`, so an agent can retrieve on demand, and
:meth:`augment`, so you can do classic "stuff the context" RAG.
"""

from __future__ import annotations

from typing import Any, Optional

from ..memory import Embedder, hashing_embedder
from .chunking import Chunker, CharacterChunker
from .rerank import Reranker
from .store import Filter, InMemoryVectorStore, VectorStore
from .types import Document, RetrievedChunk


class KnowledgeBase:
    """A ready-to-use RAG knowledge base over pluggable components."""

    def __init__(
        self,
        *,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        chunker: Optional[Chunker] = None,
        reranker: Optional[Reranker] = None,
        context_guard: Optional[Any] = None,
        min_score: float = 0.0,
        name: str = "knowledge",
    ) -> None:
        self.embedder = embedder or hashing_embedder()
        self.store = store or InMemoryVectorStore()
        self.chunker = chunker or CharacterChunker()
        self.reranker = reranker
        #: Retrieval guardrail — a callable ``(RetrievedChunk) -> bool`` that
        #: drops a chunk when it returns False (context-poisoning / leakage
        #: defense applied *before* context reaches the model).
        self.context_guard = context_guard
        #: Drop chunks scoring below this threshold (filter weak/off-topic recall).
        self.min_score = min_score
        self.name = name
        self._seen: set[str] = set()  # content hashes, for incremental dedup

    # --- ingestion -----------------------------------------------------
    def add(self, documents: list[Document] | Document, *, dedup: bool = True) -> int:
        """Chunk, embed, and store documents. Returns the chunk count added.

        With ``dedup`` (default), chunks whose content was already indexed are
        skipped — re-ingesting an unchanged corpus is a cheap no-op and repeated
        runs don't duplicate context (LangChain RecordManager-style incremental
        indexing).
        """
        import hashlib

        docs = [documents] if isinstance(documents, Document) else documents
        all_chunks = []
        for doc in docs:
            for chunk in self.chunker.split(doc):
                if dedup:
                    key = hashlib.sha256(chunk.text.encode()).hexdigest()
                    if key in self._seen:
                        continue
                    self._seen.add(key)
                chunk.embedding = self.embedder(chunk.text)
                all_chunks.append(chunk)
        self.store.add(all_chunks)
        return len(all_chunks)

    def add_text(
        self, text: str, *, source: Optional[str] = None, metadata: Optional[dict] = None
    ) -> int:
        return self.add(Document(text=text, source=source, metadata=metadata or {}))

    def delete(self, *, source: str) -> int:
        """Remove all chunks originating from ``source``. Returns count removed."""
        return self.store.delete(where={"source": source})

    def reindex(self, documents: list[Document] | Document, *, source: str) -> int:
        """Replace a source's chunks with freshly-ingested ones (incremental update)."""
        self.delete(source=source)
        return self.add(documents, dedup=False)

    # --- retrieval -----------------------------------------------------
    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        where: Optional[Filter] = None,
        rerank_top_n: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top chunks for ``query`` (recall → optional rerank)."""
        embedding = self.embedder(query)
        # Over-fetch when reranking so the reranker has candidates to sharpen.
        fetch_k = max(k, rerank_top_n or 0, k * 2) if self.reranker else k
        results = self.store.query(embedding, k=fetch_k, where=where)
        if self.reranker is not None:
            top_n = rerank_top_n or k
            if hasattr(self.reranker, "arerank"):
                results = await self.reranker.arerank(query, results, top_n=top_n)
            else:
                results = self.reranker.rerank(query, results, top_n=top_n)
        # Retrieval guardrails: drop weak scores and anything the guard rejects.
        if self.min_score > 0.0:
            results = [r for r in results if r.score >= self.min_score]
        if self.context_guard is not None:
            results = [r for r in results if self.context_guard(r)]
        return results[:k]

    async def augment(
        self, query: str, *, k: int = 5, where: Optional[Filter] = None
    ) -> tuple[str, list[RetrievedChunk]]:
        """Return a context block (with citations) plus the retrieved chunks.

        Use for classic context-stuffing RAG: prepend the block to a prompt.
        """
        results = await self.retrieve(query, k=k, where=where)
        block = "\n\n".join(
            f"[{r.citation()}] {r.text}" for r in results
        )
        return block, results

    def count(self) -> int:
        return self.store.count()

    # --- as a tool -----------------------------------------------------
    def as_tool(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        k: int = 5,
        scope_from_deps: Optional[str] = None,
    ) -> Any:
        """Expose retrieval as an agent tool.

        ``scope_from_deps``, if set, names a field read from ``ctx.deps`` and
        used as a metadata filter value (keyed by the same name) — so an agent
        run for user "alice" only retrieves alice's documents.
        """
        from ..tools.base import FunctionTool
        from ..types import RunContext

        kb = self

        async def search_knowledge(ctx: RunContext, query: str) -> str:
            where: Optional[Filter] = None
            if scope_from_deps is not None and ctx.deps is not None:
                value = getattr(ctx.deps, scope_from_deps, None)
                if value is not None:
                    where = {scope_from_deps: value}
            results = await kb.retrieve(query, k=k, where=where)
            if not results:
                return "No relevant information found."
            return "\n\n".join(f"[{r.citation()}] {r.text}" for r in results)

        return FunctionTool(
            search_knowledge,
            name=name or f"search_{self.name}",
            description=description
            or f"Search the {self.name} knowledge base for relevant information.",
        )


__all__ = ["KnowledgeBase"]
