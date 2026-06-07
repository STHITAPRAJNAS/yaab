"""KnowledgeBase — the entry point to RAG.

Ties together a chunker, an embedder, a vector store, and an optional reranker:
``add()`` ingests documents (chunk → embed → store); ``retrieve()`` embeds the
query, recalls candidates, optionally reranks, and returns scored chunks with
source attribution. Per-tenant isolation rides on a metadata ``where`` filter.

A KnowledgeBase exposes :meth:`as_tool`, so an agent can retrieve on demand, and
:meth:`augment`, so you can do classic "stuff the context" RAG.
"""

from __future__ import annotations

from typing import Any

from ..memory import Embedder, resolve_embedder
from .chunking import CharacterChunker, Chunker
from .rerank import Reranker
from .store import Filter, InMemoryVectorStore, VectorStore
from .types import Document, RetrievedChunk


class KnowledgeBase:
    """A ready-to-use RAG knowledge base over pluggable components."""

    def __init__(
        self,
        *,
        embedder: Embedder | str | None = None,
        store: VectorStore | None = None,
        chunker: Chunker | None = None,
        reranker: Reranker | None = None,
        context_guard: Any | None = None,
        min_score: float = 0.0,
        name: str = "knowledge",
        hybrid: bool = False,
    ) -> None:
        self.embedder = resolve_embedder(embedder)
        self.store = store or InMemoryVectorStore()
        self.chunker = chunker or CharacterChunker()
        self.reranker = reranker
        #: When set, recall combines a dense (embedding) and a sparse (BM25)
        #: ranking, fused by reciprocal rank — so exact rare terms surface even
        #: when the embedder generalizes them away. Off by default (dense-only).
        self.hybrid = hybrid
        self._bm25: Any | None = None
        self._chunks_by_id: dict[str, Any] = {}
        if hybrid:
            from .hybrid import BM25Index

            self._bm25 = BM25Index()
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
        runs don't duplicate context (incremental indexing).
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
                if self._bm25 is not None:
                    self._bm25.add(chunk.id, chunk.text)
                    self._chunks_by_id[chunk.id] = chunk
        self.store.add(all_chunks)
        return len(all_chunks)

    def add_text(
        self, text: str, *, source: str | None = None, metadata: dict | None = None
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
        where: Filter | None = None,
        rerank_top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top chunks for ``query`` (recall → optional rerank).

        With ``hybrid=True`` the dense (embedding) recall is fused with a sparse
        BM25 recall via reciprocal-rank fusion before any reranking, so a query's
        exact rare terms reliably surface their chunk.
        """
        embedding = self.embedder(query)
        # Over-fetch when reranking so the reranker has candidates to sharpen.
        fetch_k = max(k, rerank_top_n or 0, k * 2) if self.reranker else k
        if self._bm25 is not None:
            results = self._hybrid_recall(query, embedding, fetch_k, where)
        else:
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

    def _hybrid_recall(
        self, query: str, embedding: list[float], k: int, where: Filter | None
    ) -> list[RetrievedChunk]:
        """Fuse a dense and a sparse (BM25) recall by reciprocal rank."""
        from .hybrid import reciprocal_rank_fusion
        from .types import RetrievedChunk

        assert self._bm25 is not None
        # Over-fetch both arms so fusion has candidates from each signal.
        dense = self.store.query(embedding, k=k * 2, where=where)
        dense_ids = [r.chunk.id for r in dense]
        sparse_ids = [cid for cid, _ in self._bm25.search(query, k=k * 2)]
        # A where-filter applies to the sparse arm too: keep only chunks that the
        # dense (filtered) arm could also return, by intersecting on known chunks.
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
        by_id = {r.chunk.id: r for r in dense}
        out: list[RetrievedChunk] = []
        for cid, score in fused:
            if cid in by_id:
                out.append(RetrievedChunk(chunk=by_id[cid].chunk, score=score))
            elif cid in self._chunks_by_id and where is None:
                out.append(RetrievedChunk(chunk=self._chunks_by_id[cid], score=score))
            if len(out) >= k:
                break
        return out

    async def augment(
        self, query: str, *, k: int = 5, where: Filter | None = None
    ) -> tuple[str, list[RetrievedChunk]]:
        """Return a context block (with citations) plus the retrieved chunks.

        Use for classic context-stuffing RAG: prepend the block to a prompt.
        """
        results = await self.retrieve(query, k=k, where=where)
        block = "\n\n".join(f"[{r.citation()}] {r.text}" for r in results)
        return block, results

    def count(self) -> int:
        return self.store.count()

    # --- as a tool -----------------------------------------------------
    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        k: int = 5,
        scope_from_deps: str | None = None,
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
            where: Filter | None = None
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
