"""External vector-store adapters: Chroma and Qdrant.

Both satisfy the :class:`~yaab.rag.store.VectorStore` protocol and are registered
as ``vectorstore`` components, so they drop into a ``KnowledgeBase`` unchanged.
Their client libraries are imported lazily — install only what you use.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..extensions import register
from .store import Filter
from .types import Chunk, RetrievedChunk


class ChromaVectorStore:
    """Chroma-backed store (``pip install chromadb``).

    Uses a persistent or in-memory Chroma collection. Embeddings are supplied by
    YAAB (we pass precomputed vectors), so Chroma is used purely as the index.
    """

    def __init__(
        self,
        *,
        collection: str = "yaab",
        path: str | None = None,
        client: Any = None,
    ) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("chromadb is required. `pip install chromadb`.") from exc
        self._client = client or (
            chromadb.PersistentClient(path=path) if path else chromadb.Client()
        )
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> None:
        embedded = [c for c in chunks if c.embedding]
        if not embedded:
            return
        self._col.add(
            ids=[c.id for c in embedded],
            embeddings=[c.embedding for c in embedded],
            documents=[c.text for c in embedded],
            metadatas=[
                {
                    **c.metadata,
                    "_source": c.source or "",
                    "_doc": c.document_id or "",
                    "_idx": c.index,
                }
                for c in embedded
            ],
        )

    def query(
        self, embedding: list[float], *, k: int = 5, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        res = self._col.query(query_embeddings=[embedding], n_results=k, where=where or None)
        out: list[RetrievedChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            meta = dict(meta or {})
            chunk = Chunk(
                id=cid,
                text=text,
                source=meta.pop("_source", None) or None,
                document_id=meta.pop("_doc", None) or None,
                index=int(meta.pop("_idx", 0)),
                metadata=meta,
            )
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist)))
        return out

    def delete(self, *, where: Filter | None = None) -> int:
        before = self.count()
        self._col.delete(where=where or None)
        return before - self.count()

    def count(self) -> int:
        return self._col.count()


class QdrantVectorStore:
    """Qdrant-backed store (``pip install qdrant-client``)."""

    def __init__(
        self,
        *,
        collection: str = "yaab",
        url: str | None = None,
        location: str = ":memory:",
        dim: int = 1536,
        client: Any = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("qdrant-client is required. `pip install qdrant-client`.") from exc
        self._client = client or (QdrantClient(url=url) if url else QdrantClient(location=location))
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, chunks: list[Chunk]) -> None:
        from qdrant_client.models import PointStruct  # type: ignore

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=c.embedding,
                payload={
                    "text": c.text,
                    "source": c.source,
                    "document_id": c.document_id,
                    "index": c.index,
                    **c.metadata,
                },
            )
            for c in chunks
            if c.embedding
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def query(
        self, embedding: list[float], *, k: int = 5, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        flt = None
        if where:
            from qdrant_client.models import FieldCondition, MatchValue  # type: ignore
            from qdrant_client.models import Filter as QFilter

            flt = QFilter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=val))
                    for key, val in where.items()
                ]
            )
        hits = self._client.search(
            collection_name=self._collection, query_vector=embedding, limit=k, query_filter=flt
        )
        out: list[RetrievedChunk] = []
        for h in hits:
            p = dict(h.payload or {})
            chunk = Chunk(
                text=p.pop("text", ""),
                source=p.pop("source", None),
                document_id=p.pop("document_id", None),
                index=int(p.pop("index", 0)),
                metadata=p,
            )
            out.append(RetrievedChunk(chunk=chunk, score=float(h.score)))
        return out

    def delete(self, *, where: Filter | None = None) -> int:
        before = self.count()
        if where is None:
            self._client.delete_collection(self._collection)
            return before
        from qdrant_client.models import FieldCondition, FilterSelector, MatchValue  # type: ignore
        from qdrant_client.models import Filter as QFilter

        flt = QFilter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
        )
        self._client.delete(
            collection_name=self._collection, points_selector=FilterSelector(filter=flt)
        )
        return before - self.count()

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count


register("vectorstore", "chroma", lambda **kw: ChromaVectorStore(**kw))
register("vectorstore", "qdrant", lambda **kw: QdrantVectorStore(**kw))

__all__ = ["ChromaVectorStore", "QdrantVectorStore"]
