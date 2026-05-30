"""Vector store abstraction for RAG.

A :class:`VectorStore` holds embedded :class:`Chunk`s and returns the most
similar ones for a query embedding, with optional metadata filtering (the basis
for per-user / per-tenant knowledge isolation and document-level access
control). Retrieval uses the Rust-accelerated :func:`yaab._core.top_k` (with a
pure-Python fallback).

Ships an in-memory store (default/dev) and a pgvector-backed store
(``PgVectorStore``, lazy psycopg) registered in the component registry so third
parties can add Chroma/Qdrant/Pinecone/etc. behind the same protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .. import _core
from ..extensions import register
from .types import Chunk, RetrievedChunk

# A metadata filter is an exact-match dict applied to chunk.metadata.
Filter = dict[str, Any]


def _matches(metadata: dict[str, Any], where: Filter | None) -> bool:
    if not where:
        return True
    return all(metadata.get(k) == v for k, v in where.items())


@runtime_checkable
class VectorStore(Protocol):
    """Pluggable vector storage + similarity search."""

    def add(self, chunks: list[Chunk]) -> None: ...

    def query(
        self, embedding: list[float], *, k: int = 5, where: Filter | None = None
    ) -> list[RetrievedChunk]: ...

    def delete(self, *, where: Filter | None = None) -> int: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    """Process-local vector store over the Rust top-k similarity op."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(c for c in chunks if c.embedding)

    def query(
        self, embedding: list[float], *, k: int = 5, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        candidates = [c for c in self._chunks if _matches(c.metadata, where)]
        if not candidates:
            return []
        matrix = [c.embedding for c in candidates]
        hits = _core.top_k(embedding, matrix, k)
        return [RetrievedChunk(chunk=candidates[i], score=score) for i, score in hits]

    def delete(self, *, where: Filter | None = None) -> int:
        if where is None:
            n = len(self._chunks)
            self._chunks.clear()
            return n
        keep = [c for c in self._chunks if not _matches(c.metadata, where)]
        removed = len(self._chunks) - len(keep)
        self._chunks = keep
        return removed

    def count(self) -> int:
        return len(self._chunks)


class PgVectorStore:
    """pgvector-backed store (psycopg v3, imported lazily).

    Stores chunks in a table with a ``vector`` column and a JSONB ``metadata``
    column; similarity uses pgvector's ``<=>`` cosine-distance operator. Metadata
    filters are applied as JSONB containment so per-tenant isolation pushes down
    to the database.
    """

    def __init__(self, dsn: str, *, table: str = "yaab_chunks", dim: int = 1536) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "psycopg is required for PgVectorStore. `pip install 'yaab[postgres]'` "
                "and enable the pgvector extension in your database."
            ) from exc
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        self._dim = dim
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"id TEXT PRIMARY KEY, text TEXT, document_id TEXT, source TEXT, "
            f"idx INTEGER, embedding vector({dim}), metadata JSONB)"
        )

    def add(self, chunks: list[Chunk]) -> None:
        import json

        with self._conn.cursor() as cur:
            for c in chunks:
                if not c.embedding:
                    continue
                cur.execute(
                    f"INSERT INTO {self._table} "
                    f"(id, text, document_id, source, idx, embedding, metadata) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (
                        c.id,
                        c.text,
                        c.document_id,
                        c.source,
                        c.index,
                        str(c.embedding),
                        json.dumps(c.metadata),
                    ),
                )

    def query(
        self, embedding: list[float], *, k: int = 5, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        import json

        sql = (
            f"SELECT id, text, document_id, source, idx, metadata, "
            f"1 - (embedding <=> %s) AS score FROM {self._table}"
        )
        params: list[Any] = [str(embedding)]
        if where:
            sql += " WHERE metadata @> %s"
            params.append(json.dumps(where))
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.extend([str(embedding), k])
        rows = self._conn.execute(sql, params).fetchall()
        out: list[RetrievedChunk] = []
        for rid, text, doc_id, source, idx, metadata, score in rows:
            chunk = Chunk(
                id=rid,
                text=text,
                document_id=doc_id,
                source=source,
                index=idx or 0,
                metadata=metadata or {},
            )
            out.append(RetrievedChunk(chunk=chunk, score=float(score)))
        return out

    def delete(self, *, where: Filter | None = None) -> int:
        import json

        if where is None:
            cur = self._conn.execute(f"DELETE FROM {self._table}")
            return cur.rowcount
        cur = self._conn.execute(
            f"DELETE FROM {self._table} WHERE metadata @> %s", (json.dumps(where),)
        )
        return cur.rowcount

    def count(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]


register("vectorstore", "memory", lambda **kw: InMemoryVectorStore())
register("vectorstore", "pgvector", lambda **kw: PgVectorStore(**kw))


# Register external store names eagerly (factories lazy-import the client libs),
# so they appear in available("vectorstore") without importing chromadb/qdrant.
def _make_chroma(**kw: Any) -> Any:
    from .stores_external import ChromaVectorStore

    return ChromaVectorStore(**kw)


def _make_qdrant(**kw: Any) -> Any:
    from .stores_external import QdrantVectorStore

    return QdrantVectorStore(**kw)


def _make_opensearch(**kw: Any) -> Any:
    from .stores_external import OpenSearchVectorStore

    return OpenSearchVectorStore(**kw)


def _make_oracle(**kw: Any) -> Any:
    from .stores_external import OracleVectorStore

    return OracleVectorStore(**kw)


def _make_pinecone(**kw: Any) -> Any:
    from .stores_external import PineconeVectorStore

    return PineconeVectorStore(**kw)


def _make_weaviate(**kw: Any) -> Any:
    from .stores_external import WeaviateVectorStore

    return WeaviateVectorStore(**kw)


register("vectorstore", "chroma", _make_chroma)
register("vectorstore", "qdrant", _make_qdrant)
register("vectorstore", "opensearch", _make_opensearch)
register("vectorstore", "oracle", _make_oracle)
register("vectorstore", "pinecone", _make_pinecone)
register("vectorstore", "weaviate", _make_weaviate)
# Aurora PostgreSQL (and any Postgres with the pgvector extension) is served by
# the pgvector store — the connection string just points at the Aurora endpoint.
register("vectorstore", "aurora", lambda **kw: PgVectorStore(**kw))

__all__ = ["VectorStore", "InMemoryVectorStore", "PgVectorStore", "Filter"]
