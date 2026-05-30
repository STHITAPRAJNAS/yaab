"""External vector-store adapters: Chroma, Qdrant, OpenSearch, and Oracle 23ai.

Each satisfies the :class:`~yaab.rag.store.VectorStore` protocol and is
registered as a ``vectorstore`` component, so they drop into a ``KnowledgeBase``
unchanged. Their client libraries are imported lazily — install only what you
use. (Aurora PostgreSQL and any pgvector-enabled Postgres are served by
:class:`~yaab.rag.store.PgVectorStore`.)
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


class OpenSearchVectorStore:
    """OpenSearch / Amazon OpenSearch Service k-NN vector store.

    Requires ``opensearch-py`` (``pip install opensearch-py``). Uses a knn_vector
    field with cosine similarity; metadata is stored alongside and filtered with
    a bool/term query, so per-tenant isolation pushes down to the cluster.

    Works against self-managed OpenSearch, Amazon OpenSearch Service, and
    OpenSearch Serverless — pass a configured ``client`` or connection kwargs.
    """

    def __init__(
        self,
        *,
        index: str = "yaab",
        dim: int = 1536,
        hosts: Any = None,
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            from opensearchpy import OpenSearch  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "opensearch-py is required for OpenSearchVectorStore. `pip install opensearch-py`."
            ) from exc
        self._client = client or OpenSearch(
            hosts=hosts or ["http://localhost:9200"], **client_kwargs
        )
        self._index = index
        self._dim = dim
        if not self._client.indices.exists(index=index):
            self._client.indices.create(
                index=index,
                body={
                    "settings": {"index": {"knn": True}},
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "knn_vector", "dimension": dim},
                            "text": {"type": "text"},
                            "source": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "idx": {"type": "integer"},
                            "metadata": {"type": "object", "enabled": True},
                        }
                    },
                },
            )

    def add(self, chunks: list[Chunk]) -> None:
        from opensearchpy.helpers import bulk  # type: ignore

        actions = [
            {
                "_index": self._index,
                "_id": c.id,
                "_source": {
                    "embedding": c.embedding,
                    "text": c.text,
                    "source": c.source,
                    "document_id": c.document_id,
                    "idx": c.index,
                    "metadata": c.metadata,
                },
            }
            for c in chunks
            if c.embedding
        ]
        if actions:
            bulk(self._client, actions, refresh=True)

    def query(
        self, embedding: list[float], *, k: int = 5, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        knn = {"knn": {"embedding": {"vector": embedding, "k": k}}}
        if where:
            query: dict = {
                "bool": {
                    "must": [knn],
                    "filter": [{"term": {f"metadata.{key}": val}} for key, val in where.items()],
                }
            }
        else:
            query = knn
        res = self._client.search(index=self._index, body={"size": k, "query": query})
        out: list[RetrievedChunk] = []
        for hit in res["hits"]["hits"]:
            src = hit["_source"]
            chunk = Chunk(
                id=hit["_id"],
                text=src.get("text", ""),
                source=src.get("source"),
                document_id=src.get("document_id"),
                index=int(src.get("idx", 0)),
                metadata=src.get("metadata", {}),
            )
            out.append(RetrievedChunk(chunk=chunk, score=float(hit["_score"])))
        return out

    def delete(self, *, where: Filter | None = None) -> int:
        if where is None:
            body = {"query": {"match_all": {}}}
        else:
            body = {
                "query": {
                    "bool": {"filter": [{"term": {f"metadata.{k}": v}} for k, v in where.items()]}
                }
            }
        res = self._client.delete_by_query(index=self._index, body=body, refresh=True)
        return int(res.get("deleted", 0))

    def count(self) -> int:
        return int(self._client.count(index=self._index)["count"])


class OracleVectorStore:
    """Oracle Database 23ai AI Vector Search store.

    Requires ``oracledb`` (``pip install oracledb``) and Oracle Database 23ai
    (which adds the native ``VECTOR`` type and ``VECTOR_DISTANCE``). Pass a live
    ``connection`` or a DSN + credentials. Metadata is stored as JSON and
    filtered with ``JSON_VALUE`` so tenant isolation runs in the database.
    """

    def __init__(
        self,
        *,
        table: str = "yaab_chunks",
        dim: int = 1536,
        connection: Any = None,
        dsn: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        try:
            import oracledb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "oracledb is required for OracleVectorStore. `pip install oracledb`."
            ) from exc
        self._conn = connection or oracledb.connect(user=user, password=password, dsn=dsn)
        self._table = table
        self._dim = dim
        with self._conn.cursor() as cur:
            cur.execute(
                f"""BEGIN
                    EXECUTE IMMEDIATE 'CREATE TABLE {table} (
                        id VARCHAR2(64) PRIMARY KEY, text CLOB, source VARCHAR2(512),
                        document_id VARCHAR2(64), idx NUMBER, metadata JSON,
                        embedding VECTOR({dim}, FLOAT32))';
                EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF;
                END;"""
            )
        self._conn.commit()

    def add(self, chunks: list[Chunk]) -> None:
        import array
        import json

        rows = [
            (
                c.id,
                c.text,
                c.source,
                c.document_id,
                c.index,
                json.dumps(c.metadata),
                array.array("f", c.embedding),
            )
            for c in chunks
            if c.embedding
        ]
        if not rows:
            return
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self._table} "
                f"(id, text, source, document_id, idx, metadata, embedding) "
                f"VALUES (:1, :2, :3, :4, :5, :6, :7)",
                rows,
            )
        self._conn.commit()

    def query(
        self, embedding: list[float], *, k: int = 5, where: Filter | None = None
    ) -> list[RetrievedChunk]:
        import array

        vec = array.array("f", embedding)
        sql = (
            f"SELECT id, text, source, document_id, idx, metadata, "
            f"VECTOR_DISTANCE(embedding, :vec, COSINE) AS dist FROM {self._table}"
        )
        binds: dict[str, Any] = {"vec": vec}
        if where:
            clauses = []
            for i, (key, val) in enumerate(where.items()):
                clauses.append(f"JSON_VALUE(metadata, '$.{key}') = :w{i}")
                binds[f"w{i}"] = str(val)
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY dist FETCH FIRST :k ROWS ONLY"
        binds["k"] = k
        out: list[RetrievedChunk] = []
        with self._conn.cursor() as cur:
            cur.execute(sql, binds)
            for rid, text, source, doc_id, idx, metadata, dist in cur.fetchall():
                import json

                meta = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
                text_val = text.read() if hasattr(text, "read") else text
                chunk = Chunk(
                    id=rid,
                    text=text_val or "",
                    source=source,
                    document_id=doc_id,
                    index=int(idx or 0),
                    metadata=meta,
                )
                out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist)))
        return out

    def delete(self, *, where: Filter | None = None) -> int:
        sql = f"DELETE FROM {self._table}"
        binds: dict[str, Any] = {}
        if where is not None:
            clauses = []
            for i, (key, val) in enumerate(where.items()):
                clauses.append(f"JSON_VALUE(metadata, '$.{key}') = :w{i}")
                binds[f"w{i}"] = str(val)
            sql += " WHERE " + " AND ".join(clauses)
        with self._conn.cursor() as cur:
            cur.execute(sql, binds)
            n = cur.rowcount
        self._conn.commit()
        return int(n or 0)

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return int(cur.fetchone()[0])


register("vectorstore", "chroma", lambda **kw: ChromaVectorStore(**kw))
register("vectorstore", "qdrant", lambda **kw: QdrantVectorStore(**kw))
register("vectorstore", "opensearch", lambda **kw: OpenSearchVectorStore(**kw))
register("vectorstore", "oracle", lambda **kw: OracleVectorStore(**kw))

__all__ = [
    "ChromaVectorStore",
    "QdrantVectorStore",
    "OpenSearchVectorStore",
    "OracleVectorStore",
]
