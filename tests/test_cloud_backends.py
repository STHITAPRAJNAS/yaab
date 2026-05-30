"""Tests for cloud DB / vector-store / session backends: registration,
lazy imports, helpful errors, and an in-memory functional check."""

from __future__ import annotations

import pytest

from yaab.extensions import available, get
from yaab.rag.store import InMemoryVectorStore
from yaab.rag.types import Chunk


# --- registration: every backend is discoverable by name --------------
def test_vectorstores_registered():
    names = set(available("vectorstore"))
    assert {
        "memory",
        "pgvector",
        "aurora",
        "chroma",
        "qdrant",
        "opensearch",
        "oracle",
    } <= names


def test_sessions_registered():
    names = set(available("session"))
    assert {"memory", "sqlite", "postgres", "aurora", "redis"} <= names


# --- lazy import: classes resolve without their client libs -----------
def test_external_vectorstore_classes_resolve():
    from yaab.rag import (
        ChromaVectorStore,
        OpenSearchVectorStore,
        OracleVectorStore,
        QdrantVectorStore,
    )

    assert all(
        isinstance(c, type)
        for c in (ChromaVectorStore, QdrantVectorStore, OpenSearchVectorStore, OracleVectorStore)
    )


def test_session_classes_resolve():
    from yaab.sessions import PostgresSessionService, RedisSessionService

    assert isinstance(PostgresSessionService, type)
    assert isinstance(RedisSessionService, type)


# --- helpful errors when the driver is missing ------------------------
def test_opensearch_requires_driver():
    pytest.importorskip  # noqa: B018 - sentinel; we expect opensearch-py absent
    try:
        import opensearchpy  # noqa: F401

        pytest.skip("opensearch-py is installed")
    except ImportError:
        pass
    from yaab.rag import OpenSearchVectorStore

    with pytest.raises(RuntimeError, match="opensearch-py"):
        OpenSearchVectorStore()


def test_oracle_requires_driver():
    try:
        import oracledb  # noqa: F401

        pytest.skip("oracledb is installed")
    except ImportError:
        pass
    from yaab.rag import OracleVectorStore

    with pytest.raises(RuntimeError, match="oracledb"):
        OracleVectorStore()


# --- the in-memory default actually works end to end ------------------
def test_in_memory_vectorstore_roundtrip():
    store = InMemoryVectorStore()
    store.add([Chunk(text="a", embedding=[1.0, 0.0], metadata={"u": "x"})])
    store.add([Chunk(text="b", embedding=[0.0, 1.0], metadata={"u": "y"})])
    hits = store.query([1.0, 0.0], k=1)
    assert hits[0].chunk.text == "a"
    # metadata filter (per-tenant isolation) works on the default store too
    hits = store.query([1.0, 0.0], k=5, where={"u": "y"})
    assert all(h.chunk.metadata["u"] == "y" for h in hits)


@pytest.mark.asyncio
async def test_in_memory_session_via_registry():
    svc = get("session", "memory")
    s = await svc.get_or_create("s1")
    assert s.id == "s1"
    await svc.save(s)
    assert (await svc.get("s1")).id == "s1"


def test_custom_backend_is_extensible():
    # A third party can register their own store and select it by name.
    from yaab.extensions import register

    class MyStore(InMemoryVectorStore):
        pass

    register("vectorstore", "mystore", lambda **kw: MyStore())
    assert "mystore" in available("vectorstore")
    assert isinstance(get("vectorstore", "mystore"), MyStore)
