"""Tests for the added backends: Pinecone/Weaviate stores + Postgres/Redis savers."""

from __future__ import annotations

import pytest

from yaab.extensions import available


def test_new_vectorstores_registered():
    names = set(available("vectorstore"))
    assert {"pinecone", "weaviate"} <= names


def test_checkpointers_registered():
    names = set(available("checkpointer"))
    assert {"memory", "sqlite", "postgres", "aurora", "redis"} <= names


def test_new_store_classes_resolve():
    from yaab.rag import PineconeVectorStore, WeaviateVectorStore

    assert isinstance(PineconeVectorStore, type)
    assert isinstance(WeaviateVectorStore, type)


def test_saver_classes_resolve():
    from yaab.graph import PostgresSaver, RedisSaver

    assert isinstance(PostgresSaver, type)
    assert isinstance(RedisSaver, type)


def test_pinecone_requires_driver():
    try:
        import pinecone  # noqa: F401

        pytest.skip("pinecone is installed")
    except ImportError:
        pass
    from yaab.rag import PineconeVectorStore

    with pytest.raises(RuntimeError, match="pinecone"):
        PineconeVectorStore()


def test_weaviate_requires_driver():
    try:
        import weaviate  # noqa: F401

        pytest.skip("weaviate-client is installed")
    except ImportError:
        pass
    from yaab.rag import WeaviateVectorStore

    with pytest.raises(RuntimeError, match="weaviate"):
        WeaviateVectorStore()


def test_postgres_saver_requires_driver():
    try:
        import psycopg  # noqa: F401

        pytest.skip("psycopg is installed")
    except ImportError:
        pass
    from yaab.graph import PostgresSaver

    with pytest.raises(RuntimeError, match="psycopg"):
        PostgresSaver("postgresql://x")


def test_redis_saver_requires_driver():
    try:
        import redis  # noqa: F401

        pytest.skip("redis is installed")
    except ImportError:
        pass
    from yaab.graph import RedisSaver

    with pytest.raises(RuntimeError, match="redis"):
        RedisSaver()


def test_redis_saver_with_fake_client_roundtrips():
    # A fake Redis hash client proves the saver logic without the driver.
    class FakeRedis:
        def __init__(self):
            self.store: dict[str, dict[str, str]] = {}

        def hset(self, key, field, value):
            self.store.setdefault(key, {})[field] = value

        def hgetall(self, key):
            return dict(self.store.get(key, {}))

        def expire(self, key, ttl):
            pass

    from yaab.graph import RedisSaver

    saver = RedisSaver(client=FakeRedis())
    saver.put("t1", 0, {"count": 1})
    saver.put("t1", 1, {"count": 2})
    step, state = saver.get("t1")
    assert step == 1 and state == {"count": 2}
    hist = saver.history("t1")
    assert [s for s, _ in hist] == [0, 1]


def test_graph_uses_redis_saver_via_fake():
    from yaab.graph import END, START, Channel, RedisSaver, StateGraph

    class FakeRedis:
        def __init__(self):
            self.store = {}

        def hset(self, key, field, value):
            self.store.setdefault(key, {})[field] = value

        def hgetall(self, key):
            return dict(self.store.get(key, {}))

        def expire(self, key, ttl):
            pass

    g = StateGraph(channels={"count": Channel("add", default=0)})
    g.add_node("inc", lambda s: {"count": 1})
    g.add_edge(START, "inc")
    g.add_conditional_edges(
        "inc", lambda s: "inc" if s["count"] < 3 else END, {"inc": "inc", END: END}
    )
    app = g.compile(checkpointer=RedisSaver(client=FakeRedis()))
    result = app.invoke({}, thread_id="job")
    assert result.state["count"] == 3
