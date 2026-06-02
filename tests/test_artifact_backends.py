"""Tests for durable artifact backends and a shared version index.

These prove that artifact bytes and their version history survive a process and
are visible to a second process pointed at the same store (simulated by two
service instances over one SQLite file or one fake Redis client). Offline only:
SQLite tempfiles and injected fake clients, no network.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from yaab.artifacts import (
    Artifact,
    ArtifactService,
    InMemoryArtifactService,
    SQLiteArtifactService,
)
from yaab.artifacts.manager import ArtifactManager


def _tmp_db() -> str:
    d = tempfile.mkdtemp()
    return str(Path(d) / "artifacts.db")


def test_sqlite_roundtrip(anyio_backend: object = None) -> None:
    import asyncio

    path = _tmp_db()

    async def main() -> None:
        svc = SQLiteArtifactService(path)
        art = await svc.put("report.bin", b"\x00\x01\x02", mime_type="application/octet-stream")
        assert isinstance(art, Artifact)
        assert art.size == 3
        got = await svc.get(art.id)
        assert got == b"\x00\x01\x02"
        info = await svc.info(art.id)
        assert info is not None and info.name == "report.bin"
        assert await svc.get("nope") is None
        assert await svc.info("nope") is None

    asyncio.run(main())


def test_sqlite_cross_instance_visibility() -> None:
    """A second service over the same file sees bytes written by the first."""
    import asyncio

    path = _tmp_db()

    async def main() -> None:
        a = SQLiteArtifactService(path)
        art = await a.put("shared.dat", b"hello-pods")
        # Simulate a second pod opening the same durable store.
        b = SQLiteArtifactService(path)
        assert await b.get(art.id) == b"hello-pods"
        info = await b.info(art.id)
        assert info is not None and info.size == len(b"hello-pods")

    asyncio.run(main())


def test_sqlite_implements_protocol() -> None:
    svc = SQLiteArtifactService(_tmp_db())
    assert isinstance(svc, ArtifactService)


def test_manager_durable_version_index_is_shared() -> None:
    """Version history written by one manager is visible to a second manager.

    The durable backend owns the version index, so two managers over the same
    SQLite file agree on how many versions exist and can load each one.
    """
    import asyncio

    path = _tmp_db()

    async def main() -> None:
        svc1 = SQLiteArtifactService(path)
        m1 = ArtifactManager(svc1)
        assert await m1.save("doc", b"v1") == 1
        assert await m1.save("doc", b"v2") == 2

        # A fresh manager + fresh service over the same file (a second pod).
        svc2 = SQLiteArtifactService(path)
        m2 = ArtifactManager(svc2)
        assert await m2.list_versions("doc") == 2
        assert await m2.load("doc", version=1) == b"v1"
        assert await m2.load("doc", version=2) == b"v2"
        assert await m2.load("doc") == b"v2"  # latest
        # And a third write from the second pod is seen by the first.
        assert await m2.save("doc", b"v3") == 3
        assert await m1.list_versions("doc") == 3
        assert await m1.load("doc") == b"v3"

    asyncio.run(main())


def test_manager_inmemory_still_works() -> None:
    """The in-memory manager path is unchanged (per-process version index)."""
    import asyncio

    async def main() -> None:
        m = ArtifactManager(InMemoryArtifactService())
        assert await m.save("x", b"a") == 1
        assert await m.save("x", b"b") == 2
        assert await m.list_versions("x") == 2
        assert await m.load("x", version=1) == b"a"
        assert await m.load("x") == b"b"
        assert "x" in await m.list_artifacts()

    asyncio.run(main())


def test_manager_list_artifacts_durable() -> None:
    import asyncio

    path = _tmp_db()

    async def main() -> None:
        m = ArtifactManager(SQLiteArtifactService(path))
        await m.save("a", b"1")
        await m.save("b", b"2")
        names = await m.list_artifacts()
        assert set(names) == {"a", "b"}

    asyncio.run(main())


def test_backends_registered() -> None:
    from yaab.extensions import available, get

    names = set(available("artifact"))
    assert {"memory", "sqlite"} <= names
    svc = get("artifact", "sqlite", path=_tmp_db())
    assert isinstance(svc, SQLiteArtifactService)
    mem = get("artifact", "memory")
    assert isinstance(mem, InMemoryArtifactService)


def test_lazy_postgres_and_redis_exports() -> None:
    from yaab import artifacts

    assert isinstance(artifacts.PostgresArtifactService, type)
    assert isinstance(artifacts.RedisArtifactService, type)


def test_redis_artifact_with_fake_client_roundtrips() -> None:
    """A fake Redis client proves the backend logic without the driver."""
    import asyncio

    from yaab.artifacts import RedisArtifactService

    class FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, bytes] = {}
            self.hashes: dict[str, dict[str, str]] = {}

        def set(self, key: str, value: bytes, ex: object = None) -> None:
            self.store[key] = value

        def get(self, key: str) -> bytes | None:
            return self.store.get(key)

        def hset(self, key: str, field: str, value: str) -> None:
            self.hashes.setdefault(key, {})[field] = value

        def hget(self, key: str, field: str) -> str | None:
            return self.hashes.get(key, {}).get(field)

        def rpush(self, key: str, value: str) -> None:
            self.hashes.setdefault(key, {})
            lst = self.hashes[key].setdefault("__list__", "")
            self.hashes[key]["__list__"] = (lst + "\n" + value) if lst else value

        def lrange(self, key: str, start: int, end: int) -> list[str]:
            raw = self.hashes.get(key, {}).get("__list__", "")
            return raw.split("\n") if raw else []

    client = FakeRedis()
    svc = RedisArtifactService(client=client)

    async def main() -> None:
        art = await svc.put("blob", b"\xff\xfe", mime_type="image/png")
        assert await svc.get(art.id) == b"\xff\xfe"
        info = await svc.info(art.id)
        assert info is not None and info.mime_type == "image/png"

        # Cross-instance: a second service over the same client sees it.
        svc2 = RedisArtifactService(client=client)
        assert await svc2.get(art.id) == b"\xff\xfe"

    asyncio.run(main())


def test_redis_artifact_requires_driver() -> None:
    try:
        import redis  # noqa: F401

        pytest.skip("redis is installed")
    except ImportError:
        pass
    from yaab.artifacts import RedisArtifactService

    with pytest.raises(RuntimeError, match="redis"):
        RedisArtifactService()


def test_postgres_artifact_requires_driver() -> None:
    try:
        import psycopg  # noqa: F401

        pytest.skip("psycopg is installed")
    except ImportError:
        pass
    from yaab.artifacts import PostgresArtifactService

    with pytest.raises(RuntimeError, match="psycopg"):
        PostgresArtifactService("postgresql://x")
