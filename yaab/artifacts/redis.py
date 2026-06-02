"""Redis artifact backend for distributed/cloud deployments.

Stores artifact bytes as Redis string values and metadata as a hash, with a
per-scope list holding the ordered version history so artifact versioning is
shared across every replica. Uses ``redis`` (``pip install 'yaab-sdk[redis]'``),
imported lazily. Works against self-managed Redis, Amazon ElastiCache /
MemoryDB, Azure Cache for Redis, and Redis Cloud.
"""

from __future__ import annotations

import json
from typing import Any

from . import Artifact


class RedisArtifactService:
    """Persist artifact bytes (and version history) in Redis."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "yaab:artifact",
        ttl_seconds: int | None = None,
        client: Any = None,
    ) -> None:
        self._redis: Any
        if client is not None:
            self._redis = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "redis is required for RedisArtifactService. "
                    "Install with `pip install 'yaab-sdk[redis]'`."
                ) from exc
            # Raw bytes for blob fidelity; metadata is decoded per-field on read.
            self._redis = redis.Redis.from_url(url)
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _data_key(self, artifact_id: str) -> str:
        return f"{self._prefix}:data:{artifact_id}"

    def _meta_key(self, artifact_id: str) -> str:
        return f"{self._prefix}:meta:{artifact_id}"

    def _versions_key(self, scope: str) -> str:
        return f"{self._prefix}:versions:{scope}"

    async def put(
        self, name: str, data: bytes, *, mime_type: str = "application/octet-stream"
    ) -> Artifact:
        art = Artifact(name=name, mime_type=mime_type, size=len(data))
        self._redis.set(self._data_key(art.id), data, ex=self._ttl)
        self._redis.hset(self._meta_key(art.id), "info", art.model_dump_json())
        return art

    async def get(self, artifact_id: str) -> bytes | None:
        raw = self._redis.get(self._data_key(artifact_id))
        if raw is None:
            return None
        return raw if isinstance(raw, bytes) else bytes(raw, "latin-1")

    async def info(self, artifact_id: str) -> Artifact | None:
        raw = self._redis.hget(self._meta_key(artifact_id), "info")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return Artifact.model_validate(json.loads(raw))

    # --- shared version index -----------------------------------------------

    async def append_version(self, scope: str, artifact_id: str) -> int:
        self._redis.rpush(self._versions_key(scope), artifact_id)
        return len(await self.list_version_ids(scope))

    async def list_version_ids(self, scope: str) -> list[str]:
        raw = self._redis.lrange(self._versions_key(scope), 0, -1)
        out: list[str] = []
        for item in raw:
            out.append(item.decode("utf-8") if isinstance(item, bytes) else item)
        return out


__all__ = ["RedisArtifactService"]
