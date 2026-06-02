"""Postgres artifact backend for production, high-availability deployments.

Stores artifact bytes in a ``BYTEA`` column keyed by artifact id, with a shared
version index table so named/versioned artifact history is consistent across
every replica. Uses ``psycopg`` (v3), imported lazily so it is only required
when this backend is actually constructed.
"""

from __future__ import annotations

import json

from . import Artifact


def _require_psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "psycopg is required for the Postgres backends. "
            "Install with `pip install 'yaab-sdk[postgres]'`."
        ) from exc
    return psycopg


class PostgresArtifactService:
    """Persist artifact bytes and version history in Postgres / Aurora."""

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "yaab_artifacts",
        versions_table: str = "yaab_artifact_versions",
    ) -> None:
        psycopg = _require_psycopg()
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._table = table
        self._versions_table = versions_table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"id TEXT PRIMARY KEY, name TEXT NOT NULL, mime_type TEXT NOT NULL, "
            f"size INTEGER NOT NULL, metadata JSONB NOT NULL, data BYTEA NOT NULL)"
        )
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {versions_table} ("
            f"scope TEXT NOT NULL, version INTEGER NOT NULL, artifact_id TEXT NOT NULL, "
            f"PRIMARY KEY (scope, version))"
        )

    async def put(
        self, name: str, data: bytes, *, mime_type: str = "application/octet-stream"
    ) -> Artifact:
        art = Artifact(name=name, mime_type=mime_type, size=len(data))
        self._conn.execute(
            f"INSERT INTO {self._table} (id, name, mime_type, size, metadata, data) "
            f"VALUES (%s, %s, %s, %s, %s, %s) "
            f"ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data",
            (art.id, art.name, art.mime_type, art.size, json.dumps(art.metadata), data),
        )
        return art

    async def get(self, artifact_id: str) -> bytes | None:
        row = self._conn.execute(
            f"SELECT data FROM {self._table} WHERE id = %s", (artifact_id,)
        ).fetchone()
        return bytes(row[0]) if row is not None else None

    async def info(self, artifact_id: str) -> Artifact | None:
        row = self._conn.execute(
            f"SELECT id, name, mime_type, size, metadata FROM {self._table} WHERE id = %s",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        meta = row[4] if isinstance(row[4], dict) else json.loads(row[4])
        return Artifact(id=row[0], name=row[1], mime_type=row[2], size=row[3], metadata=meta)

    # --- shared version index -----------------------------------------------

    async def append_version(self, scope: str, artifact_id: str) -> int:
        row = self._conn.execute(
            f"SELECT COALESCE(MAX(version), 0) FROM {self._versions_table} WHERE scope = %s",
            (scope,),
        ).fetchone()
        version = int(row[0]) + 1
        self._conn.execute(
            f"INSERT INTO {self._versions_table} (scope, version, artifact_id) VALUES (%s, %s, %s)",
            (scope, version, artifact_id),
        )
        return version

    async def list_version_ids(self, scope: str) -> list[str]:
        rows = self._conn.execute(
            f"SELECT artifact_id FROM {self._versions_table} WHERE scope = %s ORDER BY version",
            (scope,),
        ).fetchall()
        return [r[0] for r in rows]

    async def version_scopes(self, prefix: str) -> list[str]:
        rows = self._conn.execute(
            f"SELECT DISTINCT scope FROM {self._versions_table} WHERE scope LIKE %s",
            (prefix + "%",),
        ).fetchall()
        return [r[0] for r in rows]


__all__ = ["PostgresArtifactService"]
