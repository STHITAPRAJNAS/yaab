"""SQLite artifact backend — durable binary storage for single-node deployments.

Stores artifact bytes in a ``BLOB`` column keyed by artifact id, plus a small
version index so the named/versioned history kept by
:class:`~yaab.artifacts.manager.ArtifactManager` survives a restart and is shared
across processes pointed at the same file.
"""

from __future__ import annotations

import sqlite3

from . import Artifact


class SQLiteArtifactService:
    """Persist artifact bytes and version history in a SQLite database."""

    def __init__(self, path: str = "yaab_artifacts.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS artifacts ("
            "id TEXT PRIMARY KEY, name TEXT NOT NULL, mime_type TEXT NOT NULL, "
            "size INTEGER NOT NULL, metadata TEXT NOT NULL, data BLOB NOT NULL)"
        )
        # Shared version index: scoped name -> ordered artifact ids (one per row).
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS artifact_versions ("
            "scope TEXT NOT NULL, version INTEGER NOT NULL, artifact_id TEXT NOT NULL, "
            "PRIMARY KEY (scope, version))"
        )
        self._conn.commit()

    async def put(
        self, name: str, data: bytes, *, mime_type: str = "application/octet-stream"
    ) -> Artifact:
        art = Artifact(name=name, mime_type=mime_type, size=len(data))
        self._conn.execute(
            "INSERT OR REPLACE INTO artifacts (id, name, mime_type, size, metadata, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (art.id, art.name, art.mime_type, art.size, _dump_meta(art), data),
        )
        self._conn.commit()
        return art

    async def get(self, artifact_id: str) -> bytes | None:
        row = self._conn.execute(
            "SELECT data FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        return bytes(row[0]) if row is not None else None

    async def info(self, artifact_id: str) -> Artifact | None:
        row = self._conn.execute(
            "SELECT id, name, mime_type, size, metadata FROM artifacts WHERE id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_artifact(row)

    # --- shared version index (used by ArtifactManager when durable) ---------

    async def append_version(self, scope: str, artifact_id: str) -> int:
        """Record ``artifact_id`` as the next version of ``scope``; return count."""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM artifact_versions WHERE scope = ?",
            (scope,),
        ).fetchone()
        version = int(row[0]) + 1
        self._conn.execute(
            "INSERT INTO artifact_versions (scope, version, artifact_id) VALUES (?, ?, ?)",
            (scope, version, artifact_id),
        )
        self._conn.commit()
        return version

    async def list_version_ids(self, scope: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT artifact_id FROM artifact_versions WHERE scope = ? ORDER BY version",
            (scope,),
        ).fetchall()
        return [r[0] for r in rows]

    async def version_scopes(self, prefix: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT scope FROM artifact_versions WHERE scope LIKE ?",
            (prefix + "%",),
        ).fetchall()
        return [r[0] for r in rows]


def _dump_meta(art: Artifact) -> str:
    import json

    return json.dumps(art.metadata)


def _row_to_artifact(row: tuple) -> Artifact:
    import json

    return Artifact(
        id=row[0],
        name=row[1],
        mime_type=row[2],
        size=row[3],
        metadata=json.loads(row[4]),
    )


__all__ = ["SQLiteArtifactService"]
