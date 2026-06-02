"""Redis run store — a distributed, durable run queue.

Each run record is a JSON value under a per-run key; a ``queued`` list gives
FIFO claim order, and ``LMOVE`` atomically pops the oldest queued id onto a
``processing`` list so exactly one worker claims it. Per-run lease keys carry a
TTL, and the reaper re-queues any running run whose lease key has expired —
turning a crashed replica's in-flight runs back into claimable work.

A cancel sets a durable flag on the record (the correctness floor) and also
publishes on ``<prefix>:cancel:<run_id>`` for an optional instant cross-replica
fast path.

Uses ``redis`` (``pip install 'yaab-sdk[redis]'``), imported lazily. A client can
be injected via ``client=`` for offline tests against a fake.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import RunRecord, RunStatus

# Alias for ``list[str]`` used after ``def list`` shadows the builtin.
_RunIds = list


class RedisRunStore:
    """Persist run records and the work queue in Redis."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "yaab:run",
        client: Any = None,
    ) -> None:
        self._redis: Any
        if client is not None:
            self._redis = client
        else:
            try:
                import redis  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "redis is required for RedisRunStore. `pip install 'yaab-sdk[redis]'`."
                ) from exc
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    # --- key helpers ------------------------------------------------------
    def _rec_key(self, run_id: str) -> str:
        return f"{self._prefix}:rec:{run_id}"

    def _lease_key(self, run_id: str) -> str:
        return f"{self._prefix}:lease:{run_id}"

    @property
    def _index_key(self) -> str:
        return f"{self._prefix}:index"

    @property
    def _queue_key(self) -> str:
        return f"{self._prefix}:queued"

    @property
    def _processing_key(self) -> str:
        return f"{self._prefix}:processing"

    def _cancel_channel(self, run_id: str) -> str:
        return f"{self._prefix}:cancel:{run_id}"

    # --- (de)serialization ------------------------------------------------
    @staticmethod
    def _decode(raw: Any) -> RunRecord | None:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return RunRecord.model_validate_json(raw)

    def _write(self, record: RunRecord) -> None:
        self._redis.set(self._rec_key(record.run_id), record.model_dump_json())
        # Maintain a sortable index of (created_at -> run_id) for listing.
        self._redis.zadd(self._index_key, {record.run_id: record.created_at})

    def _read(self, run_id: str) -> RunRecord | None:
        return self._decode(self._redis.get(self._rec_key(run_id)))

    # --- lifecycle --------------------------------------------------------
    async def create(self, record: RunRecord) -> None:
        self._write(record)
        if record.status is RunStatus.QUEUED:
            self._redis.rpush(self._queue_key, record.run_id)

    async def get(self, run_id: str) -> RunRecord | None:
        return self._read(run_id)

    async def update(
        self, run_id: str, *, expect_status: RunStatus | None = None, **fields: Any
    ) -> RunRecord | None:
        rec = self._read(run_id)
        if rec is None:
            return None
        if expect_status is not None and rec.status is not expect_status:
            return None
        fields.setdefault("updated_at", time.time())
        updated = rec.model_copy(update=fields)
        self._write(updated)
        return updated

    async def list(self, *, limit: int = 100, status: RunStatus | None = None) -> list[RunRecord]:
        # Newest-first ids from the sorted index, fetched in bounded pages rather
        # than all at once: ``zrevrange(0, -1)`` would pull every run id into
        # memory before slicing. We walk the index a page at a time and stop as
        # soon as ``limit`` matches are collected. (Status-filtered scans may walk
        # further, but never materialize the whole index in one call.)
        records: list[RunRecord] = []
        page = max(limit, 100)
        offset = 0
        while len(records) < limit:
            ids = self._redis.zrevrange(self._index_key, offset, offset + page - 1)
            if not ids:
                break
            for rid in ids:
                if isinstance(rid, bytes):
                    rid = rid.decode("utf-8")
                rec = self._read(rid)
                if rec is None:
                    continue
                if status is not None and rec.status is not status:
                    continue
                records.append(rec)
                if len(records) >= limit:
                    break
            offset += page
        return records

    async def request_cancel(self, run_id: str) -> bool:
        rec = self._read(run_id)
        if rec is None:
            return False
        self._write(rec.model_copy(update={"cancel_requested": True, "updated_at": time.time()}))
        # Optional instant cross-replica fast path; polling the flag is the floor.
        try:
            self._redis.publish(self._cancel_channel(run_id), "1")
        except Exception:  # noqa: BLE001 - pub/sub is best-effort
            pass
        return True

    # --- worker queue primitives -----------------------------------------
    #: A claim must be atomic across three keys: pop the oldest queued id onto the
    #: processing list, flip its record to RUNNING (with owner + fenced lease
    #: generation), and set the lease TTL key — with no gap a reaper could observe
    #: a popped-but-still-QUEUED record through. A Lua script runs the whole
    #: sequence as one server-side step so exactly one worker ever wins a run and
    #: the record is RUNNING the instant it leaves the queue.
    _CLAIM_LUA = """
    local run_id = redis.call('LMOVE', KEYS[1], KEYS[2], 'LEFT', 'RIGHT')
    if not run_id then return nil end
    local rec_key = ARGV[1] .. run_id
    local raw = redis.call('GET', rec_key)
    if not raw then
        redis.call('LREM', KEYS[2], 0, run_id)
        return nil
    end
    return cjson.encode({run_id = run_id, data = raw})
    """

    async def claim_next(self, *, pod_id: str, lease_seconds: float) -> RunRecord | None:
        now = time.time()
        rec_prefix = f"{self._prefix}:rec:"
        raw = self._eval_claim(rec_prefix)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        popped = json.loads(raw)
        run_id = popped["run_id"]
        if isinstance(run_id, bytes):
            run_id = run_id.decode("utf-8")
        rec = self._decode(popped["data"])
        if rec is None:  # pragma: no cover - script guards this, defensive only
            self._redis.lrem(self._processing_key, 0, run_id)
            return None
        claimed = rec.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "owner_pod": pod_id,
                "lease_expires_at": now + lease_seconds,
                "started_at": rec.started_at or now,
                "updated_at": now,
                "lease_generation": rec.lease_generation + 1,
            }
        )
        # Flip the record to RUNNING and arm the lease as one pipeline so the
        # window where the popped record still reads QUEUED is closed: a reaper
        # only ever sees it as RUNNING (claimed) or still in the queue.
        pipe = self._redis.pipeline()
        pipe.set(self._rec_key(run_id), claimed.model_dump_json())
        pipe.zadd(self._index_key, {run_id: claimed.created_at})
        pipe.set(self._lease_key(run_id), pod_id, ex=int(lease_seconds) + 1)
        pipe.execute()
        return claimed

    def _eval_claim(self, rec_prefix: str) -> Any:
        """Run the atomic LMOVE-and-read claim script (server-side, one step)."""
        return self._redis.eval(
            self._CLAIM_LUA,
            2,
            self._queue_key,
            self._processing_key,
            rec_prefix,
        )

    async def heartbeat(self, run_id: str, *, pod_id: str, lease_seconds: float) -> None:
        ttl = int(lease_seconds) + 1
        self._redis.set(self._lease_key(run_id), pod_id, ex=ttl)
        await self.update(
            run_id,
            owner_pod=pod_id,
            lease_expires_at=time.time() + lease_seconds,
        )

    async def reap_expired_leases(self) -> _RunIds[str]:
        now = time.time()
        reaped: list[str] = []
        processing = self._redis.lrange(self._processing_key, 0, -1)
        for run_id in processing:
            if isinstance(run_id, bytes):
                run_id = run_id.decode("utf-8")
            rec = self._read(run_id)
            if rec is None or rec.status is not RunStatus.RUNNING:
                self._redis.lrem(self._processing_key, 0, run_id)
                continue
            # The lease has expired when its TTL key is gone, or the recorded
            # deadline has passed (covers fakes without real TTL expiry).
            lease_present = self._redis.exists(self._lease_key(run_id))
            deadline_passed = rec.lease_expires_at is not None and rec.lease_expires_at < now
            if lease_present and not deadline_passed:
                continue
            self._write(
                rec.model_copy(
                    update={
                        "status": RunStatus.QUEUED,
                        "owner_pod": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                        # Fence: a reaped run gets a new generation so the stale
                        # worker can no longer finalize over the re-claimer.
                        "lease_generation": rec.lease_generation + 1,
                    }
                )
            )
            self._redis.lrem(self._processing_key, 0, run_id)
            self._redis.rpush(self._queue_key, run_id)
            self._redis.delete(self._lease_key(run_id))
            reaped.append(run_id)
        return reaped


__all__ = ["RedisRunStore"]
