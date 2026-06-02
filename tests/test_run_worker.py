"""Background worker — durable runs that drain a queue and survive a restart.

These tests prove the worker turns a queued run row into a completed (or
failed, cancelled, or paused) one without ever pinning a live task to a request:
it claims a queued run, executes the agent, heartbeats its lease while it runs,
records the terminal outcome on the durable record, and posts a completion
callback so callers need not poll. Crash recovery is covered by the reaper:
a run abandoned by a dead replica (an expired lease) becomes claimable again.

All offline: TestModel/FunctionModel agents, an in-memory or SQLite run store,
asyncio.Event-gated fake tools, and an in-process ASGI app to capture webhooks.
No network, no real keys, no sleeps for correctness.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from yaab import Agent, tool
from yaab.models.test_model import TestModel
from yaab.runs.base import RunRecord, RunStatus
from yaab.runs.memory import InMemoryRunStore
from yaab.runs.sqlite import SQLiteRunStore
from yaab.runs.worker import RunWorker


def _agent(out: str = "worker-output", **kw) -> Agent:
    return Agent("svc", model=TestModel(out), registry_id="svc", **kw)


def _record(run_id: str, *, prompt: str = "hi", agent: str = "svc") -> RunRecord:
    now = time.time()
    return RunRecord(
        run_id=run_id,
        agent=agent,
        status=RunStatus.QUEUED,
        prompt=prompt,
        background=True,
        resume_id=run_id,
        created_at=now,
        updated_at=now,
    )


async def _drain_one(worker: RunWorker, store) -> RunRecord:
    """Claim and execute a single queued run, returning its final record."""
    record = await store.claim_next(pod_id=worker.pod_id, lease_seconds=worker.lease_seconds)
    assert record is not None
    await worker._execute(record)
    final = await store.get(record.run_id)
    assert final is not None
    return final


# --- a claimed run executes to a terminal COMPLETED record ------------
def test_execute_completes_a_run():
    store = InMemoryRunStore()
    worker = RunWorker(_agent("answered"), store)

    async def go() -> None:
        await store.create(_record("r1", prompt="question?"))
        final = await _drain_one(worker, store)
        assert final.status is RunStatus.COMPLETED
        assert final.output == "answered"
        assert final.usage is not None and final.usage["requests"] >= 1
        assert final.finished_at is not None
        assert final.error is None

    asyncio.run(go())


# --- a model-level error surfaces as a FAILED record ------------------
def test_execute_records_failure():
    # A model that raises drives the runner to a terminal ERROR event, which the
    # worker records as a failed run (a tool that raises is, by contrast, handled
    # inside the loop and never fails the run — so we fail at the model layer).
    class _BoomModel:
        name = "boom"

        async def complete(self, messages, **kw):
            raise RuntimeError("kaboom")

        async def stream(self, messages, **kw):  # pragma: no cover - unused
            if False:  # pragma: no cover - makes this an async generator
                yield None
            raise RuntimeError("kaboom")

    agent = Agent("svc", model=_BoomModel())
    store = InMemoryRunStore()
    worker = RunWorker(agent, store)

    async def go() -> None:
        await store.create(_record("r1"))
        final = await _drain_one(worker, store)
        assert final.status is RunStatus.FAILED
        assert final.error is not None and "kaboom" in final.error
        assert final.finished_at is not None

    asyncio.run(go())


# --- a cooperative cancel surfaces as CANCELLED, not FAILED -----------
def test_execute_cancel_records_cancelled():
    gate = asyncio.Event()

    @tool
    async def slow() -> str:
        """Waits for an external gate before returning."""
        await gate.wait()
        return "released"

    agent = Agent("svc", model=TestModel("done", call_tools=["slow"]), tools=[slow])
    store = InMemoryRunStore()
    worker = RunWorker(agent, store, lease_seconds=30.0)

    async def go() -> None:
        await store.create(_record("r1"))
        record = await store.claim_next(pod_id=worker.pod_id, lease_seconds=30.0)
        assert record is not None
        task = asyncio.create_task(worker._execute(record))
        # Let the run reach the gated tool, then request a cross-replica cancel.
        for _ in range(200):
            await asyncio.sleep(0)
            if gate._waiters:  # tool is parked on the gate
                break
        await store.request_cancel("r1")
        gate.set()
        await task
        final = await store.get("r1")
        assert final is not None
        assert final.status is RunStatus.CANCELLED

    asyncio.run(go())


# --- an approval-required run is PAUSED and the lease released --------
def test_execute_pause_on_approval_releases_lease():
    from yaab.models.base import ModelResponse
    from yaab.types import EventType

    # A fake runner that yields a single APPROVAL_REQUIRED event then stops,
    # exercising the worker's eviction-on-pause path without the full HITL stack.
    class _PausingRunner:
        def __init__(self) -> None:
            self.calls = 0

        async def run_stream(self, agent, prompt, **kw):
            self.calls += 1
            from yaab.types import Event

            yield Event(
                type=EventType.APPROVAL_REQUIRED,
                agent=agent.name,
                run_id="run_x",
                payload={"approval_id": "ap1", "tool": "wire", "arguments": {}},
            )

    store = InMemoryRunStore()
    worker = RunWorker(_agent(), store, runner=_PausingRunner())

    async def go() -> None:
        await store.create(_record("r1"))
        record = await store.claim_next(pod_id=worker.pod_id, lease_seconds=30.0)
        assert record is not None and record.status is RunStatus.RUNNING
        await worker._execute(record)
        final = await store.get("r1")
        assert final is not None
        # Paused, and the worker slot is freed: no owner, no live lease.
        assert final.status is RunStatus.PAUSED
        assert final.owner_pod is None
        assert final.lease_expires_at is None
        assert final.finished_at is None  # a pause is not terminal

    asyncio.run(go())
    # Silence unused imports if the fast path changes.
    assert ModelResponse is not None


# --- the worker heartbeats the lease while a run is in flight ----------
def test_execute_heartbeats_lease():
    gate = asyncio.Event()
    seen: list[float] = []

    @tool
    async def slow() -> str:
        """Waits for the gate so the heartbeat task has time to fire."""
        await gate.wait()
        return "ok"

    agent = Agent("svc", model=TestModel("done", call_tools=["slow"]), tools=[slow])
    store = InMemoryRunStore()
    # A short heartbeat interval so a few fire while the tool is gated.
    worker = RunWorker(agent, store, lease_seconds=30.0, heartbeat_interval=0.01)

    async def go() -> None:
        await store.create(_record("r1"))
        record = await store.claim_next(pod_id=worker.pod_id, lease_seconds=30.0)
        assert record is not None
        first = (await store.get("r1")).lease_expires_at
        task = asyncio.create_task(worker._execute(record))
        # Let heartbeats fire while the tool is parked on the gate.
        for _ in range(50):
            await asyncio.sleep(0.01)
            rec = await store.get("r1")
            seen.append(rec.lease_expires_at or 0.0)
            if gate._waiters and len(seen) > 3:
                break
        gate.set()
        await task
        # The lease was pushed forward at least once beyond its claim value.
        assert max(seen) > (first or 0.0)

    asyncio.run(go())


# --- the run loop is bounded by a stop event --------------------------
def test_run_forever_stops_on_event():
    store = InMemoryRunStore()
    worker = RunWorker(_agent("x"), store, poll_interval=0.001)

    async def go() -> None:
        for i in range(3):
            await store.create(_record(f"r{i}"))
        task = asyncio.create_task(worker.run_forever())
        # Wait until all three are drained, then stop the loop.
        for _ in range(500):
            await asyncio.sleep(0.005)
            done = await store.list(status=RunStatus.COMPLETED)
            if len(done) == 3:
                break
        worker.stop()
        await asyncio.wait_for(task, timeout=2.0)
        done = await store.list(status=RunStatus.COMPLETED)
        assert {r.run_id for r in done} == {"r0", "r1", "r2"}

    asyncio.run(go())


# --- crash recovery: a reaped run is re-driven by a fresh worker ------
def test_reaper_requeues_and_second_worker_completes(tmp_path):
    path = str(tmp_path / "runs.db")
    dead_store = SQLiteRunStore(path)
    live_store = SQLiteRunStore(path)
    reaper_worker = RunWorker(_agent(), live_store, pod_id="live")
    finisher = RunWorker(_agent("recovered"), live_store, pod_id="fresh")

    async def go() -> None:
        await dead_store.create(_record("r1"))
        # A dead replica claimed it, then died (lease forced into the past).
        await dead_store.claim_next(pod_id="dead", lease_seconds=30.0)
        await dead_store.update("r1", lease_expires_at=time.time() - 1.0)

        reaped = await reaper_worker.reap()
        assert "r1" in reaped
        # A fresh worker now drains it to completion.
        final = await _drain_one(finisher, live_store)
        assert final.status is RunStatus.COMPLETED
        assert final.output == "recovered"

    asyncio.run(go())


# --- webhook: a terminal callback is POSTed to a configured URL -------
def test_webhook_posts_terminal_status():
    pytest.importorskip("httpx")
    import json as _json

    import httpx

    captured: list[dict] = []

    # A minimal in-process ASGI app that records the JSON body of every POST.
    # Using a bare ASGI callable (not a framework route) keeps the capture
    # deterministic and free of annotation/route resolution quirks.
    async def capture_app(scope, receive, send) -> None:
        assert scope["type"] == "http"
        chunks = b""
        more = True
        while more:
            message = await receive()
            chunks += message.get("body", b"")
            more = message.get("more_body", False)
        captured.append(_json.loads(chunks or b"{}"))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok":true}'})

    store = InMemoryRunStore()
    worker = RunWorker(_agent("hooked"), store, webhook="http://hook.local/hook")
    invoked: list[tuple[str, dict]] = []

    # Route the worker's POST through an in-process ASGI transport — no socket,
    # no port — so the callback is captured deterministically in the same loop.
    async def in_process_post(url: str, body: dict) -> None:
        invoked.append((url, body))
        transport = httpx.ASGITransport(app=capture_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://hook.local") as client:
            await client.post("/hook", json=body)

    worker._post_webhook = in_process_post  # type: ignore[assignment]

    async def go() -> None:
        await store.create(_record("r1"))
        await _drain_one(worker, store)

    asyncio.run(go())
    # The worker fired the callback with the terminal status...
    assert invoked, "worker never invoked the webhook"
    assert invoked[0][0] == "http://hook.local/hook"
    # ...and the in-process ASGI app received it.
    assert captured, "webhook never fired"
    body = captured[0]
    assert body["run_id"] == "r1"
    assert body["status"] == "completed"
    assert body["output"] == "hooked"


# --- per-run webhook override beats the worker default ----------------
class _RecordWithWebhook(RunRecord):
    """A run record carrying a per-run webhook override.

    The worker reads the override defensively via ``getattr`` so a deployment
    can attach it without the base record needing the field; this subclass
    stands in for that field-bearing record in tests.
    """

    webhook: str | None = None


def test_per_run_webhook_override():
    captured: list[str] = []

    async def fake_post(url: str, body: dict) -> None:
        captured.append(url)

    store = InMemoryRunStore()
    worker = RunWorker(_agent(), store, webhook="http://default/hook")
    worker._post_webhook = fake_post  # type: ignore[assignment]

    async def go() -> None:
        now = time.time()
        rec = _RecordWithWebhook(
            run_id="r1",
            agent="svc",
            status=RunStatus.QUEUED,
            prompt="hi",
            background=True,
            resume_id="r1",
            webhook="http://per-run/hook",
            created_at=now,
            updated_at=now,
        )
        await store.create(rec)
        await _drain_one(worker, store)

    asyncio.run(go())
    assert captured == ["http://per-run/hook"]


# --- no webhook configured: nothing is posted, run still completes ----
def test_no_webhook_is_silent():
    posts: list[str] = []
    store = InMemoryRunStore()
    worker = RunWorker(_agent("quiet"), store)

    async def fake_post(url: str, body: dict) -> None:
        posts.append(url)

    worker._post_webhook = fake_post  # type: ignore[assignment]

    async def go() -> None:
        await store.create(_record("r1"))
        final = await _drain_one(worker, store)
        assert final.status is RunStatus.COMPLETED

    asyncio.run(go())
    assert posts == []
