"""Background worker — durable runs that drain a queue and survive a restart.

A submitted background run stops being a fleeting in-process task and becomes a
durable queued row. The :class:`RunWorker` drains that queue with a hard ceiling
on how many runs execute at once, leases each run it picks up, heartbeats the
lease while the run is in flight, records the terminal outcome on the durable
record, and optionally posts a completion callback so callers need not poll.

Three properties make this safe to run as a fleet behind a load balancer:

* **Bounded concurrency.** A semaphore caps in-flight runs, so a thousand
  submissions enqueue a thousand rows but never spawn a thousand tasks. Queue
  depth is the natural backpressure signal.
* **Crash and rolling-deploy survival.** Each running row holds a lease the
  worker refreshes; if a replica dies mid-run its lease expires and the reaper
  (on any replica) re-queues the run for another worker to finish — resuming
  from its last checkpoint when a resumable run store is configured.
* **Eviction-on-pause.** When a run parks for out-of-band human sign-off, the
  worker releases the lease and frees the slot, so a paused run consumes zero
  worker capacity and can resume on any replica.

The same worker also materializes due schedules (see :mod:`yaab.runs.cron`) into
queued runs, reusing the one run-creation path.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import Any

from ..exceptions import ApprovalRequired, RunCancelled
from ..types import Event, EventType
from .base import RunRecord, RunStatus, RunStore
from .cancel import StoreCancellationToken

# Terminal statuses that fire a completion webhook.
_WEBHOOK_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})


class RunWorker:
    """Drain a durable run queue with bounded concurrency, leases, and webhooks.

    Args:
        agent: The agent to execute for each claimed run. (A single-agent worker;
            multi-agent routing can be layered by inspecting ``record.agent``.)
        store: The durable run store to claim from and report outcomes to.
        runner: The orchestration engine to run the agent. Defaults to a fresh
            :class:`~yaab.runner.Runner`. Any object exposing an async
            ``run_stream(agent, prompt, **kwargs)`` works (so tests can inject a
            stub).
        max_concurrency: The most runs this worker executes at once. Submissions
            beyond the cap wait in the queue (backpressure), never as live tasks.
        lease_seconds: How long a claim's lease lasts before it is considered
            abandoned and reapable.
        pod_id: This worker's identity in the fleet (the lease holder). Defaults
            to a random id.
        webhook: Default URL to POST a run's terminal status to. A run may carry
            its own ``webhook`` to override this.
        heartbeat_interval: Seconds between lease renewals while a run is in
            flight. Defaults to a third of ``lease_seconds`` so a lease never
            lapses mid-run.
        poll_interval: Seconds to wait before re-polling an empty queue.
        cron_store: Optional schedule store; :meth:`cron_tick` materializes due
            schedules into queued runs.
        reaper_interval: Seconds between reaper sweeps in :meth:`run_forever`.
            ``None`` (default ``lease_seconds``) keeps crash recovery automatic.
        cron_interval: Seconds between schedule ticks in :meth:`run_forever`.
    """

    def __init__(
        self,
        agent: Any,
        store: RunStore,
        *,
        runner: Any | None = None,
        max_concurrency: int = 10,
        lease_seconds: float = 30.0,
        pod_id: str | None = None,
        webhook: str | None = None,
        heartbeat_interval: float | None = None,
        poll_interval: float = 0.1,
        cron_store: Any | None = None,
        reaper_interval: float | None = None,
        cron_interval: float = 1.0,
    ) -> None:
        self.agent = agent
        self.store = store
        if runner is None:
            from ..runner import Runner

            runner = Runner()
        self.runner = runner
        self.max_concurrency = max_concurrency
        self.lease_seconds = lease_seconds
        self.pod_id = pod_id or f"pod-{uuid.uuid4().hex[:8]}"
        self.webhook = webhook
        self.heartbeat_interval = (
            heartbeat_interval if heartbeat_interval is not None else lease_seconds / 3.0
        )
        self.poll_interval = poll_interval
        self.cron_store = cron_store
        self.reaper_interval = reaper_interval if reaper_interval is not None else lease_seconds
        self.cron_interval = cron_interval

        self._sem = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------
    @property
    def in_flight(self) -> int:
        """How many runs are currently executing (live ``_execute`` tasks)."""
        return len(self._tasks)

    def stop(self) -> None:
        """Ask the claim loop to exit after the current poll."""
        self._stop.set()

    # ------------------------------------------------------------------
    async def run_forever(self) -> None:
        """Claim and execute queued runs until :meth:`stop` is called.

        The loop is bounded by a semaphore: it acquires a slot *before* claiming,
        so at most ``max_concurrency`` runs are ever claimed-and-in-flight. An
        empty queue releases the slot and waits ``poll_interval``; a claimed run
        is handed to a background ``_execute`` task that releases the slot when
        it finishes. The reaper and schedule ticks are interleaved on their own
        cadence so a single worker also recovers crashed runs and fires crons.
        """
        last_reap = 0.0
        last_cron = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_reap >= self.reaper_interval:
                last_reap = now
                with contextlib.suppress(Exception):
                    await self.reap()
            if self.cron_store is not None and now - last_cron >= self.cron_interval:
                last_cron = now
                with contextlib.suppress(Exception):
                    await self.cron_tick()

            # Acquire a slot up front so we never claim past the concurrency cap.
            await self._sem.acquire()
            if self._stop.is_set():
                self._sem.release()
                break
            record = await self.store.claim_next(
                pod_id=self.pod_id, lease_seconds=self.lease_seconds
            )
            if record is None:
                # Nothing to do — give the slot back and wait before re-polling.
                self._sem.release()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                continue

            task = asyncio.create_task(self._run_and_release(record))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        # Drain in-flight runs so a clean stop does not abandon active leases.
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _run_and_release(self, record: RunRecord) -> None:
        """Execute one run and always return its concurrency slot."""
        try:
            await self._execute(record)
        finally:
            self._sem.release()

    # ------------------------------------------------------------------
    async def _execute(self, record: RunRecord) -> None:
        """Run one claimed record to a terminal (or paused) durable outcome.

        Builds a store-backed cancellation token so a cancel issued on any
        replica stops this run, keeps the lease fresh with a heartbeat task,
        and folds the runner's event stream into a single outcome: a final
        result completes the record, a cancel marks it cancelled, any other
        error marks it failed, and an approval request pauses it and frees the
        slot (eviction-on-pause).
        """
        run_id = record.run_id
        token = StoreCancellationToken(run_id, self.store, poll_interval=0.0)
        heartbeat = asyncio.create_task(self._heartbeat_loop(run_id))

        result_event: Event | None = None
        error: BaseException | None = None
        paused_payload: dict[str, Any] | None = None
        try:
            async for event in self.runner.run_stream(
                self.agent,
                record.prompt,
                session_id=record.session_id,
                identity=record.identity,
                cancellation=token,
                resume_id=record.resume_id or run_id,
            ):
                if event.type is EventType.APPROVAL_REQUIRED:
                    paused_payload = dict(event.payload)
                    break
                if event.type is EventType.RUN_END:
                    result_event = event
                elif event.type is EventType.ERROR:
                    error = event.payload.get("error")
        except RunCancelled as exc:
            error = exc
        except ApprovalRequired as exc:
            # A runner that raises (no checkpointer wired) still pauses durably.
            paused_payload = {
                "tool": getattr(exc, "tool", None),
                "arguments": getattr(exc, "arguments", {}),
                "approval_id": getattr(exc, "approval_id", None),
            }
        except Exception as exc:  # noqa: BLE001 - record any failure on the run
            error = exc
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

        if paused_payload is not None:
            # Eviction-on-pause: drop the lease so the slot frees and the run can
            # resume on any replica once a reviewer decides.
            await self.store.update(
                run_id,
                status=RunStatus.PAUSED,
                owner_pod=None,
                lease_expires_at=None,
            )
            return

        await self._finalize(record, result_event, error)

    async def _finalize(
        self, record: RunRecord, result_event: Event | None, error: BaseException | None
    ) -> None:
        """Write the terminal record and fire the completion webhook."""
        run_id = record.run_id
        now = time.time()
        if error is not None:
            status = RunStatus.CANCELLED if isinstance(error, RunCancelled) else RunStatus.FAILED
            fields: dict[str, Any] = {
                "status": status,
                "error": str(error),
                "owner_pod": None,
                "lease_expires_at": None,
                "finished_at": now,
            }
        elif result_event is not None:
            result = result_event.payload.get("result")
            output = _result_output(result)
            usage = _result_usage(result)
            fields = {
                "status": RunStatus.COMPLETED,
                "output": output,
                "usage": usage,
                "owner_pod": None,
                "lease_expires_at": None,
                "finished_at": now,
            }
        else:
            # The stream ended without a terminal event — treat as a failure so
            # the run never lingers as "running" forever.
            fields = {
                "status": RunStatus.FAILED,
                "error": "run produced no terminal result",
                "owner_pod": None,
                "lease_expires_at": None,
                "finished_at": now,
            }

        updated = await self.store.update(run_id, **fields)
        await self._maybe_webhook(record, updated, fields["status"])

    # ------------------------------------------------------------------
    async def _heartbeat_loop(self, run_id: str) -> None:
        """Refresh the lease on a cadence until the run finishes."""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                await self.store.heartbeat(
                    run_id, pod_id=self.pod_id, lease_seconds=self.lease_seconds
                )
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    async def reap(self) -> list[str]:
        """Re-queue runs abandoned by a crashed replica (expired leases)."""
        return await self.store.reap_expired_leases()

    # ------------------------------------------------------------------
    async def cron_tick(self, *, now: float | None = None) -> list[str]:
        """Materialize every due schedule into exactly one queued run.

        Returns the ids of the runs created. A schedule is rolled forward as it
        fires, so a tick is idempotent for a given moment: the same schedule does
        not re-fire until its next window arrives.
        """
        if self.cron_store is None:
            return []
        moment = time.time() if now is None else now
        created: list[str] = []
        for cron in await self.cron_store.due(now=moment):
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            record = RunRecord(
                run_id=run_id,
                agent=cron.agent,
                status=RunStatus.QUEUED,
                prompt=cron.prompt,
                session_id=cron.session_id,
                identity=cron.identity,
                background=True,
                resume_id=run_id,
                created_at=moment,
                updated_at=moment,
            )
            await self.store.create(record)
            # Roll the schedule forward only after the run is durably queued, so a
            # crash between the two leaves the schedule due (at-least-once), never
            # silently skipped.
            await self.cron_store.mark_run(cron.cron_id, now=moment)
            created.append(run_id)
        return created

    # ------------------------------------------------------------------
    async def _maybe_webhook(
        self, record: RunRecord, updated: RunRecord | None, status: RunStatus
    ) -> None:
        """POST the terminal status to the run's webhook, if one is configured."""
        if status not in _WEBHOOK_STATUSES:
            return
        url = getattr(record, "webhook", None) or self.webhook
        if not url:
            return
        final = updated if updated is not None else record
        body = {
            "run_id": record.run_id,
            "status": status.value,
            "agent": record.agent,
            "output": getattr(final, "output", None),
            "usage": getattr(final, "usage", None),
            "error": getattr(final, "error", None),
        }
        with contextlib.suppress(Exception):
            await self._post_webhook(url, body)

    async def _post_webhook(self, url: str, body: dict[str, Any]) -> None:
        """POST ``body`` as JSON to ``url`` (lazy ``httpx`` import).

        Isolated so tests can replace it with an in-process capture; failures are
        swallowed by the caller so a flaky callback never fails a finished run.
        """
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json=body)


def _result_output(result: Any) -> Any:
    """Pull the JSON-safe output off a RunResult-like object."""
    if result is None:
        return None
    output: Any = getattr(result, "output", None)
    dump = getattr(output, "model_dump", None)
    if callable(dump):
        return dump()
    return output


def _result_usage(result: Any) -> dict[str, Any] | None:
    """Pull the usage dict off a RunResult-like object."""
    if result is None:
        return None
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return None


__all__ = ["RunWorker"]
