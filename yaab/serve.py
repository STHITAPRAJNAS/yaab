"""Serve agents over HTTP — a FastAPI app and an A2A-compatible server.

``fastapi_server_app(agent)`` returns a ready-to-mount ASGI app exposing:

* ``GET  /.well-known/agent.json`` — the A2A Agent Card (discovery);
* ``POST /run``                    — run the agent (YAAB-native; ``background``
  submits it as a task and returns ``202`` immediately);
* ``GET  /runs`` / ``GET /runs/{id}`` / ``POST /runs/{id}/cancel`` — the run
  lifecycle API (poll status, list, remotely cancel an in-flight run);
* ``POST /a2a/tasks``              — A2A task submission (agent-to-agent);
* ``GET  /health``                 — liveness.

With no extra wiring the app behaves exactly as before: an in-process registry
holds runs, a background submission fires a task, and only an in-memory view is
available. Passing durable backends makes the same endpoints production-grade:

* ``run_store`` turns a background run into a durable queued row drained by an
  in-process :class:`~yaab.runs.worker.RunWorker`, so the run survives a restart
  and is visible (and cancellable) from any replica;
* ``approval_store`` adds out-of-band human sign-off — list, approve, deny, and
  resume parked runs over HTTP;
* ``trace_store`` persists each run's per-step timeline so a debugger can replay
  it with model/tool/token/cost/latency detail (``/runs/{id}/events`` and
  ``/runs/{id}/trace``);
* ``run_checkpointer`` makes background runs fault-tolerant (resume from the last
  step after a crash) and is the seam approvals resume through;
* ``cron_store`` adds durable schedules (``/crons``).

Every new parameter defaults to ``None`` (today's behavior, byte-for-byte). The
new endpoints return a clean ``404`` when their backing store is not configured.

Authentication is pluggable via :mod:`yaab.auth`; the resolved identity flows
into the run context and the audit log. FastAPI is an optional dependency,
imported lazily so importing YAAB never requires a web stack.
"""

import time
import uuid
from typing import Any

from .auth import AuthError, AuthScheme, NoAuth
from .exceptions import RunCancelled
from .limits import CancellationToken

# In-process store of submitted A2A tasks, so clients can poll by id. A durable
# deployment would back this with the session/artifact services.
_A2A_TASKS: dict[str, dict] = {}

# In-process registry of agent runs (sync and background) so clients can poll
# status, list, and remotely cancel. Each entry is a mutable dict; see
# ``_register_run``. A durable deployment would persist this. ``insertion order``
# is preserved by dict so listing newest-first is just a reversed view.
_RUN_REGISTRY: dict[str, dict[str, Any]] = {}

#: Cap on retained *finished* runs. Active runs are never evicted; once we
#: exceed the cap we drop the oldest finished entries (FIFO) so a long-lived
#: server doesn't grow without bound.
_MAX_FINISHED_RUNS = 1000

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


def _register_run(token: CancellationToken) -> tuple[str, dict[str, Any]]:
    """Create and store a fresh 'running' registry entry, returning (id, entry)."""
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    entry: dict[str, Any] = {
        "id": run_id,
        "status": "running",
        "token": token,
        "result": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }
    _RUN_REGISTRY[run_id] = entry
    return run_id, entry


def _finish_run(
    entry: dict[str, Any], *, status: str, result: Any = None, error: Any = None
) -> None:
    """Transition a registry entry to a terminal state and evict stale runs.

    A run already cancelled (via the API) stays 'cancelled' even if the
    underlying coroutine then surfaces a different error — the user-visible
    intent wins. We still record the error text for observability.
    """
    if entry.get("_api_cancelled") and status == "failed":
        status = "cancelled"
    entry["status"] = status
    entry["result"] = result
    entry["error"] = error
    entry["finished_at"] = time.time()
    _evict_finished()


def _evict_finished() -> None:
    """Drop the oldest finished runs once the retained-finished cap is exceeded."""
    finished = [rid for rid, e in _RUN_REGISTRY.items() if e["status"] in _TERMINAL_STATES]
    overflow = len(finished) - _MAX_FINISHED_RUNS
    for rid in finished[:overflow] if overflow > 0 else ():
        _RUN_REGISTRY.pop(rid, None)


def _run_view(entry: dict[str, Any]) -> dict[str, Any]:
    """Serialize a registry entry into a JSON-safe status document.

    Includes ``output``/``usage`` when the run completed, or ``error`` when it
    failed/was cancelled. The live :class:`CancellationToken` is never exposed.
    """
    view: dict[str, Any] = {
        "run_id": entry["id"],
        "status": entry["status"],
        "started_at": entry["started_at"],
        "finished_at": entry["finished_at"],
    }
    result = entry.get("result")
    if result is not None:
        from .runner import _safe

        view["output"] = _safe(result.output)
        view["usage"] = result.usage.model_dump()
    if entry.get("error") is not None:
        view["error"] = str(entry["error"])
    return view


# ---------------------------------------------------------------------------
# Durable run-store views and span/waterfall computation.
# ---------------------------------------------------------------------------
def _record_view(record: Any) -> dict[str, Any]:
    """Serialize a durable :class:`RunRecord` into a JSON-safe status document.

    Mirrors :func:`_run_view`'s shape (``run_id``/``status``/``output``/
    ``usage``/``error``) so a caller polling ``GET /runs/{id}`` sees the same
    document whether the run lives in the in-process registry or the durable
    store. The ``status`` is the record's enum value (``queued``/``running``/
    ``paused``/``completed``/``failed``/``cancelled``).
    """
    status = record.status.value if hasattr(record.status, "value") else str(record.status)
    view: dict[str, Any] = {
        "run_id": record.run_id,
        "status": status,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }
    if record.output is not None:
        view["output"] = record.output
    if record.usage is not None:
        view["usage"] = record.usage
    if record.error is not None:
        view["error"] = record.error
    return view


def _record_list_item(record: Any) -> dict[str, Any]:
    """The compact ``{id, status, started_at}`` row used by ``GET /runs``."""
    status = record.status.value if hasattr(record.status, "value") else str(record.status)
    return {"id": record.run_id, "status": status, "started_at": record.created_at}


def _validate_webhook(url: str | None) -> str | None:
    """Reject webhook URLs that could drive a server-side request forgery.

    A webhook is fetched server-side by the worker, so an attacker-controlled URL
    is an SSRF vector: it could probe internal services or hit a cloud metadata
    endpoint. We require ``https`` and refuse loopback, link-local (incl. the
    169.254 metadata range), and RFC1918 private hosts. ``None``/empty passes
    through (no webhook). Raises ``ValueError`` on a rejected URL.
    """
    if not url:
        return url
    import ipaddress
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("webhook must use https")
    host = parsed.hostname
    if not host:
        raise ValueError("webhook has no host")
    lowered = host.lower()
    if lowered in ("localhost", "localhost.localdomain") or lowered.endswith(".localhost"):
        raise ValueError("webhook host is not allowed")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (
        ip.is_loopback
        or ip.is_link_local  # covers 169.254.0.0/16 (cloud metadata) and fe80::/10
        or ip.is_private
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise ValueError("webhook host is not allowed")
    return url


#: Substrings (case-insensitive) of argument names whose values are masked in a
#: persisted trace before it is returned over HTTP. Secrets should never travel
#: as tool arguments, but a trace that captured one must not leak it to a reader.
_SENSITIVE_ARG_HINTS = (
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "token",
    "auth",
    "credential",
    "private_key",
    "access_key",
    "session_key",
)
_REDACTED = "***redacted***"


def _is_sensitive_arg(name: str) -> bool:
    lowered = str(name).lower()
    return any(hint in lowered for hint in _SENSITIVE_ARG_HINTS)


def _redact_arguments(value: Any) -> Any:
    """Recursively mask values of sensitive-looking keys in tool arguments.

    Keys like ``api_key``/``password``/``token``/``secret`` have their values
    replaced with a redaction marker; the structure is otherwise preserved so a
    debugger still sees which arguments were passed, just not their secrets.
    """
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _is_sensitive_arg(k) else _redact_arguments(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_arguments(v) for v in value]
    return value


def _redact_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of the trace with tool-call/approval arguments redacted."""
    redacted: list[dict[str, Any]] = []
    for ev in events:
        payload = ev.get("payload")
        if isinstance(payload, dict) and "arguments" in payload:
            new_payload = {**payload, "arguments": _redact_arguments(payload["arguments"])}
            ev = {**ev, "payload": new_payload}
        redacted.append(ev)
    return redacted


def _compute_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute a span/waterfall + rollups from a run's persisted event trace.

    Each event is the JSON-safe shape the runner's trace store records:
    ``{type, agent, run_id, seq, timestamp, duration_ms, payload}``. We fold the
    model/tool/transfer/approval events into ordered spans carrying their
    duration and, for model spans, the model name, finish reason, token counts,
    and cost — then roll the totals up so the console can render per-run latency,
    tokens, and cost without re-deriving them on the client.
    """
    spans: list[dict[str, Any]] = []
    total_input = total_output = total_tokens = total_cached = 0
    total_cost = 0.0
    total_latency_ms = 0.0
    model_rollup: dict[str, dict[str, Any]] = {}
    tool_rollup: dict[str, dict[str, Any]] = {}

    for ev in events:
        etype = ev.get("type")
        payload = ev.get("payload") or {}
        duration = ev.get("duration_ms")
        start = ev.get("timestamp")

        if etype == "model_response":
            usage = payload.get("usage") or {}
            inp = int(usage.get("input_tokens", 0) or 0)
            out = int(usage.get("output_tokens", 0) or 0)
            tot = int(usage.get("total_tokens", 0) or 0)
            cached = int(usage.get("cached_input_tokens", 0) or 0)
            cost = float(usage.get("cost_usd", 0.0) or 0.0)
            model = payload.get("model")
            span = {
                "type": "model_call",
                "start": start,
                "duration_ms": duration,
                "model": model,
                "finish_reason": payload.get("finish_reason"),
                "input_tokens": inp,
                "output_tokens": out,
                "cost_usd": cost,
            }
            spans.append(span)
            total_input += inp
            total_output += out
            total_tokens += tot
            total_cached += cached
            total_cost += cost
            if duration:
                total_latency_ms += float(duration)
            key = model or "unknown"
            roll = model_rollup.setdefault(
                key,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            roll["calls"] += 1
            roll["input_tokens"] += inp
            roll["output_tokens"] += out
            roll["cost_usd"] += cost
        elif etype == "tool_result":
            name = payload.get("name")
            spans.append(
                {
                    "type": "tool_call",
                    "start": start,
                    "duration_ms": duration,
                    "name": name,
                }
            )
            if duration:
                total_latency_ms += float(duration)
            roll = tool_rollup.setdefault(name or "unknown", {"calls": 0, "duration_ms": 0.0})
            roll["calls"] += 1
            if duration:
                roll["duration_ms"] += float(duration)
        elif etype == "agent_transfer":
            spans.append(
                {
                    "type": "transfer",
                    "start": start,
                    "duration_ms": duration,
                    "to": payload.get("to"),
                }
            )
        elif etype == "approval_required":
            spans.append(
                {
                    "type": "approval",
                    "start": start,
                    "duration_ms": duration,
                    "tool": payload.get("tool"),
                    "approval_id": payload.get("approval_id"),
                }
            )
        elif etype == "run_end":
            # Prefer the authoritative run-level usage when the RUN_END event
            # carries it (it is the aggregate, not a per-call delta).
            result = payload.get("result") or {}
            usage = result.get("usage") if isinstance(result, dict) else None
            if isinstance(usage, dict):
                total_input = int(usage.get("input_tokens", total_input) or total_input)
                total_output = int(usage.get("output_tokens", total_output) or total_output)
                total_tokens = int(usage.get("total_tokens", total_tokens) or total_tokens)
                total_cached = int(usage.get("cached_input_tokens", total_cached) or total_cached)
                total_cost = float(usage.get("cost_usd", total_cost) or total_cost)
            if duration:
                total_latency_ms = float(duration)

    return {
        "spans": spans,
        "totals": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "cached_input_tokens": total_cached,
            "cost_usd": total_cost,
            "latency_ms": total_latency_ms,
        },
        "models": model_rollup,
        "tools": tool_rollup,
    }


def _default_checkpointer_for(run_store: Any) -> Any:
    """Derive a run checkpointer from a durable run store's backend, if any.

    A durable run store implies the deployment wants background runs to be
    fault-tolerant, so we point the runner's checkpointer at the same kind of
    backend (SQLite/Postgres/Redis). The in-memory store keeps the classic
    zero-overhead fast path (returns ``None``). Best-effort: any import/parameter
    mismatch falls back to ``None`` so wiring a checkpointer never blocks serving.
    """
    name = type(run_store).__name__
    try:
        if name == "SQLiteRunStore":
            from .graph.checkpoint import SQLiteSaver

            return SQLiteSaver()
        if name in ("PostgresRunStore", "RedisRunStore"):
            from .graph.checkpoint import MemorySaver

            # A shared durable checkpointer for these backends is deployment
            # specific; default to an in-process saver so the resume seam still
            # works within a process. Callers pass an explicit one for HA.
            return MemorySaver()
    except Exception:  # noqa: BLE001 - never let checkpointer wiring break serving
        return None
    return None


def _webhook_record_cls() -> Any:
    """Build (once) a ``RunRecord`` subclass that carries a per-run webhook field.

    The base record intentionally omits ``webhook`` (the worker reads it
    defensively via ``getattr``); this subclass lets a background submission
    attach a per-run callback URL that the in-process worker can fire on a
    terminal status.
    """
    cls = getattr(_webhook_record_cls, "_cls", None)
    if cls is not None:
        return cls
    from .runs.base import RunRecord

    class RunRecordWithWebhook(RunRecord):
        webhook: str | None = None

    _webhook_record_cls._cls = RunRecordWithWebhook  # type: ignore[attr-defined]
    return RunRecordWithWebhook


def fastapi_server_app(
    agent: Any,
    *,
    runner: Any | None = None,
    auth: AuthScheme | None = None,
    base_url: str = "",
    run_store: Any | None = None,
    approval_store: Any | None = None,
    trace_store: Any | None = None,
    run_checkpointer: Any | None = None,
    cron_store: Any | None = None,
    worker: Any | None = None,
) -> Any:
    """Build a FastAPI app that serves ``agent`` (YAAB-native + A2A endpoints).

    Args:
        agent: The agent to serve.
        runner: An explicit :class:`~yaab.runner.Runner`; one is derived from the
            agent when omitted.
        auth: Pluggable auth scheme (defaults to no auth).
        base_url: Public base URL advertised on the agent card.
        run_store: A durable run store. When set, a background ``POST /run``
            becomes a durable queued row drained by an in-process worker, and the
            run lifecycle endpoints read/cancel through the store so a run
            survives a restart and is visible across replicas. ``None`` keeps the
            in-process registry (today's behavior).
        approval_store: A durable approval store enabling the out-of-band
            sign-off endpoints (``/approvals*`` and ``/runs/{id}/resume``). The
            endpoints are absent/``404`` without it.
        trace_store: A durable per-run trace store enabling the history/trace
            endpoints (``/runs/{id}/events`` and ``/runs/{id}/trace``).
        run_checkpointer: A checkpointer that makes background runs
            fault-tolerant and is the seam approvals resume through. When ``None``
            and a durable ``run_store`` is configured, one is derived from the
            store's backend; the bare in-memory path stays zero-overhead.
        cron_store: A durable schedule store enabling ``/crons`` and the worker's
            schedule ticks.
        worker: Controls the in-process queue worker. ``None`` (default) builds
            one from the agent/store when a ``run_store`` is set. An explicit
            :class:`~yaab.runs.worker.RunWorker` is adopted as-is. **Pass
            ``worker=False`` to disable the embedded worker** — required for
            multi-replica deployments: run ONE external worker process (a
            standalone ``RunWorker.run_forever()`` loop, not embedded in each API
            replica) so several replicas don't each claim from the same queue.
            When a ``run_store`` is configured and the embedded worker is left
            enabled, this app is single-worker by design; scale the API tier
            horizontally only with ``worker=False`` plus a dedicated worker.
    """
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "FastAPI is required to serve agents. Install with `pip install fastapi uvicorn`."
        ) from exc

    auth_scheme = auth or NoAuth()

    # When a durable run store is configured, default the runner's checkpointer
    # from the store's backend (unless one is passed) so background runs are
    # fault-tolerant out of the box; the resume seam for approvals rides on it.
    effective_checkpointer = run_checkpointer
    if effective_checkpointer is None and run_store is not None:
        effective_checkpointer = _default_checkpointer_for(run_store)

    # Build (or adopt) the runner so background runs and the in-process worker
    # share one engine wired with the checkpointer and trace store. The classic
    # paths (sync /run, streams) keep using the agent's own runner when nothing
    # durable is configured, preserving today's behavior byte-for-byte.
    served_runner = runner
    if served_runner is None and (
        run_store is not None or trace_store is not None or effective_checkpointer is not None
    ):
        from .runner import Runner

        served_runner = Runner(
            run_checkpointer=effective_checkpointer,
            trace_store=trace_store,
        )

    # The in-process worker that drains the durable queue (background runs and
    # resumed approvals). Started/stopped by the app lifespan. ``worker=False``
    # disables the embedded worker so a multi-replica deployment can run a single
    # external worker instead of one per API replica.
    served_worker = None if worker is False else worker
    if served_worker is None and worker is not False and run_store is not None:
        from .runs.worker import RunWorker

        served_worker = RunWorker(
            agent,
            run_store,
            runner=served_runner,
            cron_store=cron_store,
            approval_store=approval_store,
        )

    @asynccontextmanager
    async def lifespan(_app: Any) -> Any:
        """Start the in-process queue worker for the lifetime of the app.

        Mirrors the prior behavior for the in-memory path (a background run still
        completes while a ``TestClient`` context is open), now routed through the
        durable queue when a ``run_store`` is configured.
        """
        from .runs.safety import warn_if_ephemeral

        warn_if_ephemeral(
            run_store=run_store,
            approval_store=approval_store,
            trace_store=trace_store,
        )
        task = None
        if served_worker is not None:
            import asyncio

            task = asyncio.create_task(served_worker.run_forever())
        try:
            yield
        finally:
            if served_worker is not None:
                served_worker.stop()
            if task is not None:
                import asyncio
                import contextlib

                # Give in-flight runs room to checkpoint/finalize on a clean
                # shutdown rather than abandoning leases. The grace period scales
                # with the lease/heartbeat cadence so an active run can drain
                # within it; anything still running past the grace is left for the
                # reaper to re-queue after its lease lapses.
                lease = float(getattr(served_worker, "lease_seconds", 30.0))
                heartbeat = float(getattr(served_worker, "heartbeat_interval", lease / 3.0))
                grace = max(15.0, lease + heartbeat)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(task, timeout=grace)

    app = FastAPI(
        title=f"YAAB · {agent.name}",
        version=getattr(agent, "version", "0.1.0"),
        lifespan=lifespan,
    )

    def _identify(request: Request) -> str:
        try:
            return auth_scheme.authenticate(dict(request.headers)) or "anonymous"
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def _owns_record(record: Any, identity: str) -> bool:
        """True if ``identity`` may access ``record`` (cross-tenant guard).

        A run records the identity that submitted it. A caller may read a run
        only when it carries no identity (legacy/anonymous, open by design) or it
        matches the caller. This blocks identity A from reading identity B's run,
        trace, events, or session state by guessing an id.
        """
        owner = getattr(record, "identity", None)
        return owner is None or owner == identity

    async def _authorized_record(run_id: str, identity: str) -> Any | None:
        """Fetch a run record, enforcing the cross-tenant access guard.

        Returns the record when the caller owns it, ``None`` when there is no such
        run, and raises ``403`` when the run exists but belongs to someone else —
        so a probe can't distinguish "not yours" by reading another's data.
        """
        if run_store is None:
            return None
        record = await run_store.get(run_id)
        if record is None:
            return None
        if not _owns_record(record, identity):
            raise HTTPException(status_code=403, detail="forbidden")
        return record

    def _card() -> dict:
        from .governance.registry import AgentCard

        card = AgentCard(agent_id=agent.registry_id or agent.name, name=agent.name)
        body = card.to_a2a_card(url=base_url)
        body["securitySchemes"] = {auth_scheme.name: auth_scheme.describe()}
        return body

    @app.get("/health")
    async def health() -> dict:
        from . import __version__

        return {"status": "ok", "agent": agent.name, "yaab": __version__}

    @app.get("/.well-known/agent.json")
    async def agent_card() -> dict:
        return _card()

    async def _invoke(
        prompt: str, session_id: str | None, identity: str, token: CancellationToken
    ) -> Any:
        """Run the agent under a cancellation token (shared by sync + background)."""
        return await agent.run(
            prompt,
            session_id=session_id,
            identity=identity,
            cancellation=token,
        )

    async def _enqueue_background(
        prompt: str,
        session_id: str | None,
        identity: str,
        *,
        webhook: str | None,
    ) -> str:
        """Create a durable QUEUED run row; the in-process worker drains it.

        Returns the new ``run_id``. ``resume_id`` is the run id so a reaped or
        approval-paused run resumes from its last checkpoint.
        """
        from .runs.base import RunStatus

        assert run_store is not None  # only called from a run_store-guarded path
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = time.time()
        record_cls = _webhook_record_cls() if webhook else None
        if record_cls is not None:
            record = record_cls(
                run_id=run_id,
                agent=agent.name,
                status=RunStatus.QUEUED,
                prompt=prompt,
                session_id=session_id,
                identity=identity,
                background=True,
                resume_id=run_id,
                webhook=webhook,
                created_at=now,
                updated_at=now,
            )
        else:
            from .runs.base import RunRecord

            record = RunRecord(
                run_id=run_id,
                agent=agent.name,
                status=RunStatus.QUEUED,
                prompt=prompt,
                session_id=session_id,
                identity=identity,
                background=True,
                resume_id=run_id,
                created_at=now,
                updated_at=now,
            )
        await run_store.create(record)
        return run_id

    async def _active_session_run(session_id: str) -> Any | None:
        """Return a non-terminal run for ``session_id`` in the durable store, if any."""
        from .runs.base import RunStatus

        assert run_store is not None  # only called from a run_store-guarded path
        for status in (RunStatus.RUNNING, RunStatus.QUEUED, RunStatus.PAUSED):
            for rec in await run_store.list(status=status):
                if rec.session_id == session_id:
                    return rec
        return None

    @app.post("/run")
    async def run(request: Request) -> Any:
        identity = _identify(request)
        body = await request.json()
        prompt = body.get("prompt") or body.get("input") or ""
        session_id = body.get("session_id")

        # Background submission. With a durable run store, enqueue a queued row
        # the worker drains (survives restart, visible across replicas); without
        # one, keep the classic in-process fire-and-poll task.
        if body.get("background"):
            # SSRF guard: a webhook is fetched server-side by the worker, so a
            # caller-supplied URL must be validated before it is stored/fired.
            try:
                webhook = _validate_webhook(body.get("webhook"))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if run_store is not None:
                # multitask_strategy guards a session with an already-active run.
                strategy = body.get("multitask_strategy", "enqueue")
                if session_id is not None and strategy in ("reject", "cancel"):
                    active = await _active_session_run(session_id)
                    if active is not None:
                        if strategy == "reject":
                            raise HTTPException(
                                status_code=409,
                                detail=f"session {session_id} already has an active run",
                            )
                        # cancel: ask the active run to stop, then enqueue this one.
                        await run_store.request_cancel(active.run_id)
                run_id = await _enqueue_background(prompt, session_id, identity, webhook=webhook)
                return JSONResponse({"run_id": run_id, "status": "queued"}, status_code=202)

            import asyncio

            token = CancellationToken()
            api_run_id, entry = _register_run(token)

            async def _background() -> None:
                try:
                    result = await _invoke(prompt, session_id, identity, token)
                    _finish_run(entry, status="completed", result=result)
                except Exception as exc:  # noqa: BLE001 - record, don't crash the loop
                    status = "cancelled" if isinstance(exc, RunCancelled) else "failed"
                    _finish_run(entry, status=status, error=exc)

            entry["_task"] = asyncio.create_task(_background())
            return JSONResponse({"run_id": api_run_id, "status": "running"}, status_code=202)

        # Synchronous: keep the exact prior response shape, but still register so
        # this run is visible to /runs and cancellable mid-flight from elsewhere.
        from .runner import _safe

        token = CancellationToken()
        api_run_id, entry = _register_run(token)
        try:
            result = await _invoke(prompt, session_id, identity, token)
        except Exception as exc:  # noqa: BLE001
            status = "cancelled" if isinstance(exc, RunCancelled) else "failed"
            _finish_run(entry, status=status, error=exc)
            raise
        _finish_run(entry, status="completed", result=result)
        return JSONResponse(
            {
                "output": _safe(result.output),
                "run_id": api_run_id,
                "usage": result.usage.model_dump(),
            }
        )

    @app.get("/runs")
    async def list_runs(request: Request) -> Any:
        """List known runs (id, status, started_at), most recent first."""
        _identify(request)
        if run_store is not None:
            records = await run_store.list()
            return JSONResponse([_record_list_item(r) for r in records])
        items = [
            {"id": e["id"], "status": e["status"], "started_at": e["started_at"]}
            for e in _RUN_REGISTRY.values()
        ]
        items.reverse()  # dict preserves insertion order; reverse = newest first
        return JSONResponse(items)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> Any:
        """Status of a single run (+ output/usage or error once finished)."""
        identity = _identify(request)
        if run_store is not None:
            record = await _authorized_record(run_id, identity)
            if record is not None:
                return JSONResponse(_record_view(record))
            # Fall through to the in-process registry for sync runs not stored.
        entry = _RUN_REGISTRY.get(run_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return JSONResponse(_run_view(entry))

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request) -> Any:
        """Remotely cancel an in-flight run. No-op for an already-finished run."""
        identity = _identify(request)
        if run_store is not None:
            record = await _authorized_record(run_id, identity)
            if record is not None:
                await run_store.request_cancel(run_id)
                status = (
                    record.status.value if hasattr(record.status, "value") else str(record.status)
                )
                return JSONResponse({"run_id": run_id, "status": status})
            # Fall through to the in-process registry for sync runs.
        entry = _RUN_REGISTRY.get(run_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        if entry["status"] == "running":
            entry["_api_cancelled"] = True
            entry["token"].cancel("api_cancel")
        return JSONResponse({"run_id": run_id, "status": entry["status"]})

    # --- run history + trace + state inspector ---------------------------
    def _trace_limit(request: Request) -> int | None:
        """Parse an optional ``?limit=N`` bound on how many events to materialize."""
        raw = request.query_params.get("limit")
        if raw is None:
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="limit must be an integer") from exc
        if value < 0:
            raise HTTPException(status_code=400, detail="limit must be >= 0")
        return value

    async def _trace_events(run_id: str, limit: int | None) -> list[dict[str, Any]]:
        """Load a run's events, capping the in-memory payload when ``limit`` is set.

        A trace can grow unbounded; loading every event into memory and computing
        over it is O(events). The optional cap pages the store directly so a
        single request's payload (and the work behind it) stays bounded — large
        traces are paged, never materialized whole.
        """
        assert trace_store is not None  # callers guard on a configured store
        return await trace_store.get(run_id, limit=limit)

    @app.get("/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> Any:
        """The full persisted event trace for a run (requires a trace store)."""
        identity = _identify(request)
        if trace_store is None:
            raise HTTPException(
                status_code=404, detail="no trace store configured; enable one to persist runs"
            )
        # Cross-tenant guard: when runs are durable, only the run's owner may read
        # its trace (which carries tool arguments). 403 if it belongs to another.
        await _authorized_record(run_id, identity)
        events = await _trace_events(run_id, _trace_limit(request))
        # Mask any secret-looking tool arguments before returning the raw events.
        return JSONResponse({"run_id": run_id, "events": _redact_events(events)})

    @app.get("/runs/{run_id}/trace")
    async def run_trace(run_id: str, request: Request) -> Any:
        """A computed span/waterfall with token/cost/latency rollups for a run."""
        identity = _identify(request)
        if trace_store is None:
            raise HTTPException(
                status_code=404, detail="no trace store configured; enable one to persist runs"
            )
        await _authorized_record(run_id, identity)
        events = await _trace_events(run_id, _trace_limit(request))
        trace = _compute_trace(_redact_events(events))
        trace["run_id"] = run_id
        return JSONResponse(trace)

    async def _session_state(session_id: str) -> dict[str, Any] | None:
        """Read a session's KV state through the runner's session service."""
        engine = served_runner or agent._get_runner()
        service = getattr(engine, "session_service", None)
        if service is None:
            return None
        session = await service.get(session_id)
        if session is None:
            return None
        return dict(getattr(session, "state", {}) or {})

    async def _session_owned_by(session_id: str, identity: str) -> bool:
        """True if ``identity`` owns at least one run on ``session_id``.

        With a durable run store we can map a session back to the identities that
        ran on it and gate the state inspector accordingly. Sessions with only
        anonymous/legacy runs (no identity) stay open, matching the run guard.
        """
        if run_store is None:
            return True  # no run store to attribute the session; legacy-open
        owner_seen = False
        for rec in await run_store.list(limit=1000):
            if rec.session_id != session_id:
                continue
            if getattr(rec, "identity", None) is None or rec.identity == identity:
                return True
            owner_seen = True
        # The session has runs, but none owned by (or anonymous to) this caller.
        return not owner_seen

    @app.get("/runs/{run_id}/state")
    async def run_state(run_id: str, request: Request) -> Any:
        """The session-state snapshot for the session a run belongs to."""
        identity = _identify(request)
        session_id: str | None = None
        if run_store is not None:
            record = await _authorized_record(run_id, identity)
            if record is not None:
                session_id = record.session_id
        if session_id is None:
            raise HTTPException(status_code=404, detail=f"no session state for run {run_id}")
        state = await _session_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"no session state for run {run_id}")
        return JSONResponse({"run_id": run_id, "session_id": session_id, "state": state})

    @app.get("/sessions/{session_id}/state")
    async def session_state(session_id: str, request: Request) -> Any:
        """The KV state snapshot for a session (the state inspector)."""
        identity = _identify(request)
        # Cross-tenant guard: only a caller who owns a run on this session may
        # read its state, so a known session id can't be used to read another's.
        if not await _session_owned_by(session_id, identity):
            raise HTTPException(status_code=403, detail="forbidden")
        state = await _session_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
        return JSONResponse({"session_id": session_id, "state": state})

    # --- out-of-band human sign-off (approvals) --------------------------
    async def _resume_run(run_id: str, *, decision: str | None) -> bool:
        """Resume a paused run after a reviewer's decision, to a terminal record.

        The reviewer's decision is threaded into the resume so the runner's
        resume-from-pending branch runs the held tool (approved) or feeds the
        model a denial (denied) — without re-requesting any captured model turns.
        Resuming in-process keeps the decision correlated to the exact checkpoint
        and writes the terminal outcome back to the durable run store so any
        replica sees the completed run. Returns whether the run existed.
        """
        import asyncio

        from .runs.base import RunStatus

        assert run_store is not None  # only called from a run_store-guarded path
        record = await run_store.get(run_id)
        if record is None:
            return False
        engine = served_runner or agent._get_runner()
        if getattr(engine, "run_checkpointer", None) is None:
            # No checkpointer wired: nothing to resume from. Flip back to queued
            # so a manual re-submission can still proceed.
            await run_store.update(run_id, status=RunStatus.QUEUED)
            return True

        # The decision may land while the worker is still evicting the run to
        # PAUSED (the approval record is created a beat before the pause is
        # durable). Briefly wait for PAUSED so we never flip a RUNNING run to
        # RUNNING (which would resume from a half-written checkpoint), and never
        # silently drop a decided approval. If it never pauses we leave the
        # decision recorded; a later manual resume can still pick it up.
        claimed = None
        deadline = time.monotonic() + 5.0
        while True:
            # Guarded transition: only a run still PAUSED may be resumed, and only
            # the caller whose compare-and-set wins flips it to RUNNING. A second
            # concurrent approval/resume sees the run already RUNNING (or
            # terminal) and returns without re-executing the held tool — closing
            # the double-resume race two simultaneous decisions could open.
            claimed = await run_store.update(
                run_id,
                expect_status=RunStatus.PAUSED,
                status=RunStatus.RUNNING,
                cancel_requested=False,
                started_at=record.started_at or time.time(),
            )
            if claimed is not None:
                break
            current = await run_store.get(run_id)
            if current is None:
                return False
            status = (
                current.status.value if hasattr(current.status, "value") else str(current.status)
            )
            # Already resumed/terminal, or another resumer won — nothing to do.
            if status != "running" or time.monotonic() > deadline:
                return True
            await asyncio.sleep(0.01)
        try:
            result = await engine.run(
                agent,
                record.prompt,
                session_id=record.session_id,
                identity=record.identity,
                resume_id=record.resume_id or run_id,
                approval_decision=decision,
            )
        except Exception as exc:  # noqa: BLE001 - record the failure on the run
            await run_store.update(
                run_id,
                status=RunStatus.FAILED,
                error=str(exc),
                finished_at=time.time(),
            )
            return True
        from .runner import _safe

        await run_store.update(
            run_id,
            status=RunStatus.COMPLETED,
            output=_safe(result.output),
            usage=result.usage.model_dump(),
            finished_at=time.time(),
        )
        return True

    # Let the in-process worker reconcile a decided-but-unresumed PAUSED run by
    # re-driving it through the same idempotent resume the HTTP path uses, so a
    # decision recorded just before a crash never orphans its run (finding: a
    # decision persisted but the resume never fired).
    if served_worker is not None and approval_store is not None:
        served_worker.resume_paused = _resume_run

    @app.get("/approvals")
    async def list_approvals(request: Request) -> Any:
        """List pending approval requests (optionally scoped by status/agent)."""
        _identify(request)
        if approval_store is None:
            raise HTTPException(status_code=404, detail="no approval store configured")
        status = request.query_params.get("status", "pending")
        agent_filter = request.query_params.get("agent")
        if status != "pending":
            # Only pending is listable without a run scope; an unknown status is
            # an empty list rather than an error.
            return JSONResponse([])
        pending = await approval_store.list_pending(agent=agent_filter)
        return JSONResponse([p.model_dump(mode="json") for p in pending])

    @app.get("/approvals/{approval_id}")
    async def get_approval(approval_id: str, request: Request) -> Any:
        """One approval request (tool + arguments) for the reviewer UI."""
        _identify(request)
        if approval_store is None:
            raise HTTPException(status_code=404, detail="no approval store configured")
        req = await approval_store.get(approval_id)
        if req is None:
            raise HTTPException(status_code=404, detail=f"unknown approval {approval_id}")
        return JSONResponse(req.model_dump(mode="json"))

    def _authorize_reviewer(req: Any, identity: str, body: dict[str, Any]) -> str:
        """Enforce intent-based access control on who may decide an approval.

        When the request pins ``allowed_reviewers``, the authenticated identity
        must be among them (or hold ``required_role`` as a capability passed in
        the body's ``capabilities``). Otherwise any authenticated reviewer may
        act (today's default). Returns the reviewer name to record — the
        authenticated identity, so the audit trail can't be spoofed via the body.
        """
        allowed = list(getattr(req, "allowed_reviewers", []) or [])
        required_role = getattr(req, "required_role", None)
        if allowed or required_role is not None:
            caps = set(body.get("capabilities") or [])
            in_list = identity in allowed if allowed else False
            has_role = required_role in caps if required_role is not None else False
            if not (in_list or has_role):
                raise HTTPException(
                    status_code=403, detail="not authorized to review this approval"
                )
        # Record the authenticated identity as the reviewer when it is concrete;
        # fall back to a body-provided label only for the open/anonymous case.
        if identity and identity != "anonymous":
            return identity
        return body.get("reviewer") or "reviewer"

    async def _decide(
        approval_id: str, decision_value: str, body: dict[str, Any], identity: str
    ) -> Any:
        """Record a decision and re-enqueue the parked run for the worker."""
        from .governance.approvals import ApprovalDecision

        if approval_store is None:
            raise HTTPException(status_code=404, detail="no approval store configured")
        req = await approval_store.get(approval_id)
        if req is None:
            raise HTTPException(status_code=404, detail=f"unknown approval {approval_id}")
        reviewer = _authorize_reviewer(req, identity, body)
        reason = body.get("reason")
        decision = (
            ApprovalDecision.APPROVED if decision_value == "approved" else ApprovalDecision.DENIED
        )
        updated = await approval_store.decide(
            approval_id, decision=decision, reviewer=reviewer, reason=reason
        )
        # Resume the parked run after the decision. The durable run row is keyed
        # by the checkpoint key (``resume_id``) the loop will resume from — the
        # runner's internal ``run_id`` differs from the durable record's id, so
        # we correlate via ``resume_id`` and fall back to ``run_id``.
        if run_store is not None:
            await _resume_run(req.resume_id or req.run_id, decision=decision_value)
        return JSONResponse(updated.model_dump(mode="json"))

    @app.post("/approvals/{approval_id}/approve")
    async def approve(approval_id: str, request: Request) -> Any:
        """Approve a parked tool call; the run resumes and runs the tool."""
        identity = _identify(request)
        body = await _safe_body(request)
        return await _decide(approval_id, "approved", body, identity)

    @app.post("/approvals/{approval_id}/deny")
    async def deny(approval_id: str, request: Request) -> Any:
        """Deny a parked tool call; the run resumes with the denial fed back."""
        identity = _identify(request)
        body = await _safe_body(request)
        return await _decide(approval_id, "denied", body, identity)

    @app.post("/runs/{run_id}/resume")
    async def resume_run(run_id: str, request: Request) -> Any:
        """Idempotent manual resume of a paused run (re-enqueue for the worker)."""
        identity = _identify(request)
        if run_store is None:
            raise HTTPException(status_code=404, detail="no run store configured")
        # Cross-tenant guard: only the run's owner may resume it (403 otherwise).
        await _authorized_record(run_id, identity)
        body = await _safe_body(request)
        decision = body.get("decision")
        existed = await _resume_run(run_id, decision=decision)
        if not existed:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return JSONResponse({"run_id": run_id, "status": "queued"})

    # --- attach/join stream + crons --------------------------------------
    @app.get("/runs/{run_id}/stream")
    async def run_join_stream(run_id: str, request: Request) -> Any:
        """Re-attach to a run: replay its persisted trace, then tail live events.

        Decouples request lifetime from run lifetime — a caller can join an
        in-flight or finished background run and receive the events it already
        emitted (from the trace store) followed by a terminal marker once the run
        record reaches a terminal state. Requires a trace store.
        """
        identity = _identify(request)
        if trace_store is None:
            raise HTTPException(
                status_code=404, detail="no trace store configured; enable one to join runs"
            )
        # Cross-tenant guard: only the run's owner may tail its (redacted) trace.
        await _authorized_record(run_id, identity)

        async def event_source():
            import asyncio
            import json

            seen = 0
            deadline = time.monotonic() + 30.0
            while True:
                events = _redact_events(await trace_store.get(run_id))
                for ev in events[seen:]:
                    etype = ev.get("type", "message")
                    yield f"event: {etype}\ndata: {json.dumps(ev)}\n\n"
                seen = len(events)
                # Stop once the run is terminal (or we time out waiting).
                terminal = await _run_is_terminal(run_id)
                if terminal or time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.02)
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    async def _run_is_terminal(run_id: str) -> bool:
        """True once the durable run record has reached a terminal status."""
        if run_store is None:
            # Without a run store we can only stop on the trace's run_end event,
            # which the caller already replayed; treat as non-terminal so the
            # deadline bounds the tail. Only reached from the trace-guarded
            # join-stream endpoint, so a trace store is always present here.
            assert trace_store is not None
            events = await trace_store.get(run_id)
            return any(ev.get("type") == "run_end" for ev in events)
        record = await run_store.get(run_id)
        if record is None:
            return True
        status = record.status.value if hasattr(record.status, "value") else str(record.status)
        return status in _TERMINAL_STATES

    @app.post("/crons")
    async def create_cron(request: Request) -> Any:
        """Register a durable schedule that materializes queued runs when due."""
        _identify(request)
        if cron_store is None:
            raise HTTPException(status_code=404, detail="no cron store configured")
        body = await request.json()
        from .runs.cron import CronRecord, parse_schedule

        schedule = body.get("schedule") or ""
        try:
            interval = parse_schedule(schedule)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # SSRF guard: a cron's webhook is fetched server-side when the schedule
        # fires, so validate it before persisting the schedule.
        try:
            webhook = _validate_webhook(body.get("webhook"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        now = time.time()
        cron_id = body.get("cron_id") or f"cron_{uuid.uuid4().hex[:12]}"
        record = CronRecord(
            cron_id=cron_id,
            schedule=schedule,
            prompt=body.get("prompt", ""),
            agent=body.get("agent") or agent.name,
            enabled=body.get("enabled", True),
            next_run_at=body.get("next_run_at", now + interval),
            created_at=now,
            session_id=body.get("session_id"),
            identity=body.get("identity"),
            webhook=webhook,
        )
        await cron_store.create(record)
        return JSONResponse(record.model_dump(mode="json"), status_code=201)

    @app.get("/crons")
    async def list_crons(request: Request) -> Any:
        """List all registered schedules."""
        _identify(request)
        if cron_store is None:
            raise HTTPException(status_code=404, detail="no cron store configured")
        records = await cron_store.list()
        return JSONResponse([r.model_dump(mode="json") for r in records])

    @app.delete("/crons/{cron_id}")
    async def delete_cron(cron_id: str, request: Request) -> Any:
        """Remove a schedule."""
        _identify(request)
        if cron_store is None:
            raise HTTPException(status_code=404, detail="no cron store configured")
        removed = await cron_store.delete(cron_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"unknown cron {cron_id}")
        return JSONResponse({"cron_id": cron_id, "deleted": True})

    @app.post("/run/stream")
    async def run_stream(request: Request) -> Any:
        """Stream the run as Server-Sent Events (one event per loop step)."""
        identity = _identify(request)
        body = await request.json()
        prompt = body.get("prompt") or body.get("input") or ""
        runner_ = served_runner or runner or agent._get_runner()

        async def event_source():
            import json

            async for event in runner_.run_stream(
                agent, prompt, session_id=body.get("session_id"), identity=identity
            ):
                payload = _safe_event_payload(event)
                yield f"event: {event.type.value}\ndata: {json.dumps(payload)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.post("/chat/stream")
    async def chat_stream(request: Request) -> Any:
        """Token-level streaming (SSE) for a single answering turn."""
        identity = _identify(request)
        body = await request.json()
        prompt = body.get("prompt") or body.get("input") or ""

        async def token_source():
            async for token in agent.stream(prompt, identity=identity):
                yield f"data: {token}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(token_source(), media_type="text/event-stream")

    def _message_text(body: dict) -> str:
        message = body.get("message", {})
        parts = message.get("parts", [])
        return " ".join(p.get("text", "") for p in parts) or body.get("prompt", "")

    def _completed_task(task_id: str, output: Any) -> dict:
        from .runner import _safe

        return {
            "id": task_id,
            "status": {"state": "completed"},
            "artifacts": [{"name": "result", "parts": [{"text": str(_safe(output))}]}],
        }

    @app.post("/a2a/tasks")
    async def a2a_task(request: Request) -> Any:
        """A2A task endpoint: accept a message, return a completed task."""
        identity = _identify(request)
        body = await request.json()
        text = _message_text(body)
        task_id = body.get("id") or f"task_{uuid.uuid4().hex[:12]}"
        result = await agent.run(text, identity=identity)
        task = _completed_task(task_id, result.output)
        _A2A_TASKS[task_id] = task
        return JSONResponse(task)

    @app.get("/a2a/tasks/{task_id}")
    async def a2a_get_task(task_id: str, request: Request) -> Any:
        """Poll a previously-submitted task by id (long-running task support)."""
        _identify(request)
        task = _A2A_TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
        return JSONResponse(task)

    @app.post("/a2a/tasks/stream")
    async def a2a_task_stream(request: Request) -> Any:
        """Stream an A2A task's progress as SSE task-status events."""
        identity = _identify(request)
        body = await request.json()
        text = _message_text(body)
        task_id = body.get("id") or f"task_{uuid.uuid4().hex[:12]}"
        runner_ = served_runner or runner or agent._get_runner()

        async def task_events():
            import json

            yield f"data: {json.dumps({'id': task_id, 'status': {'state': 'working'}})}\n\n"
            output = None
            async for event in runner_.run_stream(agent, text, identity=identity):
                if event.type.value == "final_output":
                    output = event.payload.get("output")
            task = _completed_task(task_id, output)
            _A2A_TASKS[task_id] = task
            yield f"data: {json.dumps(task)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(task_events(), media_type="text/event-stream")

    async def _safe_body(request: Request) -> dict[str, Any]:
        """Parse a JSON body, tolerating an empty/absent one (returns ``{}``)."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - empty body is valid for these endpoints
            return {}
        return body if isinstance(body, dict) else {}

    return app


def _safe_event_payload(event: Any) -> dict:
    """Render an Event into a JSON-safe SSE payload (drops live objects).

    ``RUN_END`` carries the run's :class:`RunResult`; we surface both its
    ``output`` and its aggregate ``usage`` (tokens + cost) so the live stream
    shows cost, not just text. Every event's ``duration_ms`` is included when set
    so the console can render per-step latency without inferring it.
    """
    out: dict = {"type": event.type.value, "agent": event.agent, "run_id": event.run_id}
    if getattr(event, "duration_ms", None) is not None:
        out["duration_ms"] = event.duration_ms
    for key, value in event.payload.items():
        if key in ("result", "error"):
            # RunResult -> output + usage summary; Exception -> message.
            if hasattr(value, "output"):
                from .runner import _safe

                out["output"] = _safe(value.output)
                usage = getattr(value, "usage", None)
                if usage is not None and hasattr(usage, "model_dump"):
                    out["usage"] = usage.model_dump()
            else:
                out[key] = str(value)
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, dict)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def serve(
    agent: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    auth: AuthScheme | None = None,
    run_store: Any | None = None,
    approval_store: Any | None = None,
    trace_store: Any | None = None,
    run_checkpointer: Any | None = None,
    cron_store: Any | None = None,
) -> None:  # pragma: no cover - thin uvicorn wrapper
    """Run the agent's FastAPI app with uvicorn (blocking).

    Durable backends (``run_store``/``approval_store``/``trace_store``/
    ``run_checkpointer``/``cron_store``) are forwarded so a production deployment
    gets durable background runs, out-of-band sign-off, and a persisted trace.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to serve. `pip install uvicorn`.") from exc
    app = fastapi_server_app(
        agent,
        auth=auth,
        base_url=f"http://{host}:{port}",
        run_store=run_store,
        approval_store=approval_store,
        trace_store=trace_store,
        run_checkpointer=run_checkpointer,
        cron_store=cron_store,
    )
    uvicorn.run(app, host=host, port=port)


__all__ = ["fastapi_server_app", "serve"]
