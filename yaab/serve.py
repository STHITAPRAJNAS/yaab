"""Serve agents over HTTP — a FastAPI app and an A2A-compatible server.

``fastapi_server_app(agent)`` returns a ready-to-mount ASGI app exposing:

* ``GET  /.well-known/agent.json`` — the A2A Agent Card (discovery);
* ``POST /run``                    — run the agent (YAAB-native; ``background``
  submits it as a task and returns ``202`` immediately);
* ``GET  /runs`` / ``GET /runs/{id}`` / ``POST /runs/{id}/cancel`` — the run
  lifecycle API (poll status, list, remotely cancel an in-flight run);
* ``POST /a2a/tasks``              — A2A task submission (agent-to-agent);
* ``GET  /health``                 — liveness.

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


def fastapi_server_app(
    agent: Any,
    *,
    runner: Any | None = None,
    auth: AuthScheme | None = None,
    base_url: str = "",
) -> Any:
    """Build a FastAPI app that serves ``agent`` (YAAB-native + A2A endpoints)."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "FastAPI is required to serve agents. Install with `pip install fastapi uvicorn`."
        ) from exc

    auth_scheme = auth or NoAuth()
    app = FastAPI(title=f"YAAB · {agent.name}", version=getattr(agent, "version", "0.1.0"))

    def _identify(request: Request) -> str:
        try:
            return auth_scheme.authenticate(dict(request.headers)) or "anonymous"
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

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

    @app.post("/run")
    async def run(request: Request) -> Any:
        identity = _identify(request)
        body = await request.json()
        prompt = body.get("prompt") or body.get("input") or ""
        session_id = body.get("session_id")
        token = CancellationToken()
        api_run_id, entry = _register_run(token)

        # Background: fire-and-poll. Submit as a task, register, return 202 now.
        if body.get("background"):
            import asyncio

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
        items = [
            {"id": e["id"], "status": e["status"], "started_at": e["started_at"]}
            for e in _RUN_REGISTRY.values()
        ]
        items.reverse()  # dict preserves insertion order; reverse = newest first
        return JSONResponse(items)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> Any:
        """Status of a single run (+ output/usage or error once finished)."""
        _identify(request)
        entry = _RUN_REGISTRY.get(run_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return JSONResponse(_run_view(entry))

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request) -> Any:
        """Remotely cancel an in-flight run. No-op for an already-finished run."""
        _identify(request)
        entry = _RUN_REGISTRY.get(run_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        if entry["status"] == "running":
            entry["_api_cancelled"] = True
            entry["token"].cancel("api_cancel")
        return JSONResponse({"run_id": run_id, "status": entry["status"]})

    @app.post("/run/stream")
    async def run_stream(request: Request) -> Any:
        """Stream the run as Server-Sent Events (one event per loop step)."""
        identity = _identify(request)
        body = await request.json()
        prompt = body.get("prompt") or body.get("input") or ""
        runner_ = runner or agent._get_runner()

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
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
        return JSONResponse(task)

    @app.post("/a2a/tasks/stream")
    async def a2a_task_stream(request: Request) -> Any:
        """Stream an A2A task's progress as SSE task-status events."""
        identity = _identify(request)
        body = await request.json()
        text = _message_text(body)
        task_id = body.get("id") or f"task_{uuid.uuid4().hex[:12]}"
        runner_ = runner or agent._get_runner()

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

    return app


def _safe_event_payload(event: Any) -> dict:
    """Render an Event into a JSON-safe SSE payload (drops live objects)."""
    out: dict = {"type": event.type.value, "agent": event.agent, "run_id": event.run_id}
    for key, value in event.payload.items():
        if key in ("result", "error"):
            # RunResult -> compact summary; Exception -> message.
            if hasattr(value, "output"):
                from .runner import _safe

                out["output"] = _safe(value.output)
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
) -> None:  # pragma: no cover - thin uvicorn wrapper
    """Run the agent's FastAPI app with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to serve. `pip install uvicorn`.") from exc
    app = fastapi_server_app(agent, auth=auth, base_url=f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


__all__ = ["fastapi_server_app", "serve"]
