"""Serve agents over HTTP — a FastAPI app and an A2A-compatible server.

``fastapi_server_app(agent)`` returns a ready-to-mount ASGI app exposing:

* ``GET  /.well-known/agent.json`` — the A2A Agent Card (discovery);
* ``POST /run``                    — run the agent (YAAB-native);
* ``POST /a2a/tasks``              — A2A task submission (agent-to-agent);
* ``GET  /health``                 — liveness.

Authentication is pluggable via :mod:`yaab.auth`; the resolved identity flows
into the run context and the audit log. FastAPI is an optional dependency,
imported lazily so importing YAAB never requires a web stack.
"""

import uuid
from typing import Any

from .auth import AuthError, AuthScheme, NoAuth

# In-process store of submitted A2A tasks, so clients can poll by id. A durable
# deployment would back this with the session/artifact services.
_A2A_TASKS: dict[str, dict] = {}


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

    @app.post("/run")
    async def run(request: Request) -> Any:
        identity = _identify(request)
        body = await request.json()
        prompt = body.get("prompt") or body.get("input") or ""
        result = await agent.run(prompt, session_id=body.get("session_id"), identity=identity)
        from .runner import _safe

        return JSONResponse(
            {
                "output": _safe(result.output),
                "run_id": result.run_id,
                "usage": result.usage.model_dump(),
            }
        )

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
