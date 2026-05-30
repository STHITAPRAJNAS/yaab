"""Serve agents over HTTP — a FastAPI app and an A2A-compatible server.

``get_fastapi_app(agent)`` returns a ready-to-mount ASGI app exposing:

* ``GET  /.well-known/agent.json`` — the A2A Agent Card (discovery);
* ``POST /run``                    — run the agent (YAAB-native);
* ``POST /a2a/tasks``              — A2A task submission (agent-to-agent);
* ``GET  /health``                 — liveness.

Authentication is pluggable via :mod:`yaab.auth`; the resolved identity flows
into the run context and the audit log. FastAPI is an optional dependency,
imported lazily so importing YAAB never requires a web stack.
"""

import uuid
from typing import Any, Optional

from .auth import AuthError, AuthScheme, NoAuth


def get_fastapi_app(
    agent: Any,
    *,
    runner: Optional[Any] = None,
    auth: Optional[AuthScheme] = None,
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
        result = await agent.run(
            prompt, session_id=body.get("session_id"), identity=identity
        )
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

    @app.post("/a2a/tasks")
    async def a2a_task(request: Request) -> Any:
        """Minimal A2A task endpoint: accept a message, return a completed task."""
        identity = _identify(request)
        body = await request.json()
        # A2A message: {"message": {"parts": [{"text": "..."}]}}
        message = body.get("message", {})
        parts = message.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts) or body.get("prompt", "")
        result = await agent.run(text, identity=identity)
        from .runner import _safe

        task_id = body.get("id") or f"task_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            {
                "id": task_id,
                "status": {"state": "completed"},
                "artifacts": [
                    {"name": "result", "parts": [{"text": str(_safe(result.output))}]}
                ],
            }
        )

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
    auth: Optional[AuthScheme] = None,
) -> None:  # pragma: no cover - thin uvicorn wrapper
    """Run the agent's FastAPI app with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to serve. `pip install uvicorn`.") from exc
    app = get_fastapi_app(agent, auth=auth, base_url=f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


__all__ = ["get_fastapi_app", "serve"]
