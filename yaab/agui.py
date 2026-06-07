"""AG-UI compatibility layer — stream YAAB runs as AG-UI protocol events.

AG-UI (the Agent-User Interaction protocol popularized by CopilotKit) is the
emerging standard for connecting agent backends to chat/coagent frontends. It
defines a small set of typed, streamed events. This middleware translates YAAB's
native event stream into AG-UI events so any AG-UI-compatible frontend can drive
a YAAB agent with no custom glue.

It is *middleware*, not a dependency: it wraps a `Runner.run_stream` (or
`agent.stream`) and yields AG-UI event dicts. Mount it over SSE with
:func:`agui_sse_app`, or consume :func:`run_agui` directly.

Reference event types (subset of the AG-UI spec):
``RUN_STARTED``, ``TEXT_MESSAGE_START`` / ``TEXT_MESSAGE_CONTENT`` /
``TEXT_MESSAGE_END``, ``TOOL_CALL_START`` / ``TOOL_CALL_ARGS`` /
``TOOL_CALL_END``, ``TOOL_CALL_RESULT``, ``THINKING_TEXT_MESSAGE_CONTENT``,
``STATE_SNAPSHOT`` / ``STATE_DELTA``, ``RUN_FINISHED``, ``RUN_ERROR``.
"""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .types import EventType


class AGUIEventType:
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    THINKING = "THINKING_TEXT_MESSAGE_CONTENT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    #: A run paused for human sign-off; the frontend should collect input and
    #: resume via :func:`resume_agui`. Carries ``approvalId``, ``tool``, ``args``.
    INPUT_REQUIRED = "INPUT_REQUIRED"


def _evt(type_: str, **fields: Any) -> dict[str, Any]:
    return {"type": type_, "timestamp": time.time(), **fields}


async def run_agui(
    agent: Any,
    prompt: str,
    *,
    runner: Any | None = None,
    thread_id: str | None = None,
    run_id: str | None = None,
    **run_kwargs: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Run an agent and yield AG-UI protocol events.

    Translates YAAB's semantic event stream (``run_stream``) into the AG-UI
    event vocabulary. ``run_kwargs`` (deps, session_id, identity, usage_limits,
    …) pass through to the runner.
    """
    runner = runner or agent._get_runner()
    thread_id = thread_id or f"thread_{uuid.uuid4().hex[:12]}"
    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    # Make HITL durable by default: a stable resume_id activates the pause path
    # (a guarded tool parks the run rather than erroring), so INPUT_REQUIRED can
    # be surfaced and resume_agui can continue it. Inert when no checkpointer.
    run_kwargs.setdefault("resume_id", thread_id)

    yield _evt(AGUIEventType.RUN_STARTED, threadId=thread_id, runId=run_id)
    # State round-trip: snapshot the session state to the frontend up front so it
    # can render the run's structured state, then stream deltas as steps write it.
    snapshot = await _state_snapshot(runner, run_kwargs.get("session_id"))
    yield _evt(AGUIEventType.STATE_SNAPSHOT, snapshot=snapshot)
    text_open = False

    try:
        async for event in runner.run_stream(agent, prompt, **run_kwargs):
            etype = event.type
            payload = event.payload

            if etype is EventType.STATE_WRITE:
                yield _evt(
                    AGUIEventType.STATE_DELTA,
                    key=payload.get("key"),
                    scope=payload.get("scope"),
                    delta={payload.get("key"): payload.get("value")},
                )

            elif etype is EventType.APPROVAL_REQUIRED:
                # The run durably paused for human sign-off. Tell the frontend to
                # collect input; it resumes the run via resume_agui with the
                # decision. End this stream cleanly (the run is parked, not failed).
                yield _evt(
                    AGUIEventType.INPUT_REQUIRED,
                    runId=run_id,
                    approvalId=payload.get("approval_id"),
                    tool=payload.get("tool"),
                    args=payload.get("arguments", {}),
                )
                return

            elif etype is EventType.MODEL_DELTA:
                if payload.get("reasoning"):
                    yield _evt(
                        AGUIEventType.THINKING,
                        messageId=message_id,
                        delta=payload["reasoning"],
                    )
                elif payload.get("delta"):
                    if not text_open:
                        yield _evt(
                            AGUIEventType.TEXT_MESSAGE_START,
                            messageId=message_id,
                            role="assistant",
                        )
                        text_open = True
                    yield _evt(
                        AGUIEventType.TEXT_MESSAGE_CONTENT,
                        messageId=message_id,
                        delta=payload["delta"],
                    )

            elif etype is EventType.TOOL_CALL:
                tcid = f"tc_{uuid.uuid4().hex[:8]}"
                yield _evt(
                    AGUIEventType.TOOL_CALL_START,
                    toolCallId=tcid,
                    toolCallName=payload.get("name", ""),
                )
                yield _evt(
                    AGUIEventType.TOOL_CALL_ARGS,
                    toolCallId=tcid,
                    delta=payload.get("arguments", {}),
                )
                yield _evt(AGUIEventType.TOOL_CALL_END, toolCallId=tcid)

            elif etype is EventType.TOOL_RESULT:
                yield _evt(
                    AGUIEventType.TOOL_CALL_RESULT,
                    toolCallName=payload.get("name", ""),
                    content=payload.get("result"),
                )

            elif etype is EventType.FINAL_OUTPUT:
                output = payload.get("output")
                # Emit the final text as one message if nothing streamed yet.
                if not text_open and isinstance(output, str):
                    yield _evt(
                        AGUIEventType.TEXT_MESSAGE_START,
                        messageId=message_id,
                        role="assistant",
                    )
                    yield _evt(
                        AGUIEventType.TEXT_MESSAGE_CONTENT,
                        messageId=message_id,
                        delta=output,
                    )
                    text_open = True
                if text_open:
                    yield _evt(AGUIEventType.TEXT_MESSAGE_END, messageId=message_id)
                    text_open = False

            elif etype is EventType.ERROR:
                yield _evt(
                    AGUIEventType.RUN_ERROR,
                    runId=run_id,
                    message=str(payload.get("error", "error")),
                )
                return

        yield _evt(AGUIEventType.RUN_FINISHED, threadId=thread_id, runId=run_id)

    except Exception as exc:  # noqa: BLE001 - surface as an AG-UI run error
        yield _evt(AGUIEventType.RUN_ERROR, runId=run_id, message=str(exc))


async def _state_snapshot(runner: Any, session_id: str | None) -> dict[str, Any]:
    """The session's structured state as a plain dict (empty if none yet)."""
    if session_id is None:
        return {}
    service = getattr(runner, "session_service", None)
    if service is None:
        return {}
    try:
        session = await service.get(session_id)
    except Exception:  # noqa: BLE001 - a missing/unsupported backend is just empty
        return {}
    return dict(session.state) if session is not None else {}


async def resume_agui(
    agent: Any,
    decision: Any,
    *,
    runner: Any | None = None,
    thread_id: str | None = None,
    run_id: str | None = None,
    **run_kwargs: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Resume a paused run from a frontend's decision, streaming AG-UI events.

    The other half of the HITL round-trip: after a run emits ``INPUT_REQUIRED``
    and a reviewer decides (an :class:`~yaab.governance.approvals_decide.Decision`
    from ``approvals.approve``/``deny``/``respond``), this streams the
    continuation in the same AG-UI vocabulary as :func:`run_agui` — so the
    frontend that surfaced the pause also drives the resume.
    """
    runner = runner or agent._get_runner()
    thread_id = thread_id or f"thread_{uuid.uuid4().hex[:12]}"
    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    yield _evt(AGUIEventType.RUN_STARTED, threadId=thread_id, runId=run_id)
    text_open = False
    try:
        result = await agent.run(resume=decision, **run_kwargs)
        output = result.output
        if isinstance(output, str) and output:
            yield _evt(AGUIEventType.TEXT_MESSAGE_START, messageId=message_id, role="assistant")
            yield _evt(AGUIEventType.TEXT_MESSAGE_CONTENT, messageId=message_id, delta=output)
            yield _evt(AGUIEventType.TEXT_MESSAGE_END, messageId=message_id)
            text_open = True
        # If it paused AGAIN (a second gate), surface another INPUT_REQUIRED.
        if getattr(result, "paused", False) and result.pending:
            p = result.pending[0]
            yield _evt(
                AGUIEventType.INPUT_REQUIRED,
                runId=run_id,
                approvalId=getattr(p, "approval_id", None),
                tool=getattr(p, "tool", None),
                args=getattr(p, "arguments", {}),
            )
            return
        yield _evt(AGUIEventType.RUN_FINISHED, threadId=thread_id, runId=run_id)
    except Exception as exc:  # noqa: BLE001 - surface as an AG-UI run error
        _ = text_open
        yield _evt(AGUIEventType.RUN_ERROR, runId=run_id, message=str(exc))


def agui_sse_app(agent: Any, *, runner: Any | None = None, auth: Any | None = None) -> Any:
    """Build a FastAPI app exposing the agent over AG-UI SSE at ``POST /agui``.

    The request body is an AG-UI run input: ``{"threadId": ..., "runId": ...,
    "messages": [...] }`` or a simple ``{"prompt": "..."}``. Each AG-UI event is
    streamed as an SSE ``data:`` line of JSON.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "FastAPI is required for agui_sse_app. `pip install fastapi uvicorn`."
        ) from exc

    from .auth import NoAuth

    auth_scheme = auth or NoAuth()
    app = FastAPI(title=f"YAAB AG-UI · {agent.name}")

    @app.post("/agui")
    async def agui_endpoint(request: Request) -> Any:
        import json

        identity = auth_scheme.authenticate(dict(request.headers)) or "anonymous"
        body = await request.json()
        prompt = body.get("prompt") or _last_user_text(body) or ""
        thread_id = body.get("threadId")
        run_id = body.get("runId")

        async def event_source():
            async for ev in run_agui(
                agent,
                prompt,
                runner=runner,
                thread_id=thread_id,
                run_id=run_id,
                identity=identity,
            ):
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    return app


def _last_user_text(body: dict[str, Any]) -> str:
    """Extract the latest user message text from an AG-UI run-input body."""
    messages = body.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


__all__ = ["run_agui", "resume_agui", "agui_sse_app", "AGUIEventType"]
