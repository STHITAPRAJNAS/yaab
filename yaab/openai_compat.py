"""OpenAI-compatible Chat Completions API over yaab agents.

Exposes ``POST /v1/chat/completions`` and ``GET /v1/models`` so any OpenAI-SDK
client, eval harness, or gateway can drive a yaab agent unchanged. The OpenAI
``model`` string selects an agent; without ``tools`` in the request the full
agent loop runs (its own tools execute internally) and the final answer is
returned; with ``tools`` a single model turn surfaces ``tool_calls`` for the
caller to execute (OpenAI function-calling passthrough).
"""

import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from .auth import AuthError, AuthScheme, NoAuth
from .types import Message, Role


class _OpenAIError(Exception):
    """Carries an HTTP status + OpenAI error envelope fields."""

    def __init__(self, status: int, message: str, *, type: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.code = code

    def envelope(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "code": self.code}}


def _to_yaab_messages(messages: list[dict[str, Any]]) -> list[Message]:
    out: list[Message] = []
    for m in messages:
        out.append(
            Message(
                role=Role(m["role"]),
                content=m.get("content") or "",
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id"),
            )
        )
    return out


def _usage_block(usage: Any) -> dict[str, int]:
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


async def _tool_passthrough(agent: Any, messages: list[dict[str, Any]], body: dict[str, Any]) -> Any:
    """Single model turn with caller-supplied tools; map the response to OpenAI."""
    import json

    from fastapi.responses import JSONResponse

    yaab_messages = _to_yaab_messages(messages)
    if agent.instructions and isinstance(agent.instructions, str):
        yaab_messages = [Message(role=Role.SYSTEM, content=agent.instructions), *yaab_messages]
    resp = await agent.model.complete(
        yaab_messages,
        tools=body.get("tools"),
        tool_choice=body.get("tool_choice"),
    )
    message: dict[str, Any] = {"role": "assistant", "content": resp.content or None}
    finish_reason = "stop"
    if resp.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in resp.tool_calls
        ]
        finish_reason = "tool_calls"
    return JSONResponse(
        {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": agent.name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": _usage_block(resp.usage),
        }
    )


def _stream_completion(
    runner: Any, agent: Any, prompt: str, session_id: str | None, identity: str
) -> Any:
    """Return a StreamingResponse of OpenAI ``chat.completion.chunk`` SSE events."""
    import json

    from fastapi.responses import StreamingResponse

    cid = _completion_id()
    created = int(time.time())

    def _chunk(delta: dict[str, Any], finish_reason: str | None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": agent.name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    async def _source() -> Any:
        # First chunk announces the assistant role.
        yield _chunk({"role": "assistant"}, None)
        async for token in runner.stream_text(
            agent, prompt, session_id=session_id, identity=identity
        ):
            if token:
                yield _chunk({"content": token}, None)
        yield _chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(_source(), media_type="text/event-stream")


def add_openai_routes(
    app: Any,
    resolve_agent: Callable[[str | None], Any],
    *,
    auth_scheme: AuthScheme | None = None,
    runner: Any | None = None,
) -> None:
    """Mount the OpenAI-compatible routes onto ``app``.

    ``resolve_agent(model)`` returns the agent for an OpenAI ``model`` string
    (``None`` selects a default) and raises ``_OpenAIError`` (404) for unknown
    models. ``runner`` owns the session service used to seed multi-turn history;
    one with an in-memory session service is built when omitted.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    auth = auth_scheme or NoAuth()

    if runner is None:
        from .runner import Runner
        from .sessions import InMemorySessionService

        runner = Runner(session_service=InMemorySessionService())

    def _identify(request: Request) -> str:
        try:
            return auth.authenticate(dict(request.headers)) or "anonymous"
        except AuthError as exc:
            raise _OpenAIError(
                401, str(exc), type="invalid_request_error", code="invalid_api_key"
            ) from exc

    async def _seed_session(prior: list[dict[str, Any]]) -> str | None:
        """Create an ephemeral session seeded with ``prior`` turns; return its id."""
        if not prior:
            return None
        session = await runner.session_service.create_session()
        for m in prior:
            await runner.session_service.append_message(
                session.id,
                Message(role=Role(m["role"]), content=m.get("content") or ""),
            )
        return session.id

    @app.get("/v1/models")
    async def list_models() -> Any:
        names_fn = getattr(resolve_agent, "names", None)
        ids = names_fn() if callable(names_fn) else []
        data = [{"id": name, "object": "model", "owned_by": "yaab"} for name in ids]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            identity = _identify(request)
            body = await request.json()
            agent = resolve_agent(body.get("model"))
            messages = body.get("messages") or []
            if not messages:
                raise _OpenAIError(
                    400,
                    "messages must not be empty",
                    type="invalid_request_error",
                    code="invalid_request",
                )

            # Client-side function-calling passthrough: when the request carries
            # `tools`, do a single model turn with those tools and surface any
            # tool_calls back to the caller (no server-side execution).
            if body.get("tools"):
                return await _tool_passthrough(agent, messages, body)

            if messages[-1].get("role") != "user":
                raise _OpenAIError(
                    400,
                    "the last message must have role 'user'",
                    type="invalid_request_error",
                    code="invalid_request",
                )
            prompt = messages[-1].get("content") or ""
            session_id = await _seed_session(messages[:-1])

            if body.get("stream"):
                return _stream_completion(runner, agent, prompt, session_id, identity)

            result = await runner.run(
                agent, prompt, session_id=session_id, identity=identity
            )
            return JSONResponse(
                {
                    "id": _completion_id(),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": agent.name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": str(result.output),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _usage_block(result.usage),
                }
            )
        except _OpenAIError as err:
            return JSONResponse(err.envelope(), status_code=err.status)


class _AgentRegistry:
    """Resolves an OpenAI ``model`` string to an agent; callable + introspectable."""

    def __init__(self, agents: Mapping[str, Any] | Any) -> None:
        if hasattr(agents, "items"):
            self._agents = dict(agents)
            self._default = next(iter(self._agents.values()), None)
        else:  # a bare Agent
            self._agents = {agents.name: agents}
            self._default = agents

    def names(self) -> list[str]:
        return list(self._agents)

    def __call__(self, model: str | None) -> Any:
        if model is None:
            if self._default is None:
                raise _OpenAIError(
                    404, "no default model", type="invalid_request_error", code="model_not_found"
                )
            return self._default
        agent = self._agents.get(model)
        if agent is None:
            raise _OpenAIError(
                404,
                f"model {model!r} not found",
                type="invalid_request_error",
                code="model_not_found",
            )
        return agent


def openai_compat_app(
    agents: Mapping[str, Any] | Any,
    *,
    auth: AuthScheme | None = None,
    runner: Any | None = None,
) -> Any:
    """Build a FastAPI app serving ``agents`` behind the OpenAI Chat API.

    ``agents`` is a ``{model_name: Agent}`` mapping or a single ``Agent`` (served
    under its name and as the default).
    """
    from fastapi import FastAPI

    app = FastAPI(title="YAAB · OpenAI-compatible API")
    registry = _AgentRegistry(agents)
    add_openai_routes(app, registry, auth_scheme=auth, runner=runner)
    return app
