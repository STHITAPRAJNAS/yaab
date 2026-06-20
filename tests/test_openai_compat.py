from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from yaab import Agent  # noqa: E402
from yaab.openai_compat import openai_compat_app  # noqa: E402
from yaab.testing import TestModel  # noqa: E402


def _client(agents):
    return TestClient(openai_compat_app(agents))


def test_chat_completion_nonstream_shape():
    agent = Agent("assistant", model=TestModel(custom_output="Hello there."))
    client = _client({"assistant": agent})
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "assistant", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "assistant"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hello there."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert set(body["usage"]) >= {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert body["id"].startswith("chatcmpl-")


def test_models_list():
    a = Agent("a", model=TestModel(custom_output="x"))
    b = Agent("b", model=TestModel(custom_output="y"))
    client = _client({"a": a, "b": b})
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"a", "b"}
    assert all(m["object"] == "model" for m in body["data"])


def test_unknown_model_404_openai_envelope():
    agent = Agent("a", model=TestModel(custom_output="x"))
    client = _client({"a": agent})
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"


def test_single_agent_default_model():
    # A bare Agent (not a mapping) is served under its name and as the default.
    agent = Agent("solo", model=TestModel(custom_output="ok"))
    client = _client(agent)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"


def test_multi_turn_seeds_history():
    # A multi-turn request seeds prior turns into an ephemeral session; the agent
    # sees them (asserted via the model recording the messages it was called with).
    model = TestModel(custom_output="answer")
    agent = Agent("assistant", model=model)
    client = _client({"assistant": agent})
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "assistant",
            "messages": [
                {"role": "user", "content": "my name is Alice"},
                {"role": "assistant", "content": "Hi Alice"},
                {"role": "user", "content": "what is my name?"},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "answer"
    # The model was called with the seeded prior turns in context.
    seen = " ".join(m.content for call in model.calls for m in call)
    assert "Alice" in seen


def test_ephemeral_sessions_are_cleaned_up():
    # Multi-turn requests seed ephemeral sessions; they must not accumulate.
    from yaab.runner import Runner
    from yaab.sessions import InMemorySessionService

    svc = InMemorySessionService()
    runner = Runner(session_service=svc)
    agent = Agent("a", model=TestModel(custom_output="ok"))
    from yaab.openai_compat import openai_compat_app as _app

    client = TestClient(_app({"a": agent}, runner=runner))
    for _ in range(3):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "a",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "again"},
                ],
            },
        )
        assert r.status_code == 200
    # No ephemeral sessions left behind.
    assert len(svc._store) == 0


def _parse_sse(text):
    """Return the list of JSON `data:` payloads (excluding the [DONE] sentinel)."""
    import json

    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                continue
            out.append(json.loads(payload))
    return out


def test_chat_completion_streaming():
    agent = Agent("assistant", model=TestModel(custom_output="one two three"))
    client = _client({"assistant": agent})
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "assistant",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    chunks = _parse_sse(resp.text)
    # Every chunk is a chat.completion.chunk; first carries the role.
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    # Concatenated content deltas reconstruct the answer.
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "one two three" in text
    # The last chunk carries finish_reason and the [DONE] sentinel terminates.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert resp.text.rstrip().endswith("[DONE]")


def test_tools_passthrough_surfaces_tool_calls():
    # When the request carries `tools`, a single model turn surfaces tool_calls
    # back to the caller (OpenAI function-calling passthrough) — no execution.
    agent = Agent("assistant", model=TestModel(call_tools=["get_weather"]))
    client = _client({"assistant": agent})
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "assistant",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}},
                }
            ],
        },
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    # arguments is a JSON string per the OpenAI schema.
    assert isinstance(calls[0]["function"]["arguments"], str)


def test_auth_required_401_envelope():
    from yaab.auth import BearerTokenAuth

    agent = Agent("a", model=TestModel(custom_output="x"))
    app = openai_compat_app({"a": agent}, auth=BearerTokenAuth({"sk-good": "alice"}))
    client = TestClient(app)
    body = {"model": "a", "messages": [{"role": "user", "content": "hi"}]}

    # No key -> 401 with an OpenAI error envelope.
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"

    # Valid bearer key -> 200.
    r = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-good"})
    assert r.status_code == 200


def test_exported_from_package():
    import yaab

    assert hasattr(yaab, "openai_compat_app")


def test_fastapi_server_app_openai_flag():
    from yaab.serve import fastapi_server_app

    agent = Agent("srv", model=TestModel(custom_output="served"))
    app = fastapi_server_app(agent, openai_compat=True)
    client = TestClient(app)
    # Native endpoint still works.
    assert client.get("/health").status_code == 200
    # And the OpenAI surface is mounted on the same app.
    r = client.post(
        "/v1/chat/completions",
        json={"model": "srv", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "served"


@pytest.mark.asyncio
async def test_real_openai_sdk_compatibility():
    """Drive the actual `openai` SDK against the app in-process (ASGI transport)."""
    openai = pytest.importorskip("openai")
    import httpx

    agent = Agent("gpt", model=TestModel(custom_output="hi from yaab"))
    app = openai_compat_app({"gpt": agent})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        sdk = openai.AsyncOpenAI(
            api_key="unused", base_url="http://test/v1", http_client=http_client
        )
        completion = await sdk.chat.completions.create(
            model="gpt", messages=[{"role": "user", "content": "hi"}]
        )
        assert completion.choices[0].message.content == "hi from yaab"
        assert completion.usage.total_tokens >= 0

        models = await sdk.models.list()
        assert any(m.id == "gpt" for m in models.data)
