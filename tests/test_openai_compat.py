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
