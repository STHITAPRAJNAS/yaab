"""Tests for the Content/Part multimodal type and SSE streaming."""

from __future__ import annotations

import pytest

from yaab import Agent, Content, Part, PartKind
from yaab.testing import TestModel
from yaab.types import Role


def test_text_content_renders_to_string():
    c = Content.from_text("hello")
    assert c.text == "hello"
    assert c.to_provider_content() == "hello"  # cheap path for text-only
    assert not c.is_multimodal()


def test_multimodal_content_renders_array():
    c = Content(
        role=Role.USER,
        parts=[
            Part.text_part("describe this"),
            Part.data_part(b"\x89PNG fake", "image/png"),
        ],
    )
    assert c.is_multimodal()
    rendered = c.to_provider_content()
    assert isinstance(rendered, list)
    assert rendered[0]["type"] == "text"
    assert rendered[1]["type"] == "image_url"
    assert rendered[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_data_part_roundtrip():
    part = Part.data_part(b"binary-bytes", "application/octet-stream")
    assert part.kind is PartKind.DATA
    assert part.decoded() == b"binary-bytes"


def test_message_carries_multimodal_parts():
    c = Content(
        role=Role.USER,
        parts=[Part.text_part("hi"), Part.file_part("https://x/img.png", "image/png")],
    )
    msg = c.to_message()
    assert msg.content == "hi"
    # The flat provider dict uses content_parts when present.
    msg.content_parts = c.to_provider_content()
    provider = msg.to_provider_dict()
    assert isinstance(provider["content"], list)


@pytest.mark.asyncio
async def test_agent_accepts_content_prompt():
    agent = Agent("a", model=TestModel("saw it"))
    c = Content(role=Role.USER, parts=[Part.text_part("what is this?"),
                                       Part.data_part(b"img", "image/png")])
    result = await agent.run(c)
    assert result.output == "saw it"


def test_sse_endpoint_streams_events():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from yaab.serve import fastapi_server_app

    agent = Agent("a", model=TestModel("streamed answer"))
    client = TestClient(fastapi_server_app(agent))
    with client.stream("POST", "/run/stream", json={"prompt": "hi"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = "".join(resp.iter_text())
    assert "event: run_start" in body
    assert "event: final_output" in body
    assert "event: done" in body
