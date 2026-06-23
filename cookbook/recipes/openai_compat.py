"""Recipe: OpenAI-compatible API — serve an agent for any OpenAI client.

``openai_compat_app`` exposes ``/v1/chat/completions``. This recipe drives it
in-process via an ASGI transport — the same request an OpenAI SDK would make.

    python -m cookbook.recipes.openai_compat
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent
from yaab.models.test_model import TestModel
from yaab.openai_compat import openai_compat_app


async def run() -> dict:
    import httpx

    agent: Agent = Agent("assistant", model=TestModel(custom_output="Paris is the capital."))
    app = openai_compat_app({"assistant": agent})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "assistant",
                "messages": [{"role": "user", "content": "capital of France?"}],
            },
        )
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    expect(resp.status_code == 200, f"expected 200, got {resp.status_code}")
    expect("paris" in content.lower(), f"expected Paris in the reply, got {content!r}")
    return {"status": resp.status_code, "content": content}


if __name__ == "__main__":
    print(asyncio.run(run()))
