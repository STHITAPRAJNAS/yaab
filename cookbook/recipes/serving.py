"""Recipe: serving — expose an agent over HTTP (native + A2A endpoints).

``fastapi_server_app`` builds a FastAPI app with health, an agent card, and a
``/run`` endpoint. This recipe drives it in-process.

    python -m cookbook.recipes.serving
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect
from yaab import Agent
from yaab.models.test_model import TestModel
from yaab.serve import fastapi_server_app


async def run() -> dict:
    import httpx

    agent: Agent = Agent("assistant", model=TestModel(custom_output="Hello from the server."))
    app = fastapi_server_app(agent)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        run_resp = await client.post("/run", json={"prompt": "hi"})

    expect(health.status_code == 200, f"health should be 200, got {health.status_code}")
    out = run_resp.json().get("output", "")
    expect("hello" in str(out).lower(), f"expected the agent's answer, got {out!r}")
    return {"health": health.status_code, "output": out}


if __name__ == "__main__":
    print(asyncio.run(run()))
