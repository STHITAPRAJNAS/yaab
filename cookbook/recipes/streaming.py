"""Recipe: streaming — token-by-token output.

``agent.stream(prompt)`` yields text deltas as they arrive, for a live UI.

    python -m cookbook.recipes.streaming
"""

from __future__ import annotations

import asyncio

from cookbook._harness import expect, resolve_model
from yaab import Agent
from yaab.models.test_model import TestModel


async def run() -> dict:
    model = resolve_model(offline_default=TestModel(custom_output="one two three four"))
    agent: Agent = Agent("narrator", model=model, instructions="Reply with a short phrase.")

    tokens: list[str] = []
    async for delta in agent.stream("count to four"):
        tokens.append(delta)
    text = "".join(tokens)

    expect(len(tokens) >= 1, "expected at least one streamed delta")
    expect("four" in text, f"expected the full phrase to stream, got {text!r}")
    return {"chunks": len(tokens), "text": text}


if __name__ == "__main__":
    print(asyncio.run(run()))
