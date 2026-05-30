"""Token streaming and the semantic event stream (offline)."""

import asyncio

from yaab import Agent
from yaab.testing import TestModel


async def main():
    agent = Agent("assistant", model=TestModel("Hello there, this is streamed."))

    # 1) Token-level streaming (chat UX).
    print("tokens: ", end="")
    async for token in agent.stream("say hi"):
        print(token, end="", flush=True)
    print()

    # 2) Semantic event stream (UIs / tracing / audit).
    print("events:")
    async for event in agent._get_runner().run_stream(agent, "say hi"):
        print(
            "  -",
            event.type.value,
            "|",
            {k: v for k, v in event.payload.items() if k not in ("result",)},
        )


asyncio.run(main())
