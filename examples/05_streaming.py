"""Token streaming and the semantic event stream (offline)."""

import asyncio

from yaab import Agent
from yaab.testing import TestModel


async def main() -> dict:
    """Stream a run two ways and return what was streamed."""
    agent = Agent("assistant", model=TestModel("Hello there, this is streamed."))

    # 1) Token-level streaming (chat UX).
    tokens = []
    print("tokens: ", end="")
    async for token in agent.stream("say hi"):
        tokens.append(token)
        print(token, end="", flush=True)
    print()

    # 2) Full-run event stream (UIs / tracing / audit): tokens, tool calls,
    #    and the final result as typed events.
    events = []
    print("events:")
    async for event in agent.stream_events("say hi"):
        events.append(event)
        print(
            "  -",
            event.type.value,
            "|",
            {k: v for k, v in event.payload.items() if k not in ("result",)},
        )

    return {"tokens": tokens, "event_types": [e.type for e in events]}


if __name__ == "__main__":
    asyncio.run(main())
