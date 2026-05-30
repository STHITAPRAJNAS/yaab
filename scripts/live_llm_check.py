#!/usr/bin/env python
"""Live LLM integration check — runs YAAB against a real model.

Unlike scripts/smoke_all.py (fully offline), this makes real API calls to verify
the LiteLLM integration, real token streaming, tool calling, and structured
output against a live provider. Set a model + its API key in the environment:

    # Groq (free tier) — https://console.groq.com
    export GROQ_API_KEY=...                       # your key (never commit it)
    export YAAB_LIVE_MODEL=groq/llama-3.3-70b-versatile

    # or Google Gemini (free tier)
    export GEMINI_API_KEY=...
    export YAAB_LIVE_MODEL=gemini/gemini-2.0-flash

    # or local Ollama (no key)
    export YAAB_LIVE_MODEL=ollama/llama3

    python scripts/live_llm_check.py

Requires:  pip install 'yaab[litellm]'
"""

from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YAAB_LIVE_MODEL")


async def main() -> int:
    if not MODEL:
        print("Set YAAB_LIVE_MODEL (e.g. groq/llama-3.3-70b-versatile) + the provider key.")
        return 2

    from pydantic import BaseModel

    from yaab import Agent, tool

    print(f"Live model: {MODEL}\n")
    ok = 0
    total = 0

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok, total
        total += 1
        ok += passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {detail}")

    # 1) basic completion
    try:
        agent = Agent("a", model=MODEL, instructions="Answer in one short sentence.")
        r = await agent.run("What is the capital of France?")
        report("basic completion", "paris" in r.output.lower(), r.output[:60])
    except Exception as exc:  # noqa: BLE001
        report("basic completion", False, f"{type(exc).__name__}: {exc}")

    # 2) real token streaming
    try:
        agent = Agent("a", model=MODEL, instructions="Reply briefly.")
        chunks = [c async for c in agent.stream("Count: one two three")]
        report(
            "token streaming",
            len(chunks) >= 1 and bool("".join(chunks).strip()),
            f"{len(chunks)} chunks",
        )
    except Exception as exc:  # noqa: BLE001
        report("token streaming", False, f"{type(exc).__name__}: {exc}")

    # 3) tool calling (the model must choose to call the tool)
    try:

        @tool
        def get_weather(city: str) -> str:
            """Return the current weather for a city."""
            return f"{city}: 21C and sunny"

        agent = Agent(
            "w",
            model=MODEL,
            tools=[get_weather],
            instructions="Use the get_weather tool to answer.",
        )
        r = await agent.run("What's the weather in Paris?")
        report("tool calling", "21" in r.output or "sunny" in r.output.lower(), r.output[:60])
    except Exception as exc:  # noqa: BLE001
        report("tool calling", False, f"{type(exc).__name__}: {exc}")

    # 4) structured output (validated against a Pydantic model)
    try:

        class Capital(BaseModel):
            country: str
            city: str

        agent = Agent("c", model=MODEL, output_type=Capital)
        r = await agent.run("The capital of Japan. Return country and city.")
        report(
            "structured output",
            isinstance(r.output, Capital) and "tokyo" in r.output.city.lower(),
            str(r.output)[:60],
        )
    except Exception as exc:  # noqa: BLE001
        report("structured output", False, f"{type(exc).__name__}: {exc}")

    # 5) multi-agent (sequential, real models)
    try:
        from yaab import SequentialAgent

        researcher = Agent(
            "r", model=MODEL, instructions="List 2 terse bullet facts about the topic."
        )
        writer = Agent("w", model=MODEL, instructions="Write one sentence from the bullets.")
        r = await SequentialAgent("pipe", [researcher, writer]).run("the moon")
        report("multi-agent pipeline", bool(r.output and len(r.output) > 5), r.output[:60])
    except Exception as exc:  # noqa: BLE001
        report("multi-agent pipeline", False, f"{type(exc).__name__}: {exc}")

    print(f"\n{ok}/{total} live checks passed against {MODEL}.")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
