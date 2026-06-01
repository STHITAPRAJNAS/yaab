"""Quick live check of resumable runs, user-simulation evals, and built-in tools.

Complements scripts/live_e2e.py with the newest features. Needs a real model key
in .env (defaults to gemini/gemini-2.5-flash via YAAB_LIVE_MODEL).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Load .env (keys never live in code or in this script's output).
for line in (Path(__file__).resolve().parent.parent / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

MODEL = os.environ.get("YAAB_LIVE_MODEL", "gemini/gemini-2.5-flash")


async def check_resumable_run() -> str:
    """A finished resume_id replays its result with zero extra model calls."""
    from yaab import Agent, Runner
    from yaab.graph.checkpoint import MemorySaver

    runner = Runner(run_checkpointer=MemorySaver())
    agent = Agent("r", model=MODEL, instructions="Reply with one short sentence.")
    r1 = await runner.run(agent, "Name one planet.", resume_id="live-1")
    cost1 = r1.usage.total_tokens
    # Re-invoke with the same resume_id: must replay, not re-ask the model.
    r2 = await runner.run(agent, "Name one planet.", resume_id="live-1")
    assert r2.output == r1.output, "replayed output differs"
    assert r2.usage.total_tokens == r1.usage.total_tokens, "replay consumed tokens"
    return f"run + idempotent replay OK (first run {cost1} tokens, replay 0 extra)"


async def check_user_simulation() -> str:
    """An LLM persona drives a multi-turn conversation and the goal is scored."""
    from yaab import Agent
    from yaab.governance import UserSimulator, simulate

    agent = Agent(
        "support",
        model=MODEL,
        instructions=(
            "You are a support agent for AcmeBank. The reset procedure is: "
            "go to acmebank.com/reset and enter your registered email. Be concise."
        ),
    )
    simulator = UserSimulator(
        MODEL,
        persona="A polite customer who forgot their online banking password.",
        goal="Learn how to reset the password.",
        max_turns=4,
    )
    result = await simulate(agent, simulator)
    turns = len(result.transcript)
    return f"simulated {turns} turns, goal_achieved={result.goal_achieved}"


async def check_builtin_tools_live() -> str:
    """The agent uses built-in calculator + current_time tools on a real model."""
    from yaab import Agent
    from yaab.tools.builtin import calculator, current_time

    agent = Agent(
        "calc",
        model=MODEL,
        tools=[calculator, current_time],
        instructions="Use tools for any math. Reply with just the number.",
    )
    r = await agent.run("What is 13 * 17? Use the calculator tool.")
    assert "221" in str(r.output), f"expected 221 in output, got: {r.output!r}"
    return f"calculator tool used, answer: {str(r.output).strip()[:40]}"


CHECKS = [
    ("resumable runs (checkpoint + replay)", check_resumable_run),
    ("user-simulation eval (multi-turn)", check_user_simulation),
    ("built-in tools (calculator)", check_builtin_tools_live),
]


async def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = await fn()
            print(f"  [PASS] {name:42s} {detail}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  [FAIL] {name:42s} {type(exc).__name__}: {exc}")
        await asyncio.sleep(1.0)
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} live checks passed on {MODEL}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
