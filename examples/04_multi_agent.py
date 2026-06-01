"""Multi-agent patterns: sequential, parallel, and swarm hand-off (offline)."""

import asyncio

from yaab import Agent, ParallelAgent, SequentialAgent, Swarm
from yaab.multiagent import SwarmState
from yaab.testing import TestModel


async def main() -> dict:
    """Run the three multi-agent patterns and return each one's output."""
    # Sequential pipeline: extract -> summarize.
    extract = Agent("extract", model=TestModel("raw facts"))
    summarize = Agent("summarize", model=TestModel("a tidy summary"))
    pipeline = SequentialAgent("pipeline", [extract, summarize])
    sequential = (await pipeline.run("a document")).output
    print("sequential:", sequential)

    # Parallel review board.
    legal = Agent("legal", model=TestModel("legally fine"))
    finance = Agent("finance", model=TestModel("budget approved"))
    board = ParallelAgent("board", [legal, finance])
    parallel = (await board.run("review contract")).output
    print("parallel:", parallel)

    # Swarm: triage hands off to a specialist.
    triage = Agent(
        "triage", model=TestModel(custom_output="routing", call_tools=["handoff_to_billing"])
    )
    billing = Agent("billing", model=TestModel("refund processed"))
    swarm = Swarm("support", [triage, billing], entry="triage")
    swarm_out = (await swarm.run("I was double charged", deps=SwarmState())).output
    print("swarm:", swarm_out)

    return {"sequential": sequential, "parallel": parallel, "swarm": swarm_out}


if __name__ == "__main__":
    asyncio.run(main())
