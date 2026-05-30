"""Multi-agent patterns: sequential, parallel, and swarm hand-off (offline)."""

import asyncio

from yaab import Agent, ParallelAgent, SequentialAgent, Swarm
from yaab.multiagent import SwarmState
from yaab.testing import TestModel


async def main():
    # Sequential pipeline: extract -> summarize.
    extract = Agent("extract", model=TestModel("raw facts"))
    summarize = Agent("summarize", model=TestModel("a tidy summary"))
    pipeline = SequentialAgent("pipeline", [extract, summarize])
    print("sequential:", (await pipeline.run("a document")).output)

    # Parallel review board.
    legal = Agent("legal", model=TestModel("legally fine"))
    finance = Agent("finance", model=TestModel("budget approved"))
    board = ParallelAgent("board", [legal, finance])
    print("parallel:", (await board.run("review contract")).output)

    # Swarm: triage hands off to a specialist.
    triage = Agent(
        "triage", model=TestModel(custom_output="routing", call_tools=["handoff_to_billing"])
    )
    billing = Agent("billing", model=TestModel("refund processed"))
    swarm = Swarm("support", [triage, billing], entry="triage")
    print("swarm:", (await swarm.run("I was double charged", deps=SwarmState())).output)


asyncio.run(main())
