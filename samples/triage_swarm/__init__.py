"""Triage swarm: a front-line agent hands off to the right specialist.

Pattern: autonomous multi-agent routing. The triage agent decides whether a
request is billing or technical and hands off; the chosen specialist answers.
"""

from __future__ import annotations

from typing import Any

from yaab import Agent, Swarm
from yaab.models.test_model import TestModel
from yaab.multiagent import SwarmState

from .._common import resolve_model


def build(model: Any = None, *, route_to: str = "billing") -> Swarm:
    """Return a triage→{billing,tech} swarm.

    Offline, the triage agent is scripted to hand off to ``route_to`` so the
    path is deterministic; with a real model it decides on its own.
    """
    triage = Agent(
        "triage",
        model=resolve_model(
            model,
            offline_default=TestModel(
                custom_output="routing", call_tools=[f"handoff_to_{route_to}"]
            ),
        ),
        instructions="Route the user to 'billing' or 'tech' by calling the handoff tool.",
    )
    billing = Agent(
        "billing",
        model=resolve_model(model, offline_default=TestModel("Your refund is on the way.")),
        instructions="Handle billing and refund questions.",
    )
    tech = Agent(
        "tech",
        model=resolve_model(model, offline_default=TestModel("Try restarting the device.")),
        instructions="Handle technical troubleshooting.",
    )
    return Swarm("support_swarm", [triage, billing, tech], entry="triage", max_handoffs=3)


async def run(
    message: str = "I was charged twice", model: Any = None, route_to: str = "billing"
) -> str:
    swarm = build(model, route_to=route_to)
    result = await swarm.run(message, deps=SwarmState())
    return result.output
