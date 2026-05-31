"""Customer-support agent: RAG over a help center + an order-lookup tool, run
under enforcing governance (registry + guardrails + audit).

Pattern: knowledge-grounded support bot with a side-effecting tool and an audit
trail — the shape most "talk to your docs + act" assistants take.
"""

from __future__ import annotations

from typing import Any

from yaab import Agent, KnowledgeBase, Runner, tool
from yaab.governance import AgentCard, GovernanceMode, GovernanceService, RiskTier
from yaab.models.test_model import TestModel

from .._common import resolve_model

_HELP_DOCS = [
    ("refunds.md", "Refunds are processed within 5 business days to the original payment method."),
    ("shipping.md", "Standard shipping takes 3-5 business days. Express ships next day."),
    ("returns.md", "Items can be returned within 30 days with the order number."),
]

_ORDERS = {"A100": "shipped", "A200": "processing"}


@tool
def order_status(order_id: str) -> str:
    """Look up the status of an order by its id."""
    return _ORDERS.get(order_id, "unknown order")


def build(model: Any = None) -> tuple[Agent, Runner]:
    """Return a governed support agent + a Runner wired with governance."""
    kb = KnowledgeBase(name="helpcenter")
    for source, text in _HELP_DOCS:
        kb.add_text(text, source=source)

    gov = GovernanceService(mode=GovernanceMode.ENFORCING)
    gov.registry.register(
        AgentCard(
            agent_id="support-bot",
            name="Support Bot",
            intended_use_case="Customer support triage and FAQ",
            risk_tier=RiskTier.LIMITED,
            model_approval_status="approved",  # pre-approved for the sample
        )
    )

    # Offline default: scripted to call the help-center tool, then answer.
    offline = TestModel(
        custom_output="Refunds take about 5 business days.",
        call_tools=["search_helpcenter"],
    )
    agent = Agent(
        "Support Bot",
        model=resolve_model(model, offline_default=offline),
        instructions="You are a support agent. Use the help center and order tool. Be concise.",
        tools=[kb.as_tool(), order_status],
        registry_id="support-bot",
    )
    runner = Runner(governance=gov)
    return agent, runner


async def run(query: str = "How long do refunds take?", model: Any = None) -> str:
    agent, runner = build(model)
    result = await runner.run(agent, query, identity="customer:demo")
    return result.output
