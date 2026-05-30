"""A robust, out-of-the-box agent: built-in tools, context mgmt, HITL approval,
resilience, and declarative config. Runs offline with TestModel.
"""

import asyncio

from yaab import Agent, Runner, SummarizeHistory, agent_from_dict
from yaab.governance import AuditLog, ToolApprovalPlugin
from yaab.models.resilient import CircuitBreaker, RateLimiter, ResilientModel
from yaab.models.test_model import TestModel
from yaab.tools.builtin import calculator, default_toolset


async def main():
    # 1) Built-in tools — no need to write your own.
    print("tools:", [t.name for t in default_toolset()])
    from yaab.types import RunContext

    print("calc 6*7:", await calculator.execute(RunContext(), expression="6 * 7"))

    # 2) Declarative agent (auditable config, not code).
    agent = agent_from_dict(
        {
            "name": "assistant",
            "model": "openai/gpt-4o",
            "instructions": "Be concise.",
            "tools": ["calculator", "current_time"],
            "max_steps": 5,
        }
    )
    print("config agent:", agent.name, "with", len(agent.tools), "tools")

    # 3) Context-window management + resilience on a real run (offline model).
    resilient = ResilientModel(
        TestModel("The answer is 42."),
        rate_limiter=RateLimiter(rate=60, per=60),
        circuit_breaker=CircuitBreaker(threshold=3, cooldown=30),
    )
    robust = Agent("robust", model=resilient, context_strategy=SummarizeHistory(max_tokens=4000))
    print("robust run:", (await robust.run("What is the meaning of life?")).output)

    # 4) HITL approval for a sensitive tool on the fast path.
    from yaab import tool
    from yaab.models.base import ModelResponse
    from yaab.types import ToolCall

    @tool
    def refund(amount: int = 0) -> str:
        """Issue a refund."""
        return f"refunded {amount}"

    async def approver(tool_name, args, ctx):
        print(f"  [approval] {tool_name}({args}) -> auto-approved")
        return True

    audit = AuditLog()
    model = TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="refund", arguments={"amount": 50})],
                finish_reason="tool_calls",
            ),
            "Refund processed.",
        ]
    )
    runner = Runner(plugins=[ToolApprovalPlugin(tools=["refund"], approver=approver, audit=audit)])
    result = await runner.run(Agent("billing", model=model, tools=[refund]), "refund 50")
    print("approved run:", result.output)


asyncio.run(main())
