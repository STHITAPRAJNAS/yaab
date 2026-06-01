"""A robust, out-of-the-box agent: built-in tools, context mgmt, HITL approval,
resilience, and declarative config. Runs offline with TestModel.
"""

import asyncio

from yaab import Agent, Runner, SummarizeHistory, agent_from_dict, tool
from yaab.governance import AuditLog, ToolApprovalPlugin
from yaab.models.base import ModelResponse
from yaab.models.resilient import CircuitBreaker, RateLimiter, ResilientModel
from yaab.models.test_model import TestModel
from yaab.tools.builtin import calculator, default_toolset
from yaab.types import RunContext, ToolCall


@tool
def refund(amount: int = 0) -> str:
    """Issue a refund."""
    return f"refunded {amount}"


async def main() -> dict:
    """Exercise the robustness features and return each one's result."""
    # 1) Built-in tools — no need to write your own.
    tool_names = [t.name for t in default_toolset()]
    print("tools:", tool_names)
    calc = await calculator.execute(RunContext(), expression="6 * 7")
    print("calc 6*7:", calc)

    # 2) Declarative agent (auditable config, not code).
    config_agent = agent_from_dict(
        {
            "name": "assistant",
            "model": "openai/gpt-4o",
            "instructions": "Be concise.",
            "tools": ["calculator", "current_time"],
            "max_steps": 5,
        }
    )
    print("config agent:", config_agent.name, "with", len(config_agent.tools), "tools")

    # 3) Context-window management + resilience on a real run (offline model).
    resilient = ResilientModel(
        TestModel("The answer is 42."),
        rate_limiter=RateLimiter(rate=60, per=60),
        circuit_breaker=CircuitBreaker(threshold=3, cooldown=30),
    )
    robust = Agent("robust", model=resilient, context_strategy=SummarizeHistory(max_tokens=4000))
    robust_out = (await robust.run("What is the meaning of life?")).output
    print("robust run:", robust_out)

    # 4) HITL approval for a sensitive tool on the fast path.
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

    return {
        "tool_names": tool_names,
        "calc": calc,
        "config_agent_tools": len(config_agent.tools),
        "robust_out": robust_out,
        "approved_run": result.output,
    }


if __name__ == "__main__":
    asyncio.run(main())
