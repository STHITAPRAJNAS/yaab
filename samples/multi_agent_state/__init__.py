"""Multi-agent state — passing data from one agent to the next via shared ``deps``.

In a pipeline, a later agent usually needs what an earlier agent produced. YAAB
does this the type-safe, dependency-injection way: you hand
every agent the *same* ``deps`` object, and tools read/write it through
``ctx.deps``. No globals, no hidden blackboard — the shared state is an explicit,
typed object you own.

This sample wires two agents into a ``SequentialAgent``:

1. **extractor** — reads the request and calls ``record_ticket`` to write the
   structured facts (intent, order id) into the shared workspace.
2. **reporter** — calls ``read_workspace`` to pull exactly what the extractor
   wrote and produces the customer-facing summary.

``SequentialAgent.run(prompt, deps=workspace)`` passes the one ``workspace`` to
both agents, so the reporter sees the extractor's writes. We assert on the
``workspace`` object afterwards, which is the deterministic proof that state
flowed across the agent boundary.

Note: the prefix-scoped ``ctx.state`` dict (``temp:``/``user:``/``app:``) is for
*scratch within a single run*; cross-agent, cross-run data uses ``deps`` as shown
here.

    python -m samples.multi_agent_state
    YAAB_SAMPLE_MODEL=ollama/llama3 python -m samples.multi_agent_state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from yaab import Agent, RunContext, SequentialAgent, tool
from yaab.models.base import ModelResponse
from yaab.models.test_model import FunctionModel
from yaab.types import Role, ToolCall

from .._common import resolve_model


def _tool_call_response(name: str, args: dict[str, Any]) -> ModelResponse:
    """A model response that calls exactly one tool (keeps the offline brains terse)."""
    return ModelResponse(
        tool_calls=[ToolCall(name=name, arguments=args)],
        finish_reason="tool_calls",
        model="function",
    )


def _has_tool_result(messages: list[Any]) -> bool:
    return any(getattr(m, "role", None) == Role.TOOL for m in messages)


@dataclass
class SupportWorkspace:
    """The shared state object passed between agents via ``deps``.

    The extractor writes ``ticket``; the reporter reads it and records what it
    saw in ``seen_by_reporter`` — so we can prove the hand-off happened.
    """

    ticket: dict[str, str] = field(default_factory=dict)
    seen_by_reporter: dict[str, str] = field(default_factory=dict)


@tool
def record_ticket(ctx: RunContext, intent: str, order_id: str) -> str:
    """Record the classified support ticket into the shared workspace."""
    ctx.deps.ticket = {"intent": intent, "order_id": order_id}
    return f"recorded: {intent} / {order_id}"


@tool
def read_workspace(ctx: RunContext) -> str:
    """Read the ticket the previous agent wrote into the shared workspace."""
    ticket = dict(ctx.deps.ticket)
    ctx.deps.seen_by_reporter = ticket  # prove the reporter read the extractor's write
    return f"intent={ticket.get('intent')} order={ticket.get('order_id')}"


def _extractor_offline() -> FunctionModel:
    """Offline brain for the extractor: first call records the ticket, then confirms."""

    def fn(messages: list[Any]):
        if not _has_tool_result(messages):
            return _tool_call_response("record_ticket", {"intent": "refund", "order_id": "A100"})
        return "Classified the request and saved it for the next step."

    return FunctionModel(fn)


def _reporter_offline() -> FunctionModel:
    """Offline brain for the reporter: first call reads the workspace, then summarizes."""

    def fn(messages: list[Any]):
        if not _has_tool_result(messages):
            return _tool_call_response("read_workspace", {})
        return "Customer summary: a refund was requested for order A100."

    return FunctionModel(fn)


def build(model: Any = None) -> tuple[SequentialAgent, SupportWorkspace]:
    """Build the extractor -> reporter pipeline and a fresh shared workspace."""
    extractor = Agent(
        "extractor",
        model=resolve_model(model, offline_default=_extractor_offline()),
        instructions="Classify the request; call record_ticket with the intent and order id.",
        tools=[record_ticket],
        deps_type=SupportWorkspace,
    )
    reporter = Agent(
        "reporter",
        model=resolve_model(model, offline_default=_reporter_offline()),
        instructions="Call read_workspace to get the ticket, then write a one-line summary.",
        tools=[read_workspace],
        deps_type=SupportWorkspace,
    )
    pipeline = SequentialAgent("support_pipeline", [extractor, reporter])
    return pipeline, SupportWorkspace()


async def run(prompt: str = "I want a refund for order A100.", model: Any = None) -> dict[str, Any]:
    """Run the pipeline and return the shared state, proving the cross-agent hand-off."""
    pipeline, workspace = build(model)
    result = await pipeline.run(prompt, deps=workspace)
    return {
        "final_output": result.output,
        "ticket_written_by_extractor": workspace.ticket,
        "ticket_read_by_reporter": workspace.seen_by_reporter,
        "handoff_ok": workspace.ticket == workspace.seen_by_reporter and bool(workspace.ticket),
    }
