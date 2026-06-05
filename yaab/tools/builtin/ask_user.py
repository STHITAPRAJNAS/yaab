"""``ask_user`` — let an agent pause and ask the human a question.

When an agent needs a fact it does not have ("for how many people?", "which
address?"), it calls ``ask_user`` with the question (and, optionally, a JSON
Schema describing the answer it expects). The run pauses durably with a
``question`` pending; the human's validated answer is returned here as this
tool's result, so the model reads it inline and continues.

This reuses the same pause/resume machinery as tool approval — it is not a second
subsystem. Guard ``ask_user`` with a :class:`~yaab.governance.ToolApprovalPlugin`
in ``queue`` mode (so the question is persisted and decidable out of band), and
answer it with :func:`yaab.governance.approvals.respond`::

    from yaab import Agent
    from yaab.tools.builtin import ask_user
    from yaab.governance import ToolApprovalPlugin, InMemoryApprovalStore, approvals

    store = InMemoryApprovalStore()
    agent = Agent("concierge", tools=[ask_user],
                  hitl=ToolApprovalPlugin(tools=["ask_user"], mode="queue", store=store))

    result = await agent.run("Book me a table tonight", resume_id="r")
    if result.paused and result.pending[0].kind == "question":
        answer = await approvals.respond(result, by="user", answer=4, store=store)
        result = await agent.run(resume=answer)
"""

from __future__ import annotations

from typing import Any

from ..base import tool


@tool
async def ask_user(question: str, answer_schema: dict[str, Any] | None = None) -> Any:
    """Ask the human a question and pause until they answer.

    The run pauses with a ``question`` pending carrying ``question`` (and the
    optional ``answer_schema``). The human's validated answer is returned here as
    this tool's result on resume, so the model reads it inline.

    ``answer_schema`` is an optional JSON Schema (e.g. ``{"type": "integer",
    "minimum": 1}``) the answer is validated against before it is recorded.

    This body only runs when no human-in-the-loop pause is wired (no approval
    store): then there is no person to ask, and the call surfaces the unanswered
    question text so the caller can wire one in. In the normal paused flow the
    body never executes — the resume seam injects the human's answer as the
    return value directly.
    """
    return (
        f"error: ask_user('{question}') needs a human-in-the-loop store "
        "to pause and collect an answer"
    )
