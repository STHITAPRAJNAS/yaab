"""The ``ask_user`` built-in: an agent parks the run to ask a person a question.

Pins the question pause kind, which reuses the approval pause machinery:

* calling ``ask_user`` persists a ``question`` pending record and pauses durably;
* ``result.pending[0].kind == "question"`` carries the prompt and (optional)
  answer schema;
* ``approvals.respond(answer=...)`` validates the typed answer against the schema
  and, on resume, that answer becomes ``ask_user``'s return value — read inline by
  the model.
"""

from __future__ import annotations

import pytest

from yaab import Agent, Pending, ask_user
from yaab.governance import (
    ApprovalDecision,
    InMemoryApprovalStore,
    ToolApprovalPlugin,
    approvals,
)
from yaab.governance.approvals_decide import DecisionValidationError
from yaab.graph.checkpoint import MemorySaver
from yaab.models.base import ModelResponse
from yaab.models.test_model import TestModel
from yaab.runner import Runner
from yaab.types import ToolCall


def _asks_then_answers(args: dict) -> TestModel:
    return TestModel(
        responses=[
            ModelResponse(
                tool_calls=[ToolCall(name="ask_user", arguments=args)],
                finish_reason="tool_calls",
            ),
            "booked a table for the requested party",
        ]
    )


def _build(store, model):
    runner = Runner(
        run_checkpointer=MemorySaver(),
        plugins=[ToolApprovalPlugin(tools=["ask_user"], mode="queue", store=store)],
    )
    return Agent("concierge", model=model, tools=[ask_user], runner=runner)


# --------------------------------------------------------------------------
# ask_user is exported and is a real built-in tool.
# --------------------------------------------------------------------------
def test_ask_user_is_exported_tool():
    assert ask_user.name == "ask_user"
    schema = ask_user.schema()
    assert schema["function"]["name"] == "ask_user"


# --------------------------------------------------------------------------
# Calling ask_user pauses with a question pending carrying the prompt.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_user_pauses_with_question_pending():
    store = InMemoryApprovalStore()
    agent = _build(store, _asks_then_answers({"question": "For how many people?"}))

    result = await agent.run("Book me a table tonight", resume_id="q1")
    assert result.paused
    assert len(result.pending) == 1
    p = result.pending[0]
    assert isinstance(p, Pending)
    assert p.kind == "question"
    assert p.prompt == "For how many people?"

    # A question record is in the store, visible to the approvals surface.
    pending_rows = await store.list_pending()
    assert len(pending_rows) == 1
    assert pending_rows[0].kind == "question"
    assert pending_rows[0].prompt == "For how many people?"


# --------------------------------------------------------------------------
# respond's typed answer becomes ask_user's return value on resume.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_respond_answer_is_returned_to_the_model():
    store = InMemoryApprovalStore()
    agent = _build(
        store,
        _asks_then_answers(
            {"question": "For how many people?", "answer_schema": {"type": "integer", "minimum": 1}}
        ),
    )

    result = await agent.run("Book me a table tonight", resume_id="q2")
    p = result.pending[0]
    assert p.answer_schema == {"type": "integer", "minimum": 1}

    decision = await approvals.respond(result, by="user", answer=4, store=store)
    assert decision.answer == 4

    final = await agent.run(resume=decision)
    assert not final.paused
    assert final.output == "booked a table for the requested party"

    # The answer (4) was injected as the ask_user tool result on resume.
    tool_results = [
        ev.payload.get("result") for ev in final.events if ev.type.value == "tool_result"
    ]
    assert 4 in tool_results


# --------------------------------------------------------------------------
# A typed answer that violates the schema is rejected before any write.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_typed_answer_violating_schema_is_rejected():
    store = InMemoryApprovalStore()
    agent = _build(
        store,
        _asks_then_answers(
            {"question": "How many?", "answer_schema": {"type": "integer", "minimum": 1}}
        ),
    )
    result = await agent.run("Book me a table", resume_id="q3")
    approval_id = result.pending[0].approval_id

    with pytest.raises(DecisionValidationError):
        await approvals.respond(result, by="user", answer=0, store=store)  # below minimum
    # Nothing was written: the question is still pending.
    row = await store.get(approval_id)
    assert row.decision is ApprovalDecision.PENDING
