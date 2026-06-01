"""Validate every sample runs offline (deterministic TestModel)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_customer_support():
    from samples import customer_support

    out = await customer_support.run("How long do refunds take?")
    assert "refund" in out.lower()


@pytest.mark.asyncio
async def test_personal_assistant_sessions_and_memory():
    from samples import personal_assistant

    out = await personal_assistant.run()
    # SQLite session persisted both turns of session 1 (2 user + 2 assistant msgs).
    assert out["session1_history_messages"] == 4
    # The callback wrote the durable fact to long-term memory...
    assert any("alice" in f.lower() for f in out["learned_facts"])
    # ...and a brand-new session recalls it from long-term memory.
    assert any("alice" in r.lower() for r in out["recalled_in_session2"])
    assert out["total_tokens"] > 0  # usage callback accumulated tokens


@pytest.mark.asyncio
async def test_memory_patterns_episodic_vs_long_term():
    from samples import memory_patterns

    out = await memory_patterns.run()
    # Episodic memory lives in the session; a fresh session starts empty.
    assert out["episodic_turns_in_trip_session"] == 6
    assert out["new_session_episodic_turns"] == 0
    # The episode's user turns were consolidated into long-term memory...
    assert out["consolidated_to_long_term"] == 3
    # ...and recalled across sessions, scoped to this user only.
    assert any("vegetarian" in t.lower() for t in out["recalled_dietary"])
    assert out["other_user_recall_count"] == 0


@pytest.mark.asyncio
async def test_multi_agent_state_handoff():
    from samples import multi_agent_state

    out = await multi_agent_state.run()
    # The extractor wrote the ticket; the reporter read exactly that, via shared deps.
    assert out["ticket_written_by_extractor"] == {"intent": "refund", "order_id": "A100"}
    assert out["ticket_read_by_reporter"] == out["ticket_written_by_extractor"]
    assert out["handoff_ok"] is True


def test_approval_pipeline_approved():
    from samples import approval_pipeline

    state = approval_pipeline.run(amount=5000, approve_decision=True)
    assert state["status"] == "EXECUTED"


def test_approval_pipeline_rejected():
    from samples import approval_pipeline

    state = approval_pipeline.run(amount=5000, approve_decision=False)
    assert state["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_triage_swarm_routes_to_billing():
    from samples import triage_swarm

    out = await triage_swarm.run("I was charged twice", route_to="billing")
    assert "refund" in out.lower()


@pytest.mark.asyncio
async def test_triage_swarm_routes_to_tech():
    from samples import triage_swarm

    out = await triage_swarm.run("my app crashes", route_to="tech")
    assert "restart" in out.lower()


@pytest.mark.asyncio
async def test_coding_helper_approved():
    from samples import coding_helper
    from yaab import EventType

    agent, runner = coding_helper.build()
    result = await runner.run(agent, "Compute the sum of 0..9 in Python.")

    # The sandboxed python_exec tool must have actually executed and produced
    # the answer -- a scripted final message alone is not a passing run.
    tool_results = [
        e
        for e in result.events
        if e.type is EventType.TOOL_RESULT and e.payload.get("name") == "python_exec"
    ]
    assert tool_results, "python_exec never executed"
    executed = str(tool_results[0].payload["result"])
    assert "error" not in executed.lower(), f"python_exec failed: {executed}"
    assert "45" in executed, f"sandbox did not compute the sum: {executed!r}"
    assert "45" in result.output


@pytest.mark.asyncio
async def test_coding_helper_rejected_path():
    from samples import coding_helper

    # With approval denied, the tool returns an error the model adapts to;
    # the run still completes without raising.
    agent, runner = coding_helper.build(auto_approve=False)
    result = await runner.run(agent, "compute something")
    assert isinstance(result.output, str)
