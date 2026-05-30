"""Validate every sample runs offline (deterministic TestModel)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_customer_support():
    from samples import customer_support

    out = await customer_support.run("How long do refunds take?")
    assert "refund" in out.lower()


@pytest.mark.asyncio
async def test_research_assistant():
    from samples import research_assistant

    out = await research_assistant.run("unit testing")
    assert isinstance(out, str) and out


@pytest.mark.asyncio
async def test_document_qa():
    from samples import document_qa

    out = await document_qa.run("How much PTO do I get?")
    assert "answer" in out
    assert out["citations"]  # retrieval produced cited chunks
    assert any(
        "pto" in c.lower() or "policy" in c.lower() or "security" in c.lower()
        for c in out["citations"]
    )


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

    out = await coding_helper.run("sum 0..9")
    assert "45" in out


@pytest.mark.asyncio
async def test_coding_helper_rejected_path():
    from samples import coding_helper

    # With approval denied, the tool returns an error the model adapts to;
    # the run still completes without raising.
    agent, runner = coding_helper.build(auto_approve=False)
    result = await runner.run(agent, "compute something")
    assert isinstance(result.output, str)
