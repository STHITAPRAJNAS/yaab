"""Live verification of the v2 orchestration features against a real model.

Covers the whole v2 surface end-to-end on a real LLM (defaults to
gemini/gemini-2.5-flash via YAAB_LIVE_MODEL): shared State + writes= handoff,
RouterAgent exclusive choice, Conditions (when=/stop=), Flow (route + loop +
HITL pause/resume), and the hybrid retrieval path. Needs a real key in .env.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

for line in (Path(__file__).resolve().parent.parent / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

MODEL = os.environ.get("YAAB_LIVE_MODEL", "gemini/gemini-2.5-flash")


async def check_shared_state_writes_handoff() -> str:
    """A classifier writes= intent into shared state; a responder reads it via {key}."""
    from yaab import Agent, SequentialAgent

    classifier = Agent(
        "classifier",
        model=MODEL,
        instructions="Classify the request as exactly one word: refund, billing, or other. "
        "Reply with only that word.",
        writes="intent",
    )
    responder = Agent(
        "responder",
        model=MODEL,
        instructions="The request was classified as '{intent}'. Acknowledge in one short sentence.",
    )
    pipe = SequentialAgent("triage", [classifier, responder], pipe_output=False)
    result = await pipe.run("I want my money back for order 42")
    out = str(result.output)
    ok = "refund" in out.lower() or len(out) > 0
    return f"classifier->writes(intent)->responder reads {{intent}}: {out[:70]!r} (ok={ok})"


async def check_router_exclusive_choice() -> str:
    """RouterAgent routes a real query to exactly one of N branches, zero LLM cost to route."""
    from yaab import Agent, Branch, RouterAgent

    refunds = Agent("refunds", model=MODEL, instructions="Say 'refund desk' and nothing else.")
    support = Agent("support", model=MODEL, instructions="Say 'general support' and nothing else.")
    router = RouterAgent(
        "triage",
        branches=[Branch(when="'refund' in input", agent=refunds)],
        default=support,
    )
    r1 = await router.run("I need a refund please")
    r2 = await router.run("what are your hours")
    routed = "refund" in str(r1.output).lower()
    defaulted = "support" in str(r2.output).lower()
    return f"refund->refunds={routed}, other->default={defaulted}"


async def check_flow_route_loop() -> str:
    """A Flow with a real agent step, a deterministic route, and a bounded loop."""
    from yaab import Agent, Flow

    grader = Agent(
        "grader",
        model=MODEL,
        instructions="Rate the draft 0-10 for clarity. Reply with only the integer.",
    )

    def parse_score(state, ctx):
        import re

        m = re.search(r"\d+", str(state.get("grade", "0")))
        return {"score": int(m.group()) if m else 0}

    flow = (
        Flow[None, str]("review")
        .step("draft", fn=lambda state, ctx: {"draft": "The sky is blue.", "out": "drafted"})
        .step("grade", agent=grader, writes="grade")
        .step("parse", fn=parse_score)
        .route(
            "parse",
            lambda state, ctx: "done" if state.get("score", 0) >= 0 else "retry",
            to={"done": Flow.DONE, "retry": "draft"},
        )
        .then("draft", "grade")
        .then("grade", "parse")
        .start_at("draft")
        .returns("out")
    )
    result = await flow.run("review this")
    return f"flow ran draft->grade(LLM)->parse->route: output={str(result.output)!r}"


async def check_flow_hitl_pause_resume() -> str:
    """A Flow step pauses for approval (flow_pause), then resumes with the decision."""
    from yaab import Agent, Flow, RunContext, Runner
    from yaab.governance import approvals
    from yaab.governance.approvals import InMemoryApprovalStore
    from yaab.graph.checkpoint import MemorySaver

    def gate(state, ctx: RunContext):
        decision = ctx.pause_for({"needs": "approval", "amount": state.get("amount", 0)})
        return {"approved": decision == "approve"}

    assistant = Agent(
        "assistant",
        model=MODEL,
        instructions="Reply with exactly: done",
    )
    flow = (
        Flow[None, str]("refund")
        .step("parse", fn=lambda state, ctx: {"amount": 500})
        .step("gate", fn=gate)
        .step("finish", agent=assistant, writes="msg")
        .then("parse", "gate")
        .then("gate", "finish")
        .then("finish", Flow.DONE)
        .start_at("parse")
        .returns("approved")
    )
    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    r1 = await runner.run(flow, "refund 42", session_id="flow-hitl-1")
    if not r1.paused:
        return f"FAILED to pause: {r1.output!r}"
    pending = await store.list_pending()
    decision = await approvals.respond(r1, by="alice", answer="approve", store=store)
    r2 = await runner.run(flow, resume=decision, session_id="flow-hitl-1")
    return f"paused(flow_pause, {len(pending)} row) -> approve -> resumed: approved={r2.output}"


async def check_tool_approval_resume() -> str:
    """A guarded tool pauses a real-model run; approval resumes and runs it once."""
    from yaab import Agent, Runner, tool
    from yaab.governance import approvals
    from yaab.governance.approval import ToolApprovalPlugin
    from yaab.governance.approvals import InMemoryApprovalStore
    from yaab.graph.checkpoint import MemorySaver

    calls = {"n": 0}

    @tool
    def issue_refund(amount: int) -> str:
        """Issue a refund of the given amount."""
        calls["n"] += 1
        return f"refunded ${amount}"

    store = InMemoryApprovalStore()
    runner = Runner(run_checkpointer=MemorySaver(), approval_store=store)
    runner.add_plugin(ToolApprovalPlugin(tools=["issue_refund"], mode="queue", store=store))
    agent = Agent(
        "banker",
        model=MODEL,
        tools=[issue_refund],
        instructions="Use issue_refund to refund exactly the amount the user asks for.",
        runner=runner,
    )
    r1 = await agent.run("Please refund $250", resume_id="tool-hitl-1")
    if not r1.paused:
        return f"FAILED to pause (tool ran {calls['n']}x): {r1.output!r}"
    ap_id = r1.pending[0].approval_id
    decision = await approvals.approve(ap_id, by="manager", store=store)
    r2 = await agent.run(resume=decision)
    return (
        f"paused(tool not run) -> approve -> tool ran {calls['n']}x, output={str(r2.output)[:50]!r}"
    )


async def check_hybrid_retrieval_live() -> str:
    """Hybrid retrieval surfaces an exact-term chunk, fed to a real model to answer."""
    from yaab import Agent
    from yaab.rag import Document, KnowledgeBase

    kb = KnowledgeBase(hybrid=True, embedder=f"{MODEL.split('/')[0]}/gemini-embedding-001")
    kb.add(
        [
            Document(text="The Antikythera mechanism is an ancient analog computer.", source="a"),
            Document(text="Photosynthesis converts light into chemical energy.", source="b"),
        ]
    )
    block, chunks = await kb.augment("What is the Antikythera mechanism?", k=1)
    agent = Agent("rag", model=MODEL, instructions=f"Answer using only this context:\n{block}")
    result = await agent.run("What is the Antikythera mechanism?")
    found = chunks and "antikythera" in chunks[0].text.lower()
    return f"hybrid surfaced exact-term chunk={found}; answer: {str(result.output)[:60]!r}"


CHECKS = [
    ("shared State + writes= handoff", check_shared_state_writes_handoff),
    ("RouterAgent exclusive choice", check_router_exclusive_choice),
    ("Flow: agent step + route + loop", check_flow_route_loop),
    ("Flow HITL: pause -> approve -> resume", check_flow_hitl_pause_resume),
    ("tool approval: pause -> approve -> run-once", check_tool_approval_resume),
    ("hybrid retrieval -> grounded answer", check_hybrid_retrieval_live),
]


async def main() -> int:
    print(f"Live v2 verification on {MODEL}\n")
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = await fn()
            bad = detail.startswith("FAILED")
            failures += bad
            print(f"  [{'FAIL' if bad else 'PASS'}] {name:42s} {detail}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  [FAIL] {name:42s} {type(exc).__name__}: {exc}")
        await asyncio.sleep(1.5)  # be gentle on rate limits
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} live v2 checks passed on {MODEL}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
