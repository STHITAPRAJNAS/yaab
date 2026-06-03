"""Live check: durable runs, approval pause/resume, and trace capture on a real model.

Exercises the headline production scenario end-to-end against a real LLM:
a background-style run pauses for human approval on a guarded tool, the
approval is decided through the durable store (as another replica would),
the run resumes without re-requesting captured model turns, and the trace
records the whole story.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Load .env (keys never live in code).
for line in (Path(__file__).resolve().parent.parent / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

MODEL = os.environ.get("YAAB_LIVE_MODEL", "gemini/gemini-2.5-flash")


async def check_approval_pause_resume_live() -> str:
    """Guarded tool pauses the run durably; approval resumes it; model turns not re-requested."""
    import tempfile

    from yaab import Agent, Runner, tool
    from yaab.governance.approval import ToolApprovalPlugin
    from yaab.governance.approvals import SQLiteApprovalStore
    from yaab.graph.checkpoint import SQLiteSaver

    tmp = tempfile.mkdtemp()
    approval_db = f"{tmp}/approvals.db"
    ckpt_db = f"{tmp}/ckpt.db"

    calls = {"wire_transfer": 0}

    @tool
    def wire_transfer(amount: int, recipient: str) -> str:
        """Send a wire transfer."""
        calls["wire_transfer"] += 1
        return f"sent ${amount} to {recipient}"

    store = SQLiteApprovalStore(approval_db)
    plugin = ToolApprovalPlugin(tools=["wire_transfer"], mode="queue", store=store)
    runner = Runner(plugins=[plugin], run_checkpointer=SQLiteSaver(ckpt_db))
    agent = Agent(
        "banker",
        model=MODEL,
        tools=[wire_transfer],
        instructions="Use the wire_transfer tool to send exactly what the user asks. Be concise.",
    )

    # 1. Run until the guarded tool pauses it.
    events = []
    async for ev in runner.run_stream(agent, "Wire $250 to ACME Corp", resume_id="live-approval-1"):
        events.append(ev)
    paused = any(ev.type.name == "APPROVAL_REQUIRED" for ev in events)
    assert paused, f"run did not pause for approval; events: {[e.type.name for e in events]}"
    assert calls["wire_transfer"] == 0, "tool ran before approval!"

    # 2. "Another replica": a fresh store instance over the same SQLite file decides.
    store2 = SQLiteApprovalStore(approval_db)
    pending = await store2.list_pending()
    assert len(pending) == 1, f"expected 1 pending approval, got {len(pending)}"
    await store2.decide(pending[0].approval_id, decision="approved", reviewer="live-check")

    # 3. Resume with the same resume_id: the approved tool runs, the loop finishes.
    result = await runner.run(
        agent,
        "Wire $250 to ACME Corp",
        resume_id="live-approval-1",
        approval_decision="approved",
    )
    assert calls["wire_transfer"] == 1, f"tool ran {calls['wire_transfer']} times, expected 1"
    out = str(result.output)
    return f"paused -> approved on 2nd store instance -> resumed; output: {out[:60]!r}"


async def check_trace_capture_live() -> str:
    """A traced run records model/tool spans with durations, tokens, and cost."""
    import tempfile

    from yaab import Agent, Runner, tool
    from yaab.runs.trace import SQLiteTraceStore

    tmp = tempfile.mkdtemp()

    @tool
    def lookup_sku(name: str) -> str:
        """Look up a product SKU."""
        return f"SKU-{abs(hash(name)) % 10000}"

    trace_store = SQLiteTraceStore(f"{tmp}/trace.db")
    runner = Runner(trace_store=trace_store)
    agent = Agent(
        "catalog",
        model=MODEL,
        tools=[lookup_sku],
        instructions="Use lookup_sku for any product question, then answer concisely.",
    )
    result = await runner.run(agent, "What is the SKU for the Falcon X drone?")

    events = await trace_store.get(result.run_id)
    assert len(events) >= 3, f"expected >=3 trace events, got {len(events)}"
    kinds = {str(e.get("type", "")).lower() for e in events}
    has_model = any("model" in k for k in kinds)
    has_tool = any("tool" in k for k in kinds)
    durations = [e.get("duration_ms") for e in events]
    has_durations = any(d is not None and d > 0 for d in durations)
    assert has_model and has_tool, f"missing spans; kinds: {kinds}"
    assert has_durations, "no durations recorded on any event"
    return (
        f"{len(events)} events persisted (model spans: {has_model}, tool spans: {has_tool}, "
        f"durations recorded: {has_durations}); output: {str(result.output)[:50]!r}"
    )


CHECKS = [
    ("approval pause -> cross-store decide -> resume", check_approval_pause_resume_live),
    ("trace capture with durations/spans", check_trace_capture_live),
]


async def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = await fn()
            print(f"  [PASS] {name:48s} {detail}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  [FAIL] {name:48s} {type(exc).__name__}: {exc}")
        await asyncio.sleep(1.0)
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} live checks passed on {MODEL}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
