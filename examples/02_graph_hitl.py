"""Durable graph with a human-in-the-loop approval gate.

The graph pauses at the approval node (checkpointing its state), returns control
to you, and resumes from exactly where it left off when you supply the decision.
"""

from yaab.graph import START, Channel, MemorySaver, StateGraph


def draft(state, ctx):
    return {"draft": f"Wire transfer of ${state['amount']}"}


def approve(state, ctx):
    # Pause for a human decision; on resume this returns the supplied value.
    decision = ctx.interrupt({"review": state["draft"], "amount": state["amount"]})
    return {"approved": decision}


def execute(state, ctx):
    status = "EXECUTED" if state["approved"] else "REJECTED"
    return {"status": status}


def main() -> dict:
    """Run the draft -> approve -> execute graph, pausing for approval."""
    graph = StateGraph(channels={"amount": Channel(default=0)})
    graph.add_node("draft", draft)
    graph.add_node("approve", approve)
    graph.add_node("execute", execute)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "approve")
    graph.add_edge("approve", "execute")
    graph.set_finish_point("execute")

    app = graph.compile(checkpointer=MemorySaver())

    # First call runs until the interrupt, then pauses.
    paused = app.invoke({"amount": 10_000}, thread_id="txn-1")
    print("paused:", paused.interrupted, "-> needs review of:", paused.interrupt_value)

    # A human approves; resume the same thread.
    done = app.invoke(thread_id="txn-1", resume=True)
    print("final state:", done.state)

    return {"paused": paused, "done": done}


if __name__ == "__main__":
    main()
