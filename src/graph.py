"""
Compiled LangGraph expense agent.

This is the only file that knows the graph exists.  It imports node functions
from src.nodes and wires them together.  src.nodes imports nothing from here —
that one-way dependency is what makes the unit tests in tests/test_unit.py
work without any graph machinery at all.

The compiled graph object exported as `graph` is also referenced in
langgraph.json for platform deployment:
    "expense_agent": "./src/graph.py:graph"
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes import classify_expense, finalize_expense, request_human_approval
from src.state import ExpenseState


def _needs_approval(state: ExpenseState) -> str:
    """Conditional edge: route to HITL gate or skip straight to finalization."""
    return "approval_gate" if state["requires_approval"] else "finalize"


def build_graph(checkpointer=None):
    """Build and compile the expense agent graph.

    Args:
        checkpointer: A LangGraph checkpointer instance.  Defaults to an
            in-memory MemorySaver, which is sufficient for local dev and tests.
            Swap in SqliteSaver or PostgresSaver for persistence across restarts.

    Returns:
        A compiled CompiledGraph ready to invoke.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(ExpenseState)

    # Register nodes
    builder.add_node("classify", classify_expense)
    builder.add_node("approval_gate", request_human_approval)
    builder.add_node("finalize", finalize_expense)

    # Wire edges
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", _needs_approval)
    builder.add_edge("approval_gate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# ── Module-level graph instance ───────────────────────────────────────────────
# LangGraph Platform reads this object via the path in langgraph.json.
# Tests may import build_graph() directly to get a fresh instance with a
# fresh checkpointer, avoiding state bleed between test runs.
graph = build_graph()
