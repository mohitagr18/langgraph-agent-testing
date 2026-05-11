"""
Node functions for the expense report processing agent.

ARCHITECTURAL RULE: This module imports from src.state — never from src.graph.
That one-way dependency is what makes unit testing possible: you can import any
function here and call it directly with a hand-crafted state dict without ever
touching the compiled graph object.

Each function is a pure transformation:
    state_dict_in → partial_state_dict_out

The graph layer in graph.py wires these functions together, but they have no
knowledge of one another and no knowledge that a graph exists.
"""

from __future__ import annotations

from src.state import ExpenseState

# ── Business-rule constants ───────────────────────────────────────────────────

# Expenses at or above this threshold always require a human approver.
HIGH_AMOUNT_THRESHOLD = 500.00

# Vendors that have appeared in previous policy violations. Any expense from
# these vendors is flagged regardless of amount.
FLAGGED_VENDORS: set[str] = {
    "casino royale",
    "ultra luxury resort",
    "first class airways",
    "platinum club",
}

# Categories that are exempt from the high-amount threshold
# (e.g. equipment purchases are pre-budgeted and follow a separate workflow).
THRESHOLD_EXEMPT_CATEGORIES: set[str] = {"equipment", "software"}


# ── Node 1: classify_expense ──────────────────────────────────────────────────

def classify_expense(state: ExpenseState) -> dict:
    """Examine the expense and decide whether it needs human review.

    Business rules (in priority order):
    1. If the vendor is on the flagged list → risk_flag = "FLAGGED_VENDOR",
       requires_approval = True.
    2. If amount >= HIGH_AMOUNT_THRESHOLD AND category is not exempt →
       risk_flag = "HIGH_AMOUNT", requires_approval = True.
    3. Otherwise → risk_flag = None, requires_approval = False.

    This function is deterministic and side-effect-free.  Unit tests can call
    it directly with any state dict — no graph, no LLM, no network.
    """
    vendor_lower = state["vendor"].strip().lower()
    category_lower = state["category"].strip().lower()

    if vendor_lower in FLAGGED_VENDORS:
        return {
            "risk_flag": "FLAGGED_VENDOR",
            "requires_approval": True,
        }

    if (
        state["amount"] >= HIGH_AMOUNT_THRESHOLD
        and category_lower not in THRESHOLD_EXEMPT_CATEGORIES
    ):
        return {
            "risk_flag": "HIGH_AMOUNT",
            "requires_approval": True,
        }

    return {
        "risk_flag": None,
        "requires_approval": False,
    }


# ── Node 2: request_human_approval ────────────────────────────────────────────

def request_human_approval(state: ExpenseState) -> dict:
    """Pause the graph and ask a human approver to review the flagged expense.

    This node uses LangGraph's dynamic interrupt() to suspend execution.
    The graph will remain paused at this checkpoint until a caller resumes it
    with Command(resume={"decision": "approved"|"rejected", "notes": "..."}).

    HITL two-step pattern:
      Step 1 — graph.invoke(initial_state, config)
               → graph pauses here, state is checkpointed
      Step 2 — graph.invoke(Command(resume=payload), config)
               → this function continues from the interrupt() call with the
                 resume value injected as the return value of interrupt()

    IMPORTANT: Do NOT wrap interrupt() in try/except.  LangGraph uses a special
    exception internally to implement the pause mechanism.  Catching it would
    break resumption silently.
    """
    # Import here (not at module top) so that unit tests for *other* nodes
    # can import this module without triggering the langgraph interrupt machinery.
    from langgraph.types import interrupt

    reviewer_payload = interrupt(
        {
            "expense_id": state["expense_id"],
            "vendor": state["vendor"],
            "amount": state["amount"],
            "category": state["category"],
            "submitted_by": state["submitted_by"],
            "risk_flag": state["risk_flag"],
            "prompt": (
                f"Expense #{state['expense_id']} from {state['submitted_by']} "
                f"requires your approval. "
                f"Vendor: {state['vendor']}, "
                f"Amount: ${state['amount']:.2f}, "
                f"Flag: {state['risk_flag']}. "
                "Respond with {'decision': 'approved'|'rejected', 'notes': '...'}"
            ),
        }
    )

    # reviewer_payload is whatever value was passed to Command(resume=...)
    decision = reviewer_payload.get("decision", "rejected")
    notes = reviewer_payload.get("notes", "")

    return {
        "approval_decision": decision,
        "reviewer_notes": notes,
    }


# ── Node 3: finalize_expense ──────────────────────────────────────────────────

def finalize_expense(state: ExpenseState) -> dict:
    """Compute the final status and write a human-readable audit trail.

    Routing logic:
    - If requires_approval is False → cleared automatically (no human touched it)
    - If requires_approval is True AND approval_decision == "approved" → approved
    - If requires_approval is True AND approval_decision == "rejected" → rejected
    - If requires_approval is True but approval_decision is somehow missing →
      default to "rejected" (fail-safe: never auto-approve ambiguous state)

    The audit_trail field is intentionally verbose because it is the artifact
    that compliance teams inspect.  This function is pure and unit-testable.
    """
    if not state["requires_approval"]:
        final_status = "cleared"
        audit = (
            f"Expense #{state['expense_id']} auto-cleared. "
            f"Vendor: {state['vendor']}, "
            f"Amount: ${state['amount']:.2f}, "
            f"Category: {state['category']}. "
            "No policy flags triggered; no human review required."
        )
    else:
        decision = state.get("approval_decision") or "rejected"
        notes = state.get("reviewer_notes") or "(no notes provided)"
        final_status = decision  # "approved" or "rejected"
        action = "approved" if decision == "approved" else "rejected"
        audit = (
            f"Expense #{state['expense_id']} {action} by human reviewer. "
            f"Vendor: {state['vendor']}, "
            f"Amount: ${state['amount']:.2f}, "
            f"Flag: {state['risk_flag']}, "
            f"Submitted by: {state['submitted_by']}. "
            f"Reviewer notes: {notes}"
        )

    return {
        "final_status": final_status,
        "audit_trail": audit,
    }
