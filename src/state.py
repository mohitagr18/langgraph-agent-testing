"""
State schema for the expense report processing agent.

Keeping the state definition in its own module lets both src/nodes.py and
src/graph.py import it without creating circular dependencies.  The rule is:
  - state.py  → imports nothing from this project
  - nodes.py  → imports from state.py only
  - graph.py  → imports from nodes.py and state.py
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class ExpenseState(TypedDict):
    """Full state carried through every node of the expense agent.

    Fields are intentionally named after the expense domain so that every
    test assertion reads like a business rule, not a generic data structure.
    """

    # ── Input fields (set by the caller before the first node runs) ──────────
    expense_id: str           # Unique identifier for this expense report
    vendor: str               # Merchant / vendor name on the receipt
    category: str             # E.g. "meals", "travel", "lodging", "equipment"
    amount: float             # Amount in USD
    submitted_by: str         # Employee who filed the report

    # ── Computed fields (written by classification node) ─────────────────────
    risk_flag: Optional[str]  # None if clean; "HIGH_AMOUNT" | "FLAGGED_VENDOR" if flagged
    requires_approval: bool   # True when a human must review before the report clears

    # ── HITL fields (written during the interrupt/resume cycle) ──────────────
    approval_decision: Optional[str]   # "approved" | "rejected" | None (pre-review)
    reviewer_notes: Optional[str]      # Free-text from the approver (optional)

    # ── Output fields (written by the finalization node) ─────────────────────
    final_status: Optional[str]  # "cleared" | "approved" | "rejected"
    audit_trail: Optional[str]   # Human-readable summary of what happened and why
