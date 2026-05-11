"""
Layer 1 — Unit Tests: individual node functions in complete isolation.

What this layer proves:
  - Each classification rule fires on the right inputs
  - Edge cases (exact threshold, exempt categories, flagged vendors) are handled
  - finalize_expense writes the correct status and an audit trail
  - The fail-safe "reject on ambiguous approval state" rule holds

What this layer dangerously misses (and why the other layers exist):
  - Whether the conditional edge routes to approval_gate vs. finalize correctly
  - Whether state written by classify_expense is actually visible to finalize_expense
    when they run inside the graph (state propagation between nodes)
  - Whether the interrupt fires at the right moment and whether the graph resumes
    on the correct branch — that is purely a graph-structure concern invisible here
  - Whether the thread_id config wires up to the right checkpointer namespace

HOW TO RUN:
    uv run pytest tests/test_unit.py -v
"""

import pytest

# ── The key import: node functions directly, zero graph machinery ─────────────
# This is the architectural payoff of keeping nodes.py free of graph imports.
# We never touch StateGraph, compile(), or invoke() in this file.
from src.nodes import (
    HIGH_AMOUNT_THRESHOLD,
    FLAGGED_VENDORS,
    classify_expense,
    finalize_expense,
)
from src.state import ExpenseState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(**overrides) -> ExpenseState:
    """Return a minimal valid ExpenseState with sensible defaults.

    Overrides allow individual tests to vary only the fields they care about,
    keeping test bodies focused on a single assertion.
    """
    base: ExpenseState = {
        "expense_id": "EXP-0001",
        "vendor": "Airport Diner",
        "category": "meals",
        "amount": 45.00,
        "submitted_by": "jane.doe@company.com",
        "risk_flag": None,
        "requires_approval": False,
        "approval_decision": None,
        "reviewer_notes": None,
        "final_status": None,
        "audit_trail": None,
    }
    base.update(overrides)
    return base


# ── classify_expense: clean expenses ─────────────────────────────────────────

class TestClassifyExpenseClean:
    """Expenses that should pass without triggering any flag."""

    def test_small_meals_expense_clears_automatically(self):
        state = _make_state(vendor="Airport Diner", category="meals", amount=42.50)
        result = classify_expense(state)
        assert result["risk_flag"] is None
        assert result["requires_approval"] is False

    def test_amount_just_below_threshold_clears(self):
        """Edge case: $499.99 should not trigger the HIGH_AMOUNT flag."""
        state = _make_state(amount=HIGH_AMOUNT_THRESHOLD - 0.01)
        result = classify_expense(state)
        assert result["risk_flag"] is None
        assert result["requires_approval"] is False

    def test_equipment_above_threshold_is_exempt(self):
        """Equipment purchases are pre-budgeted; threshold does not apply."""
        state = _make_state(
            vendor="Dell Technologies",
            category="equipment",
            amount=1_200.00,
        )
        result = classify_expense(state)
        assert result["risk_flag"] is None
        assert result["requires_approval"] is False

    def test_software_above_threshold_is_exempt(self):
        state = _make_state(
            vendor="JetBrains",
            category="software",
            amount=750.00,
        )
        result = classify_expense(state)
        assert result["risk_flag"] is None
        assert result["requires_approval"] is False

    def test_unknown_category_small_amount_clears(self):
        state = _make_state(category="parking", amount=18.00)
        result = classify_expense(state)
        assert result["risk_flag"] is None


# ── classify_expense: HIGH_AMOUNT flag ────────────────────────────────────────

class TestClassifyExpenseHighAmount:
    """Expenses that should trigger HIGH_AMOUNT risk flag."""

    def test_amount_exactly_at_threshold_is_flagged(self):
        """Boundary condition: $500.00 exactly must trigger the flag."""
        state = _make_state(
            vendor="Marriott",
            category="lodging",
            amount=HIGH_AMOUNT_THRESHOLD,
        )
        result = classify_expense(state)
        assert result["risk_flag"] == "HIGH_AMOUNT"
        assert result["requires_approval"] is True

    def test_large_hotel_charge_flagged(self):
        """The production bug: $3,500 hotel auto-approving is what we're preventing."""
        state = _make_state(
            vendor="Grand Hyatt",
            category="lodging",
            amount=3_500.00,
        )
        result = classify_expense(state)
        assert result["risk_flag"] == "HIGH_AMOUNT"
        assert result["requires_approval"] is True

    def test_business_class_flight_flagged(self):
        state = _make_state(
            vendor="United Airlines",
            category="travel",
            amount=2_200.00,
        )
        result = classify_expense(state)
        assert result["risk_flag"] == "HIGH_AMOUNT"
        assert result["requires_approval"] is True

    def test_expensive_team_dinner_flagged(self):
        state = _make_state(
            vendor="Le Bernardin",
            category="meals",
            amount=850.00,
        )
        result = classify_expense(state)
        assert result["risk_flag"] == "HIGH_AMOUNT"
        assert result["requires_approval"] is True


# ── classify_expense: FLAGGED_VENDOR ─────────────────────────────────────────

class TestClassifyExpenseFlaggedVendor:
    """Expenses from vendors on the policy watchlist, regardless of amount."""

    def test_flagged_vendor_triggers_approval_even_at_small_amount(self):
        """A $50 expense from a flagged vendor still needs human eyes."""
        flagged = next(iter(FLAGGED_VENDORS))  # grab any vendor from the set
        state = _make_state(vendor=flagged, amount=50.00)
        result = classify_expense(state)
        assert result["risk_flag"] == "FLAGGED_VENDOR"
        assert result["requires_approval"] is True

    def test_flagged_vendor_detection_is_case_insensitive(self):
        """Vendor names on receipts often arrive in mixed case."""
        state = _make_state(vendor="Casino Royale", amount=200.00)
        result = classify_expense(state)
        assert result["risk_flag"] == "FLAGGED_VENDOR"

    def test_flagged_vendor_detection_strips_whitespace(self):
        state = _make_state(vendor="  casino royale  ", amount=200.00)
        result = classify_expense(state)
        assert result["risk_flag"] == "FLAGGED_VENDOR"

    def test_vendor_name_partial_match_does_not_flag(self):
        """'casino royale' is flagged; 'Casino Restaurant' is not."""
        state = _make_state(vendor="Casino Restaurant & Bar", amount=80.00)
        result = classify_expense(state)
        # This vendor is NOT in the set, so it should clear
        assert result["risk_flag"] is None


# ── finalize_expense: auto-cleared path ──────────────────────────────────────

class TestFinalizeExpenseCleared:
    """Expenses that never needed a human approver."""

    def test_auto_cleared_expense_gets_cleared_status(self):
        state = _make_state(
            expense_id="EXP-0099",
            requires_approval=False,
            approval_decision=None,
        )
        result = finalize_expense(state)
        assert result["final_status"] == "cleared"

    def test_auto_cleared_audit_trail_mentions_no_flags(self):
        state = _make_state(
            expense_id="EXP-0099",
            requires_approval=False,
        )
        result = finalize_expense(state)
        assert "auto-cleared" in result["audit_trail"]
        assert "no policy flags" in result["audit_trail"].lower()

    def test_audit_trail_contains_expense_id(self):
        state = _make_state(expense_id="EXP-7777", requires_approval=False)
        result = finalize_expense(state)
        assert "EXP-7777" in result["audit_trail"]


# ── finalize_expense: human-reviewed paths ────────────────────────────────────

class TestFinalizeExpenseReviewed:
    """Expenses that went through the HITL gate."""

    def test_approved_decision_yields_approved_status(self):
        state = _make_state(
            requires_approval=True,
            risk_flag="HIGH_AMOUNT",
            approval_decision="approved",
            reviewer_notes="Confirmed conference registration, valid.",
        )
        result = finalize_expense(state)
        assert result["final_status"] == "approved"
        assert "approved" in result["audit_trail"].lower()

    def test_rejected_decision_yields_rejected_status(self):
        state = _make_state(
            requires_approval=True,
            risk_flag="FLAGGED_VENDOR",
            approval_decision="rejected",
            reviewer_notes="Vendor violates travel policy.",
        )
        result = finalize_expense(state)
        assert result["final_status"] == "rejected"
        assert "rejected" in result["audit_trail"].lower()

    def test_reviewer_notes_appear_in_audit_trail(self):
        notes = "Amount exceeds per-diem; partial approval not supported."
        state = _make_state(
            requires_approval=True,
            risk_flag="HIGH_AMOUNT",
            approval_decision="rejected",
            reviewer_notes=notes,
        )
        result = finalize_expense(state)
        assert notes in result["audit_trail"]

    def test_fail_safe_missing_approval_decision_defaults_to_rejected(self):
        """
        If requires_approval=True but approval_decision is somehow None
        (e.g., the resume payload was malformed), we must NOT auto-approve.
        The fail-safe is: default to 'rejected'.
        This is the production safety net that tracing alone would never surface.
        """
        state = _make_state(
            requires_approval=True,
            risk_flag="HIGH_AMOUNT",
            approval_decision=None,  # ← missing decision
            reviewer_notes=None,
        )
        result = finalize_expense(state)
        assert result["final_status"] == "rejected", (
            "An expense requiring approval must never be auto-approved "
            "when the approval decision is missing."
        )
