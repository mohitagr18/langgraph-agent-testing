"""
Layer 3 — HITL Tests: interrupt/resume cycle end-to-end.

What this layer proves (and what nothing else in this suite can prove):
  - The graph actually SUSPENDS at the approval_gate node — invoke() returns
    without reaching finalize when a flagged expense is submitted
  - State is checkpointed during the pause — the expense data survives the gap
    between Step 1 and Step 2 (this is the persistence guarantee under load)
  - The graph resumes on the CORRECT BRANCH after Command(resume=...) — an
    "approved" decision flows to finalize with approval_decision="approved",
    not to some default path
  - The final state is COMPLETE — both the classification fields set in Step 1
    and the approval fields set in Step 2 are present in the final snapshot

The two-step pattern is non-negotiable:
  Step 1 → graph.invoke(initial_state, config)  → interrupt fires, graph pauses
  Step 2 → graph.invoke(Command(resume=...), config)  → graph resumes, completes

A test that calls invoke() only once CANNOT test HITL behavior.  The entire
point of this layer is the GAP between Step 1 and Step 2, which simulates the
real-world delay between an agent pausing and a human opening their inbox.

HOW TO RUN:
    uv run pytest tests/test_hitl.py -v
"""

import uuid

import pytest
from langgraph.types import Command

from src.graph import build_graph


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def graph():
    """Fresh graph instance for each test — MemorySaver guarantees no cross-test
    state bleed even without unique thread IDs, but we use UUIDs anyway."""
    return build_graph()


def _unique_thread() -> dict:
    return {"configurable": {"thread_id": f"hitl-{uuid.uuid4().hex}"}}


def _flagged_expense(**overrides) -> dict:
    """An expense that will always trigger the HITL gate (high amount)."""
    base = {
        "expense_id": f"EXP-{uuid.uuid4().hex[:6].upper()}",
        "vendor": "Grand Conference Hotel",
        "category": "lodging",
        "amount": 2_200.00,
        "submitted_by": "bob.smith@company.com",
        "risk_flag": None,
        "requires_approval": False,
        "approval_decision": None,
        "reviewer_notes": None,
        "final_status": None,
        "audit_trail": None,
    }
    base.update(overrides)
    return base


# ── Core two-step pattern ─────────────────────────────────────────────────────

class TestInterruptFires:
    """Step 1 assertions: the graph must genuinely pause."""

    def test_interrupt_is_present_in_step1_result(self, graph):
        """The __interrupt__ key in the result dict is the signal that the graph
        has suspended execution.  If this fails, the interrupt() call in
        request_human_approval never ran, which means a $2,200 lodging charge
        would be auto-approved with zero human oversight."""
        config = _unique_thread()
        result = graph.invoke(_flagged_expense(), config)

        assert "__interrupt__" in result, (
            "Expected the graph to pause at the approval_gate node for a "
            "high-amount expense, but invoke() returned without an interrupt.  "
            "This is the production bug: the agent is auto-approving flagged "
            "expenses without human review."
        )

    def test_interrupt_payload_contains_expense_context(self, graph):
        """The interrupt value must carry enough context for the reviewer to
        make an informed decision — expense ID, vendor, amount, and the risk flag."""
        config = _unique_thread()
        expense_id = "EXP-CONTEXT-TEST"
        result = graph.invoke(
            _flagged_expense(expense_id=expense_id, vendor="Ultra Luxury Resort"),
            config,
        )
        interrupt_list = result.get("__interrupt__", [])
        assert len(interrupt_list) > 0
        payload = interrupt_list[0].value

        assert "expense_id" in payload or expense_id in str(payload), (
            "Interrupt payload must include the expense ID so the reviewer knows "
            "which expense they are looking at."
        )
        assert "2200" in str(payload) or "2,200" in str(payload) or payload.get("amount") == 2200.00

    def test_graph_is_paused_on_approval_gate_after_step1(self, graph):
        """After Step 1, get_state().next must point to approval_gate — proving
        the graph knows it is mid-flight and waiting for human input."""
        config = _unique_thread()
        graph.invoke(_flagged_expense(), config)
        snapshot = graph.get_state(config)

        assert len(snapshot.next) > 0, "Graph should be paused, not completed."
        assert "approval_gate" in snapshot.next, (
            f"Expected 'approval_gate' in next nodes, got {snapshot.next}"
        )


class TestStatePersistsAcrossResume:
    """State written in Step 1 must survive into Step 2.

    This is the persistence guarantee.  Between Step 1 and Step 2 there could
    be seconds, minutes, or hours of real time.  If the checkpointer fails to
    preserve the classification fields, Step 2 would resume into a partially
    blank state — and finalize_expense would compute the wrong audit trail.
    """

    def test_expense_fields_survive_the_pause(self, graph):
        config = _unique_thread()
        expense_id = "EXP-PERSIST-001"
        original_amount = 1_850.00

        # Step 1: invoke to interrupt
        graph.invoke(
            _flagged_expense(expense_id=expense_id, amount=original_amount),
            config,
        )

        # Inspect state WHILE PAUSED — before any resume
        snapshot = graph.get_state(config)
        assert snapshot.values["expense_id"] == expense_id, (
            "expense_id written at invocation must be present in the "
            "checkpoint immediately after the interrupt."
        )
        assert snapshot.values["amount"] == original_amount
        assert snapshot.values["risk_flag"] == "HIGH_AMOUNT", (
            "classify_expense must have run and written risk_flag before "
            "approval_gate interrupted — classify runs first."
        )
        assert snapshot.values["requires_approval"] is True

    def test_classification_fields_visible_after_resume(self, graph):
        """After Step 2, the final state must contain fields from BOTH steps."""
        config = _unique_thread()
        expense_id = "EXP-TWOPART"

        # Step 1
        graph.invoke(_flagged_expense(expense_id=expense_id, amount=900.00), config)

        # Step 2: resume with approval
        final = graph.invoke(
            Command(resume={"decision": "approved", "notes": "CFO pre-approved."}),
            config,
        )

        # Fields from Step 1 (classification)
        assert final["expense_id"] == expense_id
        assert final["risk_flag"] == "HIGH_AMOUNT"
        assert final["requires_approval"] is True

        # Fields from Step 2 (HITL)
        assert final["approval_decision"] == "approved"
        assert "CFO pre-approved" in final["reviewer_notes"]

        # Fields from finalize (Step 3)
        assert final["final_status"] == "approved"
        assert final["audit_trail"] is not None


# ── Approved path ─────────────────────────────────────────────────────────────

class TestHITLApprovedPath:
    """When the reviewer approves, the graph must reach final_status='approved'."""

    def test_approved_decision_yields_approved_final_status(self, graph):
        config = _unique_thread()

        # Step 1: invoke to interrupt
        graph.invoke(_flagged_expense(amount=750.00), config)

        # Step 2: resume with approval
        final = graph.invoke(
            Command(resume={"decision": "approved", "notes": "Conference fees confirmed."}),
            config,
        )

        assert final["final_status"] == "approved"

    def test_approved_audit_trail_mentions_approver_notes(self, graph):
        config = _unique_thread()
        graph.invoke(_flagged_expense(), config)
        notes = "Verified against Q3 travel budget allocation."
        final = graph.invoke(
            Command(resume={"decision": "approved", "notes": notes}),
            config,
        )
        assert notes in final["audit_trail"]

    def test_approved_expense_has_complete_state(self, graph):
        """Every field in ExpenseState must be populated after a full HITL run."""
        config = _unique_thread()
        graph.invoke(_flagged_expense(expense_id="EXP-FULL"), config)
        final = graph.invoke(
            Command(resume={"decision": "approved", "notes": "OK"}),
            config,
        )

        # All state fields must be non-None after a complete run
        assert final["expense_id"] == "EXP-FULL"
        assert final["vendor"] is not None
        assert final["amount"] is not None
        assert final["risk_flag"] == "HIGH_AMOUNT"
        assert final["requires_approval"] is True
        assert final["approval_decision"] == "approved"
        assert final["final_status"] == "approved"
        assert final["audit_trail"] is not None


# ── Rejected path ─────────────────────────────────────────────────────────────

class TestHITLRejectedPath:
    """When the reviewer rejects, the graph must reach final_status='rejected'."""

    def test_rejected_decision_yields_rejected_final_status(self, graph):
        config = _unique_thread()
        graph.invoke(_flagged_expense(amount=3_200.00), config)
        final = graph.invoke(
            Command(resume={"decision": "rejected", "notes": "Exceeds per-diem policy."}),
            config,
        )
        assert final["final_status"] == "rejected"

    def test_rejected_audit_trail_contains_rejection_language(self, graph):
        config = _unique_thread()
        graph.invoke(_flagged_expense(vendor="Casino Royale", amount=120.00), config)
        final = graph.invoke(
            Command(resume={"decision": "rejected", "notes": "Vendor violates policy."}),
            config,
        )
        assert "rejected" in final["audit_trail"].lower()

    def test_rejected_expense_does_not_show_approved(self, graph):
        """Double-negative guard: a rejected expense must never read 'approved'
        in final_status.  This is the bug that would cost the company money."""
        config = _unique_thread()
        graph.invoke(_flagged_expense(), config)
        final = graph.invoke(
            Command(resume={"decision": "rejected", "notes": ""}),
            config,
        )
        assert final["final_status"] != "approved"
        assert final["final_status"] == "rejected"


# ── Flagged vendor HITL ───────────────────────────────────────────────────────

class TestFlaggedVendorHITL:
    """FLAGGED_VENDOR risk flag also triggers the HITL gate."""

    def test_flagged_vendor_small_amount_still_interrupts(self, graph):
        """Even a $60 charge from a flagged vendor must stop for human review."""
        config = _unique_thread()
        result = graph.invoke(
            _flagged_expense(
                vendor="casino royale",
                category="meals",
                amount=60.00,
            ),
            config,
        )
        assert "__interrupt__" in result, (
            "A flagged vendor must trigger interrupt() regardless of amount.  "
            "If this fails, a small-dollar charge from a policy-violating vendor "
            "slips through with no review."
        )

    def test_flagged_vendor_approved_clears_correctly(self, graph):
        config = _unique_thread()
        graph.invoke(
            _flagged_expense(vendor="Casino Royale", category="meals", amount=60.00),
            config,
        )
        final = graph.invoke(
            Command(resume={"decision": "approved", "notes": "Exception granted by VP."}),
            config,
        )
        assert final["final_status"] == "approved"
        assert final["risk_flag"] == "FLAGGED_VENDOR"
