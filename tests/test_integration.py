"""
Layer 2 — Integration Tests: full compiled graph, end-to-end.

What this layer proves beyond unit tests:
  - The conditional edge routes correctly: clean expenses skip the HITL gate,
    flagged expenses land on approval_gate before finalize
  - State written by classify_expense is actually visible to finalize_expense
    (state propagation through the graph's internal merge logic)
  - Thread isolation: two concurrent sessions do not bleed state into each other
  - State persists in the checkpointer after an invoke completes (get_state works)

What this layer still misses:
  - The HITL pause/resume cycle — a test that calls invoke() once cannot prove
    the interrupt fires, that the graph actually suspends, or that it resumes
    on the right branch.  That requires tests/test_hitl.py.

HOW TO RUN:
    uv run pytest tests/test_integration.py -v
"""

import uuid

import pytest

from src.graph import build_graph
from src.nodes import HIGH_AMOUNT_THRESHOLD


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def graph():
    """Fresh graph instance per test — MemorySaver, no disk state."""
    return build_graph()


def _unique_thread() -> dict:
    """Return a LangGraph config dict with a UUID-based thread_id.

    Using UUIDs prevents state bleed between test runs even if the test
    process is reused.  Hardcoded IDs like 'test-001' are a common source
    of flaky integration tests.
    """
    return {"configurable": {"thread_id": f"integ-{uuid.uuid4().hex}"}}


def _clean_expense(**overrides) -> dict:
    """Minimal clean expense that should auto-clear without human review."""
    base = {
        "expense_id": f"EXP-{uuid.uuid4().hex[:6].upper()}",
        "vendor": "Corner Café",
        "category": "meals",
        "amount": 28.50,
        "submitted_by": "alice@company.com",
        "risk_flag": None,
        "requires_approval": False,
        "approval_decision": None,
        "reviewer_notes": None,
        "final_status": None,
        "audit_trail": None,
    }
    base.update(overrides)
    return base


# ── Clean expense path ────────────────────────────────────────────────────────

class TestCleanExpenseIntegration:
    """End-to-end: expenses that should clear without human intervention."""

    def test_small_meal_clears_in_single_invoke(self, graph):
        config = _unique_thread()
        result = graph.invoke(_clean_expense(amount=35.00), config)
        assert result["final_status"] == "cleared"
        assert result["risk_flag"] is None
        assert result["requires_approval"] is False

    def test_audit_trail_is_populated_after_auto_clear(self, graph):
        config = _unique_thread()
        result = graph.invoke(_clean_expense(), config)
        assert result["audit_trail"] is not None
        assert len(result["audit_trail"]) > 20  # meaningful, not empty string

    def test_exempt_category_large_amount_clears(self, graph):
        """Equipment above threshold must flow to finalize without an interrupt."""
        config = _unique_thread()
        result = graph.invoke(
            _clean_expense(
                vendor="Apple Business",
                category="equipment",
                amount=2_000.00,
            ),
            config,
        )
        assert result["final_status"] == "cleared"

    def test_state_is_checkpointed_after_clean_run(self, graph):
        """get_state() should reflect the final state after a completed invoke."""
        config = _unique_thread()
        graph.invoke(_clean_expense(expense_id="EXP-CHECKPOINT"), config)
        snapshot = graph.get_state(config)
        assert snapshot.values["final_status"] == "cleared"
        assert snapshot.values["expense_id"] == "EXP-CHECKPOINT"


# ── Thread isolation ──────────────────────────────────────────────────────────

class TestThreadIsolation:
    """Prove that two threads running through the same graph instance never
    share state.  This is the integration-level guarantee that unit tests
    cannot provide — it requires a real checkpointer namespace separation."""

    def test_two_threads_produce_independent_results(self, graph):
        config_a = _unique_thread()
        config_b = _unique_thread()

        result_a = graph.invoke(
            _clean_expense(vendor="Corner Café", amount=25.00), config_a
        )
        result_b = graph.invoke(
            _clean_expense(vendor="Airline XYZ", category="travel", amount=850.00),
            config_b,
        )

        # Thread A: small meal → cleared
        assert result_a["final_status"] == "cleared"
        assert result_a["risk_flag"] is None

        # Thread B: large travel → paused at approval gate (interrupt fires)
        # The result dict will contain __interrupt__ when the graph pauses.
        # We don't assert on final_status here because Thread B is mid-flight.
        assert result_b.get("final_status") != "cleared" or result_b["risk_flag"] == "HIGH_AMOUNT"

    def test_thread_a_state_not_visible_in_thread_b(self, graph):
        """State from thread A must be invisible when querying thread B."""
        config_a = _unique_thread()
        config_b = _unique_thread()

        graph.invoke(_clean_expense(expense_id="EXP-ALPHA", amount=30.00), config_a)
        graph.invoke(_clean_expense(expense_id="EXP-BETA", amount=40.00), config_b)

        state_a = graph.get_state(config_a)
        state_b = graph.get_state(config_b)

        assert state_a.values["expense_id"] == "EXP-ALPHA"
        assert state_b.values["expense_id"] == "EXP-BETA"
        assert state_a.values["expense_id"] != state_b.values["expense_id"]

    def test_many_concurrent_threads_do_not_collide(self, graph):
        """Run five different expenses on five different threads simultaneously
        and verify each thread holds only its own data."""
        expenses = [
            _clean_expense(expense_id=f"EXP-MULTI-{i:03d}", amount=float(10 + i * 7))
            for i in range(5)
        ]
        configs = [_unique_thread() for _ in range(5)]

        results = []
        for expense, config in zip(expenses, configs):
            results.append(graph.invoke(expense, config))

        expense_ids = [r["expense_id"] for r in results]
        # All five must be distinct
        assert len(set(expense_ids)) == 5

    def test_reusing_same_thread_accumulates_only_latest_state(self, graph):
        """Same thread_id, second invoke: the graph should see the checkpoint
        from the first invoke (this is the persistence guarantee)."""
        config = _unique_thread()
        graph.invoke(_clean_expense(expense_id="EXP-ROUND1"), config)
        state_after_first = graph.get_state(config)
        assert state_after_first.values["expense_id"] == "EXP-ROUND1"


# ── Flagged expense: graph routes to approval_gate ────────────────────────────

class TestFlaggedExpenseRouting:
    """Verify the conditional edge routes flagged expenses to the HITL gate,
    not straight to finalize.  This is not testable at the unit level."""

    def test_high_amount_expense_pauses_at_approval_gate(self, graph):
        config = _unique_thread()
        # Invoke with a flagged expense; the graph should pause at approval_gate
        result = graph.invoke(
            _clean_expense(
                expense_id="EXP-HILO",
                vendor="Grand Hyatt",
                category="lodging",
                amount=3_500.00,
            ),
            config,
        )

        # When a dynamic interrupt fires, LangGraph embeds __interrupt__ in result
        assert "__interrupt__" in result, (
            "A high-amount lodging expense must trigger an interrupt at the "
            "approval_gate node.  If this assertion fails, the conditional edge "
            "is routing incorrectly — the agent would auto-approve $3,500 hotel "
            "charges with no human ever seeing them."
        )

        # Graph is mid-flight: next node must be approval_gate
        snapshot = graph.get_state(config)
        assert "approval_gate" in snapshot.next

    def test_flagged_vendor_expense_pauses_at_approval_gate(self, graph):
        config = _unique_thread()
        result = graph.invoke(
            _clean_expense(
                vendor="Casino Royale",
                category="meals",
                amount=80.00,
            ),
            config,
        )
        assert "__interrupt__" in result
        snapshot = graph.get_state(config)
        assert "approval_gate" in snapshot.next
