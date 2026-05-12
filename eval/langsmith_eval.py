"""
LangSmith Evaluation for the Expense Report Processing Agent.

HOW TO RUN:
    uv run python eval/langsmith_eval.py

PREREQUISITES:
    - LANGSMITH_API_KEY set in .env
    - LANGSMITH_TRACING=true in .env
    - OPENAI_API_KEY set in .env (not used in evaluation itself, but graph
      init loads dotenv and will warn if missing)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY EVALUATION IS A DIFFERENT CLASS OF EVIDENCE THAN TRACING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tracing records what happened.  Evaluation proves whether what happened was
correct — and that distinction matters enormously in production.

WHAT TRACING GIVES YOU:
  - A timeline of every node execution with its input and output state
  - Latency measurements at node granularity
  - A paper trail for debugging ("why did EXP-0042 get rejected?")
  - Token counts and API costs per run
  - Error stack traces when nodes throw exceptions

WHAT TRACING DOES NOT GIVE YOU:
  - Whether the classification was *correct* (tracing records "category=billing"
    but cannot tell you whether that was the right answer)
  - Regression protection (you can observe 100 correct traces in staging and
    still ship a rule-change that breaks 30% of production cases)
  - A metric you can put in a deployment gate (CI/CD can't block a bad deploy
    based on traces; it can block based on evaluation score)
  - Cross-run aggregation (tracing shows individual runs; evaluation shows
    "accuracy dropped from 97% to 84% in this PR")

THE PROOF DIFFERENCE:
  Tracing says: "I can see the agent ran correctly on these five examples."
  Evaluation says: "The agent is correct on 94.3% of a named, versioned
  dataset, and I can prove that number is stable across code changes."

Evaluation transforms observability from post-hoc storytelling into a
reproducible, falsifiable claim about system behavior.  That is the difference
between "it looked fine in our testing" and "we have evidence it works."

IN THE EXPENSE DOMAIN SPECIFICALLY:
  Without evaluation, a rule change to HIGH_AMOUNT_THRESHOLD (say, from $500
  to $750) might look fine in traces — the agent runs, states are logged, no
  exceptions.  But 23% of previously-flagged expenses now slip through without
  human review, and no trace tells you that the slip is happening.
  An evaluation dataset with ground-truth labels catches this in CI before
  the change ships.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Bridge LANGSMITH_PROJECT → LANGCHAIN_PROJECT ──────────────────────────────
# The LangSmith SDK routes traces and evaluation experiments to the project
# named in LANGCHAIN_PROJECT.  LANGSMITH_PROJECT is the user-facing env var
# name we expose in .env.example; we sync it here so both names work.
_project = os.getenv("LANGSMITH_PROJECT")
if _project and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = _project

# ── Validate environment before importing LangSmith SDK ───────────────────────
_api_key = os.getenv("LANGSMITH_API_KEY")
if not _api_key:
    print(
        "[eval] ERROR: LANGSMITH_API_KEY is not set.\n"
        "       Copy .env.example to .env and fill in your LangSmith API key.\n"
        "       Get one at https://smith.langchain.com → Settings → API Keys"
    )
    sys.exit(1)

from langsmith import Client
from langsmith.evaluation import evaluate

from src.graph import build_graph

# ── Dataset definition ────────────────────────────────────────────────────────

DATASET_NAME = "expense-classification-eval-v1"

# Ground-truth examples: (input_state_fields, expected_output_fields)
# These are the cases that matter most from a compliance perspective.
DATASET_EXAMPLES = [
    # ── Auto-clear cases ──────────────────────────────────────────────────
    {
        "inputs": {
            "expense_id": "EVAL-001",
            "vendor": "Corner Café",
            "category": "meals",
            "amount": 35.00,
            "submitted_by": "alice@co.com",
        },
        "outputs": {
            "risk_flag": None,
            "requires_approval": False,
            "final_status": "cleared",
        },
    },
    {
        "inputs": {
            "expense_id": "EVAL-002",
            "vendor": "Dell Technologies",
            "category": "equipment",
            "amount": 1_800.00,
            "submitted_by": "bob@co.com",
        },
        "outputs": {
            "risk_flag": None,
            "requires_approval": False,
            "final_status": "cleared",
        },
    },
    {
        "inputs": {
            "expense_id": "EVAL-003",
            "vendor": "City Parking",
            "category": "parking",
            "amount": 22.00,
            "submitted_by": "carol@co.com",
        },
        "outputs": {
            "risk_flag": None,
            "requires_approval": False,
            "final_status": "cleared",
        },
    },
    # ── Flagged for HIGH_AMOUNT ───────────────────────────────────────────
    {
        "inputs": {
            "expense_id": "EVAL-004",
            "vendor": "Grand Hyatt",
            "category": "lodging",
            "amount": 3_500.00,
            "submitted_by": "david@co.com",
        },
        "outputs": {
            "risk_flag": "HIGH_AMOUNT",
            "requires_approval": True,
            # final_status is not asserted here: the graph pauses for human
            # review and the evaluation runner does not simulate the resume step.
        },
    },
    {
        "inputs": {
            "expense_id": "EVAL-005",
            "vendor": "United Airlines",
            "category": "travel",
            "amount": 2_100.00,
            "submitted_by": "eve@co.com",
        },
        "outputs": {
            "risk_flag": "HIGH_AMOUNT",
            "requires_approval": True,
        },
    },
    {
        "inputs": {
            "expense_id": "EVAL-006",
            "vendor": "Le Bernardin",
            "category": "meals",
            "amount": 750.00,
            "submitted_by": "frank@co.com",
        },
        "outputs": {
            "risk_flag": "HIGH_AMOUNT",
            "requires_approval": True,
        },
    },
    # ── Flagged for FLAGGED_VENDOR ────────────────────────────────────────
    {
        "inputs": {
            "expense_id": "EVAL-007",
            "vendor": "Casino Royale",
            "category": "meals",
            "amount": 60.00,
            "submitted_by": "grace@co.com",
        },
        "outputs": {
            "risk_flag": "FLAGGED_VENDOR",
            "requires_approval": True,
        },
    },
    {
        "inputs": {
            "expense_id": "EVAL-008",
            "vendor": "Ultra Luxury Resort",
            "category": "lodging",
            "amount": 500.00,
            "submitted_by": "hank@co.com",
        },
        "outputs": {
            "risk_flag": "FLAGGED_VENDOR",
            "requires_approval": True,
        },
    },
    # ── Boundary: amount exactly at threshold ─────────────────────────────
    {
        "inputs": {
            "expense_id": "EVAL-009",
            "vendor": "Business Hotel",
            "category": "lodging",
            "amount": 500.00,
            "submitted_by": "iris@co.com",
        },
        "outputs": {
            "risk_flag": "HIGH_AMOUNT",
            "requires_approval": True,
        },
    },
    # ── Boundary: amount just below threshold ─────────────────────────────
    {
        "inputs": {
            "expense_id": "EVAL-010",
            "vendor": "Business Hotel",
            "category": "lodging",
            "amount": 499.99,
            "submitted_by": "jack@co.com",
        },
        "outputs": {
            "risk_flag": None,
            "requires_approval": False,
            "final_status": "cleared",
        },
    },
]


# ── Evaluator functions ───────────────────────────────────────────────────────

def evaluate_risk_flag(run, example) -> dict:
    """Check whether the agent assigned the correct risk_flag.

    This is the most critical evaluator for compliance: a wrong risk_flag means
    the agent is either over-flagging (operational friction) or under-flagging
    (compliance failure).  Under-flagging is the catastrophic case.

    Returns:
        {"score": 1} if risk_flag matches expected, {"score": 0} otherwise.
        LangSmith aggregates these scores into an accuracy percentage.
    """
    predicted_flag = run.outputs.get("risk_flag")
    expected_flag = example.outputs.get("risk_flag")

    # Ground-truth examples without a risk_flag key are not asserting on this
    # dimension — skip rather than penalize.
    if "risk_flag" not in example.outputs:
        return {"score": None, "comment": "risk_flag not in ground truth; skipped"}

    matches = predicted_flag == expected_flag
    return {
        "score": 1 if matches else 0,
        "comment": f"predicted={predicted_flag!r}, expected={expected_flag!r}",
    }


def evaluate_requires_approval(run, example) -> dict:
    """Check whether the agent correctly identified that human review is needed.

    This is the safety-critical evaluator: a false negative here means an
    expense that should have been flagged for human review was auto-processed.
    That is exactly the production bug the article is about.
    """
    if "requires_approval" not in example.outputs:
        return {"score": None, "comment": "requires_approval not in ground truth; skipped"}

    predicted = run.outputs.get("requires_approval")
    expected = example.outputs.get("requires_approval")
    matches = predicted == expected

    comment = f"predicted={predicted}, expected={expected}"
    if not matches and expected is True:
        comment += " ← FALSE NEGATIVE: expense should have been flagged for human review"

    return {"score": 1 if matches else 0, "comment": comment}


def evaluate_final_status(run, example) -> dict:
    """Check whether the graph reached the correct final_status.

    Only applies to auto-clear cases (the evaluation runner does not simulate
    the HITL resume step for flagged expenses).
    """
    if "final_status" not in example.outputs:
        return {"score": None, "comment": "final_status not in ground truth; skipped"}

    predicted = run.outputs.get("final_status")
    expected = example.outputs.get("final_status")
    matches = predicted == expected
    return {
        "score": 1 if matches else 0,
        "comment": f"predicted={predicted!r}, expected={expected!r}",
    }


# ── Graph runner ──────────────────────────────────────────────────────────────

def _run_expense_agent(inputs: dict) -> dict:
    """Adapter that wraps the expense graph for LangSmith evaluate().

    For flagged expenses the graph will pause at approval_gate.  The evaluation
    runner captures the state at that point — we do NOT simulate a resume here
    because the ground-truth labels are about classification correctness, not
    the HITL decision.  HITL behavior is tested in tests/test_hitl.py.
    """
    import uuid

    g = build_graph()
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}}

    full_state = {
        "risk_flag": None,
        "requires_approval": False,
        "approval_decision": None,
        "reviewer_notes": None,
        "final_status": None,
        "audit_trail": None,
        **inputs,
    }

    result = g.invoke(full_state, config)

    # If the graph paused (interrupt), get the checkpointed state which has
    # classification fields even though finalize hasn't run yet.
    if "__interrupt__" in result:
        snapshot = g.get_state(config)
        return snapshot.values

    return result


# ── Dataset bootstrap ─────────────────────────────────────────────────────────

def _ensure_dataset(client: Client) -> str:
    """Create the evaluation dataset in LangSmith if it doesn't exist yet.

    Returns the dataset name (unchanged) for use in evaluate().
    """
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"[eval] Using existing dataset: {DATASET_NAME!r}")
        return DATASET_NAME

    print(f"[eval] Creating dataset: {DATASET_NAME!r}")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Ground-truth examples for the expense report classification agent. "
            "Tests risk_flag assignment, requires_approval routing, and "
            "final_status for auto-cleared expenses."
        ),
    )
    client.create_examples(
        inputs=[ex["inputs"] for ex in DATASET_EXAMPLES],
        outputs=[ex["outputs"] for ex in DATASET_EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"[eval] Created {len(DATASET_EXAMPLES)} examples in {DATASET_NAME!r}")
    return DATASET_NAME


# ── Main ──────────────────────────────────────────────────────────────────────

def _save_dataset_locally() -> None:
    """Write DATASET_EXAMPLES to eval/data/expense_eval_dataset.json.

    This gives every reader a local, inspectable copy of the ground-truth data.
    The file is committed to the repo so you can diff it when examples change.
    LangSmith holds the authoritative copy for evaluation runs; this JSON is
    the human-readable source of truth that lives alongside the code.
    """
    import json
    import pathlib

    data_dir = pathlib.Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    out = data_dir / "expense_eval_dataset.json"
    out.write_text(
        json.dumps(
            {
                "dataset_name": DATASET_NAME,
                "description": (
                    "Ground-truth examples for the expense report classification agent. "
                    "Tests risk_flag assignment, requires_approval routing, and "
                    "final_status for auto-cleared expenses."
                ),
                "examples": DATASET_EXAMPLES,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[eval] Dataset saved locally → {out.relative_to(pathlib.Path.cwd())}")


def main() -> None:
    # Write the dataset to disk first so readers can inspect it locally
    _save_dataset_locally()

    active_project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT", "default")
    print(f"[eval] LangSmith project: {active_project!r}")

    client = Client()
    dataset_name = _ensure_dataset(client)

    print(f"\n[eval] Running evaluation against dataset: {dataset_name!r}")
    print("[eval] Evaluators: risk_flag accuracy, requires_approval accuracy, final_status accuracy\n")

    results = evaluate(
        _run_expense_agent,
        data=dataset_name,
        evaluators=[
            evaluate_risk_flag,
            evaluate_requires_approval,
            evaluate_final_status,
        ],
        experiment_prefix="expense-agent-classification",
        metadata={
            "description": "Classification accuracy eval for expense agent",
            "project": active_project,
            "version": "v1",
        },
    )

    # Summarise aggregate scores — avoid to_pandas() which requires the optional
    # pandas dependency.  aggregate_metrics is always populated by evaluate().
    print("\n[eval] Results:")
    if hasattr(results, "aggregate_metrics") and results.aggregate_metrics:
        for key, val in results.aggregate_metrics.items():
            if val is not None:
                print(f"  {key}: {val:.1%}")
    else:
        print("  (Open LangSmith → Datasets → expense-classification-eval-v1 → Experiments)")

    print(f"\n[eval] Done. Experiment logged under project: {active_project!r}")


if __name__ == "__main__":
    main()
