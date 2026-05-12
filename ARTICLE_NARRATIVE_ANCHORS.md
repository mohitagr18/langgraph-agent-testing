# Article Narrative Anchors
### "You Shipped an AI Agent to Production Without Testing It. So Did I."

---

## Project Structure Rationale

| File/Directory | Purpose |
|---|---|
| `src/state.py` | Single shared state contract — imports nothing from this project |
| `src/nodes.py` | Pure node functions — imports `state.py` only, never `graph.py` |
| `src/graph.py` | Compiled graph — the only file that knows the graph exists |
| `tests/test_unit.py` | Layer 1: node functions in isolation, zero graph machinery |
| `tests/test_integration.py` | Layer 2: full compiled graph, routing, thread isolation |
| `tests/test_hitl.py` | Layer 3: interrupt/resume cycle, strict two-step pattern |
| `eval/langsmith_eval.py` | LangSmith evaluation with editorial comment on why eval ≠ tracing |
| `pyproject.toml` | Single source of truth for all deps (uv, no requirements.txt) |
| `langgraph.json` | LangGraph Platform deployment config pointing to `src/graph.py:graph` |
| `.env.example` | Template for required environment variables |

---

## ARTICLE NARRATIVE ANCHORS

### 1. The Confession — What Went Untested and Why It Felt Safe

**File:** `src/nodes.py`  
**Functions:** `classify_expense`, `finalize_expense`

The confession moment is: "I tested these two functions and called it done." Both
functions are pure — they take a dict in, return a dict out, and have no side effects.
They're easy to write inline assertions for in a notebook. They all pass. The author
should walk readers through `classify_expense` and explain exactly why writing tests
for it in isolation felt like sufficient coverage. The reveal is that everything
*outside* these functions — how they're wired together, which one runs when, what
happens when a human is supposed to intervene — was never touched.

The visceral detail: `finalize_expense` contains `if not state["requires_approval"]: return "cleared"`.
That line is correct in isolation. The production bug wasn't in that line — it was
that `requires_approval` was never being set to `True` when it should have been,
because the conditional edge in `graph.py` was misconfigured. No unit test could
catch that. Every unit test passed.

---

### 2. The Unit Test Layer — What It Proves and What It Dangerously Misses

**File:** `tests/test_unit.py`  
**Key tests:** `TestClassifyExpenseFlaggedVendor`, `TestFinalizeExpenseReviewed::test_fail_safe_missing_approval_decision_defaults_to_rejected`

The author should show the test class `TestClassifyExpenseClean` as the satisfying
moment — four tests, all green, pure function assertions with no setup overhead.
This is the part that *feels* like comprehensive testing.

Then pivot to the "dangerously misses" reveal. Show the docstring in `test_unit.py`:

> "What this layer dangerously misses: Whether the conditional edge routes to
> approval_gate vs. finalize correctly. Whether state written by classify_expense is
> actually visible to finalize_expense when they run inside the graph."

The key narrative anchor: unit tests prove the functions are correct. They cannot
prove the graph is correct. These are different things, and conflating them is how
$3,500 hotel charges get auto-approved.

---

### 3. The HITL Test Revelation — The Specific Moment the Production Bug Becomes Visible

**File:** `tests/test_hitl.py`  
**Key test:** `TestInterruptFires::test_interrupt_is_present_in_step1_result`

This is the article's climax. The author should quote this assertion verbatim:

```python
assert "__interrupt__" in result, (
    "Expected the graph to pause at the approval_gate node for a "
    "high-amount expense, but invoke() returned without an interrupt.  "
    "This is the production bug: the agent is auto-approving flagged "
    "expenses without human review."
)
```

The revelation: if this test had existed before the production deploy, it would have
failed. The author would have seen the assertion message — "the agent is auto-approving
flagged expenses without human review" — and fixed the graph before shipping.

The editorial moment is the phrase "a test that calls invoke() only once CANNOT test
HITL behavior." Everything the author tested before production was single-invoke.
The gap between Step 1 and Step 2 is the gap between staging and production behavior.
`TestStatePersistsAcrossResume::test_expense_fields_survive_the_pause` is the second
reveal: state written in the first invoke must survive until the second. Without a
checkpointer test, you don't know if the graph would resume into blank state.

---

### 4. The Eval and Deployment Payoff — What Changes Once You Can Prove Correctness

**File:** `eval/langsmith_eval.py`  
**Key section:** The `WHY EVALUATION IS A DIFFERENT CLASS OF EVIDENCE THAN TRACING`
comment block (lines 18–64)

The author should quote the core distinction:

> "Tracing says: 'I can see the agent ran correctly on these five examples.'
> Evaluation says: 'The agent is correct on 94.3% of a named, versioned dataset,
> and I can prove that number is stable across code changes.'"

The payoff: once you have a named evaluation dataset (`expense-classification-eval-v1`)
and a score, you can put that score in a CI/CD gate. A PR that drops accuracy from
97% to 84% gets blocked automatically — not because someone noticed during code review,
but because the evaluation ran and the threshold failed. That is the transition from
"we tested it" to "we can prove it."

**File:** `langgraph.json`  
The deployment payoff is brevity: four lines of JSON, one `langgraph deploy` command,
and the agent is live on LangGraph Platform with the same interrupt/resume behavior
the tests just proved. The article should end by pointing readers to the REST endpoint
`GET /threads/{id}/state` — because once the graph is deployed, `get_state()` in
tests is the same operation as that REST call. The test harness and the production
API speak the same language.
