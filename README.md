# langgraph-agent-testing

> Companion repository for the Medium article  
> **"You Shipped an AI Agent to Production Without Testing It. So Did I."**

A production-quality test harness for a LangGraph expense report processing agent,
demonstrating three layers of graph-native testing: unit → integration → HITL.

---

## Why This Exists

Testing individual Python functions felt sufficient. The functions all passed. The
agent went to production. Then a $3,500 hotel charge was auto-approved with no
human ever seeing it — because the graph's conditional edge, its interrupt/resume
cycle, and its deployment config were never tested as a graph.

This repository shows the three-layer harness that should have been there from
the start.

---

## Project Structure

```
langgraph-agent-testing/
├── src/
│   ├── state.py          # ExpenseState TypedDict — shared contract, no project imports
│   ├── nodes.py          # Pure node functions — import state.py only, never graph.py
│   └── graph.py          # Compiled graph — imports nodes.py and state.py
├── tests/
│   ├── test_unit.py      # Layer 1: node functions in isolation, no graph machinery
│   ├── test_integration.py  # Layer 2: full graph, thread isolation, routing
│   └── test_hitl.py      # Layer 3: interrupt/resume cycle, two-step pattern
├── eval/
│   └── langsmith_eval.py # LangSmith evaluation against named dataset
├── pyproject.toml        # Single source of truth for all dependencies (uv)
├── langgraph.json        # LangGraph Platform deployment config
└── .env.example          # Environment variable template
```

**One-way dependency rule:**  
`state.py ← nodes.py ← graph.py`  
Nodes never import from graph.py. This is what makes unit testing possible.

---

## Getting Started

> **Note:** There is no Step 0 ("create a virtual environment") because
> `uv sync` creates and manages the virtual environment automatically.
> You never need to run `python -m venv` or `source .venv/bin/activate` —
> `uv run` handles environment activation transparently for every command.

**Step 1: Clone the repo**
```bash
git clone https://github.com/mohitagr18/langgraph-agent-testing
cd langgraph-agent-testing
```

**Step 2: Install all dependencies (runtime + dev)**
```bash
uv sync --all-extras
```
This reads `pyproject.toml` and `uv.lock`, creates a `.venv`, and installs
everything — no manual pip, no conda, no poetry.

> **Do not delete `uv.lock`.** The lockfile pins every transitive dependency to
> exact versions, ensuring your environment is byte-for-byte reproducible
> across machines and CI. Commit it to version control.

**Step 3: Copy `.env.example` → `.env` and fill in your keys**
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and LANGSMITH_API_KEY
```

**Step 4: Run the tests**
```bash
uv run pytest tests/ -v
```

---

## Running Tests Individually

```bash
# Unit tests — fast, no network, pure function assertions
uv run pytest tests/test_unit.py -v

# Integration tests — full compiled graph, thread isolation
uv run pytest tests/test_integration.py -v

# HITL tests — interrupt/resume cycle, two-step pattern
uv run pytest tests/test_hitl.py -v

# All tests
uv run pytest tests/ -v
```

## Running the LangSmith Evaluation

Requires `LANGSMITH_API_KEY` in `.env`.

```bash
uv run python eval/langsmith_eval.py
```

This will:
1. Create (or reuse) the `expense-classification-eval-v1` dataset in LangSmith
2. Run the graph against all 10 ground-truth examples
3. Score risk_flag accuracy, requires_approval accuracy, and final_status accuracy
4. Log an experiment to your LangSmith project for side-by-side comparison

---

## Deploying to LangGraph Platform

The `langgraph.json` file at the project root points to `src/graph.py:graph`.

```bash
# Test locally (requires langgraph CLI)
langgraph serve --reload

# Deploy to LangGraph Platform (connect your GitHub repo in LangSmith first)
langgraph deploy
```

The deployed graph exposes REST endpoints including:
- `POST /runs` — synchronous run
- `POST /runs/stream` — streaming run  
- `GET /threads/{id}/state` — inspect checkpoint state
- `POST /threads/{id}/state` — inject resume payload (for HITL workflows)

---

## The Three Test Layers

| Layer | File | What it proves | What it misses |
|---|---|---|---|
| Unit | `test_unit.py` | Each classification rule is correct | Routing, state propagation |
| Integration | `test_integration.py` | Conditional edges route correctly; threads are isolated | The interrupt/resume cycle |
| HITL | `test_hitl.py` | Graph pauses; state survives the gap; correct branch on resume | Nothing — this is the final proof |

---

## The Production Bug This Catches

`tests/test_hitl.py::TestInterruptFires::test_interrupt_is_present_in_step1_result`

This test fails if the approval_gate node's `interrupt()` call never fires —
meaning a $3,500 hotel charge would be auto-processed with no human ever
seeing it. Unit tests pass. Integration tests (without the HITL layer) pass.
Only the two-step invoke pattern reveals the gap.

---

## Package Management

This project uses **uv exclusively**. There is no `requirements.txt`, no
`setup.py`, and no `environment.yml`. All dependency management flows through
`pyproject.toml` and `uv.lock`.

If you need to add a dependency:
```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only dependency
```
Then commit the updated `pyproject.toml` and `uv.lock`.