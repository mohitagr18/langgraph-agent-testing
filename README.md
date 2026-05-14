# langgraph-agent-testing

> Companion repository for the Medium article
> **"You Shipped an AI Agent to Production Without Testing It. So Did I."**

A production-quality test harness for a LangGraph expense report processing agent,
demonstrating three layers of graph-native testing: unit → integration → HITL.

**→ Full explanation, diagrams, and article writing guide: [`EXPLAINER.md`](EXPLAINER.md)**

---

## Getting Started

> **No Step 0.** `uv sync` creates and manages the virtual environment automatically —
> you never need `python -m venv` or `source .venv/bin/activate`. `uv run` handles
> environment activation transparently for every command.

**Step 1: Clone the repo**
```bash
git clone https://github.com/mohitagr18/langgraph-agent-testing
cd langgraph-agent-testing
```

**Step 2: Install all dependencies (runtime + dev)**
```bash
uv sync --all-extras
```

> **Do not delete `uv.lock`.** It pins every transitive dependency to exact versions,
> keeping your environment reproducible across machines and CI. Commit it.

**Step 3: Copy `.env.example` → `.env` and fill in your keys**
```bash
cp .env.example .env
# Add your OPENAI_API_KEY and LANGSMITH_API_KEY
```

**Step 4: Run the tests**
```bash
uv run pytest tests/ -v
```

---

## Running Tests

```bash
# Layer 1 — unit: node functions in isolation, no graph, no network
uv run pytest tests/test_unit.py -v

# Layer 2 — integration: full compiled graph, routing, thread isolation
uv run pytest tests/test_integration.py -v

# Layer 3 — HITL: interrupt/resume cycle, two-step pattern
uv run pytest tests/test_hitl.py -v

# All 43 tests
uv run pytest tests/ -v
```

All tests run without an OpenAI key. No LLM calls. Everything deterministic. < 0.5s total.

## Running the LangSmith Evaluation

Requires `LANGSMITH_API_KEY` in `.env`.

```bash
uv run python eval/langsmith_eval.py
```

Creates (or reuses) the `expense-classification-eval-v1` dataset in LangSmith, runs 10
ground-truth examples through the agent, and logs a scored experiment to your project.
The dataset is also saved locally at `eval/data/expense_eval_dataset.json`.

---

## Deploying to LangGraph Platform

```bash
# Test locally
langgraph serve --reload

# Deploy (connect your GitHub repo in LangSmith first)
langgraph deploy
```

The `langgraph.json` at the project root points to `src/graph.py:graph`.

---

## Package Management

This project uses **uv exclusively**. No `requirements.txt`, no `setup.py`, no `environment.yml`.

```bash
uv add <package>        # add a runtime dependency
uv add --dev <package>  # add a dev-only dependency
```

Commit both `pyproject.toml` and `uv.lock` after any change.