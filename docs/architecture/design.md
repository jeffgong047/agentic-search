# Current Prototype Design (Baseline)

## High-Level Architecture
The system is an **Agentic Retrieval System** designed for high-precision legal search. It uses a cyclic "Plan-Execute-Verify" loop.

### Core Components

1.  **The Brain (Orchestrator)**
    *   **Impl:** `orchestrator.py` (DSPy)
    *   **Logic:** Uses tailored "Signatures" to translate user queries into structured **Search Plans**.
    *   **Output:** `SearchPlan` object containing:
        *   `primary_intent`: (e.g., "Statute_Lookup", "Entity_Resolution")
        *   `hyde_passage`: Hypothetical answer for semantic alignment.
        *   `search_queries`: List of 3 queries (Lexical, Semantic, Relational).
        *   `filter_constraints`: Metadata filters.
        *   `negative_constraints`: What to avoid.

2.  **The Backend (Recall Engine)**
    *   **Active Impl:** `elasticsearch_hybrid_backend.py` (used by `main.py`)
    *   **Logic:** **Monolithic Hybrid Search**.
        *   Executes Vector (k-NN), Keyword (BM25), and Filtering in a **single Elasticsearch query**.
        *   Uses **Reciprocal Rank Fusion (RRF)** to combine results.
    *   **Alternative:** `graph_engine.py` (Parallel LangGraph nodes) exists in codebase but is not currently the default in `main.py`.

3.  **The Control Loop (Agent Engine)**
    *   **Active Impl:** `backends_wrapper.py`
    *   **Framework:** `LangGraph`
    *   **Flow:**
        1.  **Orchestrator**: Generate Plan.
        2.  **Search**: Execute monolithic hybrid search call.
        3.  **Verify**: Check "Novelty" of results (Circuit Breaker).
        4.  **Reflect**: Update memory/constraints if verification fails.
        5.  **Loop**: Repeat if necessary.

4.  **Verification & Memory**
    *   **Verifier (`verifier.py`)**: Mathematical circuit breaker (stops loop if new results match old ones > 80%).
    *   **Memory (`memory.py`)**: Maintains a "Negative Cache" to avoid repeating mistakes.

## Current Baseline Status
*   **Operational:** `main.py` runs a "Qian Chen Disambiguation" test using the Monolithic Hybrid backend.
*   **Testing:** `debug_trace.py` (newly added) tests the *Orchestrator's* logic traces specifically.
*   **Ready for Optimization:** The `graph_engine.py` contains the "Map-Reduce" parallel logic requested for the "Diversity" feature, but it needs to be wired up to the Orchestrator's parallel outputs to fully realize the goal.

## Next Steps
1.  **Run Baseline:** Execute `debug_trace.py` (Logic Baseline) and `main.py` (Retrieval Baseline).
2.  **Optimize:** Switch `main.py` to use `graph_engine.py` and enhance Orchestrator to emit *multiple* diverse plans to feed the parallel engine.
