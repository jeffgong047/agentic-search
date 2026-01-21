# Debugging Trace Tests with Claude Opus

## Overview

`test_trace_debug.py` provides **small-scale, focused tests** with **detailed traces** for debugging the agentic search system. Uses **Claude Opus 4.5** for high-quality reasoning.

## Quick Start

### 1. Get Claude Opus API Key

```bash
# Visit: https://console.anthropic.com/
# Create account → Get API key → Copy key

export ANTHROPIC_API_KEY='sk-ant-...'
```

### 2. Run Tests

```bash
cd agentic_search

# Run all trace tests
python test_trace_debug.py

# Or run individual tests in Python:
python -c "from test_trace_debug import test_1_simple_disambiguation; test_1_simple_disambiguation()"
```

## Test Suite

### TEST 1: Simple Entity Disambiguation
**Purpose**: Basic 3-document test to verify entity resolution
**What it shows**:
- DSPy orchestrator output (intent, queries, filters)
- Parallel strategy execution
- Final results with metadata

**Expected trace**:
```
🧠 DSPy Orchestrator Output:
  Intent: Entity_Resolution
  HyDE Passage: "Mickey Mouse works at Meta..."
  Search Queries:
    1. "Mickey Mouse Meta employee non-compete"
    2. "Meta researcher Mickey Mouse contract"
    3. "Mickey Mouse non-compete agreement"
  Filters: {"org": "Meta"}

📚 Retrieved 1 Documents:
  [1] ID: meta_1
      Score: 0.9234
      Org: Meta
      Content: "Mickey Mouse joined Meta in 2023..."
```

### TEST 2: Memory Evolution (Multi-Iteration)
**Purpose**: Show how agent refines search across iterations
**What it shows**:
- Iteration-by-iteration state changes
- Verified facts accumulation
- Negative cache growth
- Novelty score decay

**Expected trace**:
```
🔄 Agent State After Iteration 2:
  Step Count: 2
  Novelty Score: 0.3421
  Should Continue: False

  Verified Facts (3):
    - Mickey Mouse works at Meta
    - Focus area is LLMs
    - Signed non-compete for 1 year

  Negative Cache (2):
    - AVOID: Mickey Mouse (Shanghai) (Reason: Wrong org)
    - AVOID: Mickey Mouse (UC Berkeley) (Reason: Student, not employee)
```

### TEST 3: DSPy Chain-of-Thought Trace
**Purpose**: Inspect Claude Opus's internal reasoning
**What it shows**:
- Raw DSPy module output
- All signature fields
- LLM call history (if available)

**Expected trace**:
```
📋 Raw DSPy Output:
  primary_intent: "Entity_Resolution"

  hyde_passage:
    "A tech company employee named Mickey Mouse who works at Meta
     Research signed a non-compete agreement as part of their
     employment contract..."

  strategies: [
    {"type": "vector", "query": "..."},
    {"type": "lexical", "query": "..."},
    {"type": "graph", "query": "..."}
  ]
```

### TEST 4: Ablation (DSPy vs Raw LLM)
**Purpose**: Compare structured DSPy vs unstructured prompts
**What it shows**:
- Result quality difference
- Why typed signatures matter

**Expected trace**:
```
📊 Comparison:
  DSPy Results: 5 documents (4 relevant)
  Raw Results: 8 documents (2 relevant)

  → DSPy has higher precision due to structured output
```

## What Makes These Tests Useful?

### ✅ Small Scale
- Only 3-6 documents per test
- Fast execution (< 30 seconds per test)
- Easy to understand and verify

### ✅ Clear Traces
- Every step is logged and explained
- Shows internal agent state at each iteration
- Displays DSPy reasoning (Chain-of-Thought)

### ✅ Debugging-Focused
- Helps identify where things go wrong
- Shows memory evolution clearly
- Validates each component (orchestrator, search, memory)

### ✅ Claude Opus Quality
- Higher quality reasoning than GPT-4
- Better at complex entity disambiguation
- More reliable structured output

## Architecture Being Tested

```
User Query
    ↓
┌─────────────────────────────────────┐
│  LangGraph Agent (backends_wrapper) │
│                                      │
│  1. orchestrator_node                │
│     → DSPy generates SearchPlan      │ ← orchestrator.py
│                                      │
│  2. execute_strategy (parallel×3)    │
│     → Vector + BM25 + Graph          │ ← backend.search()
│                                      │
│  3. dedup_node                       │
│     → Merge results                  │
│                                      │
│  4. verify_node                      │
│     → Check novelty                  │ ← verifier.py
│                                      │
│  5. reflect_node                     │
│     → Update memory                  │ ← memory.py
│     → Loop if novelty > ε            │
│                                      │
└─────────────────────────────────────┘
    ↓
Final Results
```

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY='sk-ant-api03-...'
```

### "Module 'dspy' has no attribute 'Claude'"
Update DSPy:
```bash
pip install --upgrade dspy-ai
```

### Tests timeout
Claude Opus is slower than GPT-4. This is normal for better quality. If timeouts occur:
```python
# In test_trace_debug.py, increase max_tokens:
lm = dspy.Claude(
    model="claude-opus-4-5-20251101",
    max_tokens=3000,  # Increased from 2000
)
```

### Want to see more/less detail?
Edit the `print_*` functions in `test_trace_debug.py`:
- `max_show` parameter controls how many results to display
- Set `DEBUG_MODE=true` for full backend traces

## Comparison with Other Tests

| Test File | Purpose | Scale | Traces |
|-----------|---------|-------|--------|
| `test_trace_debug.py` | **Debugging** | 3-6 docs | ✅ Full |
| `test_portability.py` | Backend switching | 50 docs | ❌ Minimal |
| LegalBench pipeline | **Evaluation** | 500 docs | ❌ Metrics only |

**Use `test_trace_debug.py` when**:
- Debugging a specific issue
- Verifying new features
- Understanding agent behavior
- Showing your mentor how it works

## Next Steps

After tests pass:
1. ✅ Agent architecture is correct
2. ✅ DSPy integration works
3. ✅ Memory evolution functions
4. ✅ Ready for larger-scale evaluation

Then scale up to:
- Real legal documents
- LegalBench benchmark (162 queries, 500 docs)
- Full metrics (Recall@K, Precision@K, NDCG)
