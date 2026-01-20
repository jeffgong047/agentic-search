

# Portability Guarantee

## The Problem You Asked About

> "How can I ensure when I ship the code to my mentor it will run regardless of what database or retrieval system he uses?"

## The Solution

This codebase uses **interface abstraction** (dependency inversion principle). The agent depends on **interfaces**, not concrete implementations.

### Architecture

```
┌─────────────────────────────────────┐
│     RetrievalAgent (Your Code)      │
│                                     │
│  Depends on: SearchBackend          │ ← Interface only!
│  (NOT Elasticsearch, NOT FAISS)     │
└──────────────┬──────────────────────┘
               │
        ┌──────┴───────┐
        ▼              ▼
  ┌─────────┐    ┌─────────────┐
  │Built-in │    │  Your Custom│
  │Backends │    │  Backend    │
  └─────────┘    └─────────────┘
     │                 │
     ├─ tri-index      ├─ Your API
     ├─ elasticsearch  ├─ Your DB
     └─ hybrid         └─ Your System
```

## How It Works

### 1. Define the Interface (`interfaces.py`)

```python
class SearchBackend(ABC):
    @abstractmethod
    def index_documents(self, documents):
        pass

    @abstractmethod
    def search(self, query, search_plan, negative_cache):
        pass

    @abstractmethod
    def get_stats(self):
        pass
```

### 2. Agent Depends on Interface (`main.py`)

```python
class RetrievalAgent:
    def __init__(self, backend_type=None):
        # Agent uses the interface, not a specific implementation
        self.backend = create_best_backend(backend_type)

    def search(self, query):
        # Calls interface method (works with ANY backend!)
        return self.backend.search(query, plan, cache)
```

### 3. Multiple Implementations (`backends.py`)

```python
# Built-in: Always works
class TriIndexBackend(SearchBackend):
    def search(self, query, plan, cache):
        # Uses FAISS + BM25 + NetworkX
        return results

# Built-in: If ES available
class ElasticsearchBackend(SearchBackend):
    def search(self, query, plan, cache):
        # Uses Elasticsearch
        return results

# Custom: User-provided
class YourCustomBackend(SearchBackend):
    def search(self, query, plan, cache):
        # Uses whatever you want!
        return results
```

### 4. Register & Use

```python
# Register your custom backend
BackendFactory.register("mybackend", YourCustomBackend)

# Agent automatically uses it
agent = RetrievalAgent(backend_type="mybackend")
```

## Portability Guarantees

### ✅ Guarantee 1: Works Without Any External Services

```python
# No Elasticsearch, no PostgreSQL, no internet - still works!
agent = RetrievalAgent()  # Uses built-in tri-index
agent.load_data(documents)
results = agent.search("query")
```

**Why**: TriIndexBackend uses only built-in Python (FAISS, BM25, NetworkX).

### ✅ Guarantee 2: Works With Elasticsearch

```python
agent = RetrievalAgent(es_config={"host": "...", "index": "..."})
agent.search("query")
```

**Why**: If ES config provided, auto-detects and uses ElasticsearchBackend.

### ✅ Guarantee 3: Works With Your Mentor's System

```python
# Step 1: Create adapter (5 minutes)
class MentorBackend(SearchBackend):
    def search(self, query, plan, cache):
        # Call mentor's API/DB
        return mentor_system.search(query)

# Step 2: Register
BackendFactory.register("mentor", MentorBackend)

# Step 3: Use
agent = RetrievalAgent(backend_type="mentor")
```

**Why**: You implement the 3 methods (`index_documents`, `search`, `get_stats`), and it just works.

## For Debugging on Your End

### Strategy 1: Use Built-in Backend

```python
# Debug with mock data, no external services
agent = RetrievalAgent()
agent.load_data(get_mock_dataset())
results = agent.search("test query")

# Verify logic works
assert len(results) > 0
```

### Strategy 2: Use File-Based Backend

```python
# Ultra-simple backend using JSON files
BackendFactory.register("file", FileBasedBackend)

agent = RetrievalAgent(backend_type="file")
agent.load_data(your_test_documents)

# Results saved to ./data/documents.json
# Easy to inspect and debug
```

### Strategy 3: Use Benchmark Datasets

```python
# Load BEIR, MS MARCO, or custom benchmarks
from datasets import load_dataset

benchmark = load_dataset("BeIR/trec-covid")
documents = convert_to_our_format(benchmark)

agent = RetrievalAgent()  # Uses built-in backend
agent.load_data(documents)

# Run evaluation
for query in benchmark["queries"]:
    results = agent.search(query["text"])
    score = evaluate(results, query["relevant_docs"])
```

## For Shipping to Your Mentor

### Option 1: Mentor Provides Backend Config

```python
# mentor_config.py (they fill this out)
ES_CONFIG = {
    "host": "their-es-host",
    "port": 9200,
    "index": "their-index"
}

# Your code (unchanged)
from mentor_config import ES_CONFIG
agent = RetrievalAgent(es_config=ES_CONFIG)
```

### Option 2: Mentor Wraps Their System

```python
# mentor_backend.py (they write this)
class MentorRetrievalBackend(SearchBackend):
    def search(self, query, plan, cache):
        return our_existing_system.search(query)

# Your code (unchanged)
agent = RetrievalAgent(backend_type="mentor")
```

### Option 3: Auto-Detection (Easiest!)

```python
# Your code tries backends in order:
# 1. If ES config → use ES
# 2. If file exists → use file-based
# 3. Else → use tri-index

agent = RetrievalAgent()  # Auto-detects best backend
```

## The Key Insight

**The agent code NEVER changes**, regardless of backend.

```python
# This line works with:
# - FAISS + BM25 + NetworkX (built-in)
# - Elasticsearch (external)
# - Your API (custom)
# - MongoDB (custom)
# - File-based (simple)
# - Anything else (just implement the interface)

results = agent.search("query")
```

## Testing Portability

### Test 1: Verify Built-in Works

```bash
python demo.py
# Uses tri-index backend (no external deps)
```

### Test 2: Verify Elasticsearch Works

```bash
docker-compose up -d elasticsearch
python demo.py --local
# Uses Elasticsearch backend
```

### Test 3: Verify Custom Works

```bash
python custom_backend_example.py
# Uses file-based custom backend
```

### Test 4: Benchmark Different Backends

```python
from benchmarking import compare_backends

results = compare_backends(
    backends=["tri-index", "elasticsearch", "file"],
    queries=test_queries
)

# Compare:
# - Latency
# - Precision
# - Ease of setup
```

## Common Integration Scenarios

### Scenario 1: Mentor Has Elasticsearch

```python
# You provide:
agent = RetrievalAgent(es_config={
    "host": "FILL_THIS",
    "port": 9200,
    "index": "FILL_THIS"
})

# Mentor fills in host/index → works immediately
```

### Scenario 2: Mentor Has Custom API

```python
# mentor_adapter.py
class MentorAPIBackend(SearchBackend):
    def search(self, query, plan, cache):
        import requests
        r = requests.post("http://internal-api/search", json={"q": query})
        return parse_to_search_results(r.json())

BackendFactory.register("mentor_api", MentorAPIBackend)

# Usage
agent = RetrievalAgent(backend_type="mentor_api")
```

### Scenario 3: Mentor Has Nothing

```python
# Just use built-in!
agent = RetrievalAgent()

# Works perfectly with in-memory indexes
# Mentor can test with their data immediately
```

## Files You Need to Understand

1. **`interfaces.py`** - Defines contracts (read this first!)
2. **`backends.py`** - Built-in implementations
3. **`custom_backend_example.py`** - How to add new backends
4. **`main.py`** - Agent uses backends via interface

## Summary

**Q**: How to ensure it runs with any database/retrieval system?

**A**:
1. Agent depends on `SearchBackend` interface (not concrete classes)
2. Built-in backends work out-of-the-box (tri-index, ES, hybrid)
3. Custom backends: implement 3 methods → instant compatibility
4. Auto-detection picks best available backend
5. No code changes needed in agent itself

**You can debug locally with built-in backends, and ship to production with ANY backend your mentor uses.**

The code is **truly portable**.
