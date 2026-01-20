# Integration Guide for EvenUp Mentor

## What's Been Built

A complete agentic retrieval system with:
- DSPy orchestrator for query understanding
- LangGraph for parallel search execution
- Tri-index architecture (Vector + BM25 + Graph)
- Deterministic circuit breaker (no infinite loops)
- Built-in adapters for ES, PostgreSQL, Neo4j

## What Information You Need

### Essential (Must Have)

**1. Elasticsearch Connection**
```
Host: ?
Port: ? (usually 9200)
Index name: ?
Auth: ? (username/password if required)
```

**2. Knowledge Graph Access**
```
Option A: PostgreSQL
  - Host: ?
  - Database: ?
  - User/Password: ?

Option B: Neo4j
  - Host: ?
  - Bolt port: ? (usually 7687)
  - User/Password: ?

Option C: No separate graph
  - Just metadata in Elasticsearch
```

### Optional (Nice to Have)

**3. Sample Document**
Just one example showing the metadata schema:
```json
{
  "id": "...",
  "content": "...",
  "metadata": {
    "case_id": "...",
    "doc_type": "...",
    // What other fields exist?
  }
}
```

**4. Output Format Preference**
What should `agent.search(query)` return?
- List of doc IDs: `["doc1", "doc2"]`
- List with scores: `[("doc1", 0.95), ...]`
- Full JSON: `[{"doc_id": ..., "score": ...}]`
- Other?

## How to Test Integration

### Step 1: Clone and Install
```bash
cd retrieval_agent
pip install -r requirements.txt
export OPENAI_API_KEY='your-key-here'
```

### Step 2: Test In-Memory (No External Services)
```bash
python demo.py
```

This proves the agent works end-to-end with mock data.

### Step 3: Update Connection Configs

Edit `demo.py` lines 46-75:
```python
ES_CONFIG = {
    "host": "YOUR_ES_HOST",  # ← Fill this
    "port": 9200,
    "index": "YOUR_INDEX"    # ← Fill this
}

KG_CONFIG = {
    "host": "YOUR_KG_HOST",  # ← Fill this
    ...
}
```

### Step 4: Test with Your Services
```bash
python demo.py --production
```

### Step 5: Use in Your Test Scaffold

```python
from demo import MentorTestInterface

# Initialize once
interface = MentorTestInterface(
    es_config=ES_CONFIG,
    kg_config=KG_CONFIG
)

# If documents aren't in ES already, load them
interface.load_documents(documents)

# Run test queries
for query in test_queries:
    results = interface.search(query["text"], top_k=10)
    # results = [{"doc_id": ..., "score": ..., "rank": ...}]
```

## Tool Adapter Pattern

If your services use custom APIs (not standard ES/PostgreSQL), see `tool_adapters.py` for the pattern:

```python
class YourCustomAdapter:
    def __init__(self, endpoint, api_key):
        # Connect to your service
        pass

    def search(self, query, filters, negative_constraints, top_k):
        # Call your API
        # Translate response to agent's format
        return [{"id": ..., "content": ..., "score": ...}]
```

Then:
```python
# In demo.py or your integration code
from your_adapter import YourCustomAdapter

adapter = YourCustomAdapter(endpoint="...", api_key="...")
# Agent uses this internally
```

## FAISS / Vector Database Question

**Q: Do we need to set up FAISS, Pinecone, or LlamaIndex?**

**A: No!** FAISS is already built-in for in-memory vector search. If you want production-scale vector search:

- **Option 1**: Use Elasticsearch 8.0+ with k-NN plugin (already supported)
- **Option 2**: Use the built-in FAISS (works for millions of docs)
- **Option 3**: Don't use vector search at all (BM25 + Graph only)

You don't need external vector databases unless you have specific requirements.

## PostgreSQL vs NetworkX

The agent can use:

- **NetworkX (built-in)**: In-memory graph, good for development
- **PostgreSQL (optional)**: Persistent graph storage, see `init_db.sql`
- **Neo4j (optional)**: Native graph DB, see `docker-compose.yml`
- **No graph at all**: Just use ES + Vector + BM25

Choose based on your needs. NetworkX works great for most cases.

## Debug Mode

To see exactly what the agent is doing:

```python
# In config.py
DEBUG_MODE = True
LOG_RETRIEVAL_STEPS = True
```

This prints:
- Query transformations
- Search results from each index
- Filter/rerank steps
- Novelty calculations
- Circuit breaker decisions

## Common Integration Issues

### Issue: "Cannot connect to Elasticsearch"
**Fix**: Check host/port, ensure ES is running, verify network access

### Issue: "No module named 'dspy'"
**Fix**: `pip install -r requirements.txt`

### Issue: "OPENAI_API_KEY not set"
**Fix**: `export OPENAI_API_KEY='sk-...'`

### Issue: "Results format doesn't match eval script"
**Fix**: See `demo.py` demo_result_formatting() - we can return any format

### Issue: "Different metadata schema"
**Fix**: The agent is schema-agnostic. Just pass your documents as-is.

## What to Send Your Mentor

1. Link to this repo/folder
2. `QUICKSTART.md` - 5-minute setup guide
3. `demo.py` - Working examples
4. Ask for ES/KG connection details (see "What Information You Need" above)

## Minimal Email Template

```
Subject: Retrieval Agent Ready - Need Connection Details

Hi [Name],

The retrieval agent is implemented and tested. To integrate with
EvenUp's infrastructure, I just need:

1. Elasticsearch: host, port, index name (+ auth if required)
2. Knowledge Graph: connection details or API endpoint

See attached QUICKSTART.md for setup instructions.

The demo.py file shows how to use it. It works with in-memory
indexes for testing, and can connect to real ES/KG once you
provide the connection details.

Let me know what output format your evaluation scripts expect
and I'll make sure it matches.

Code: [link to repo]

Thanks,
[Your name]
```

## Files Overview

```
retrieval_agent/
├── demo.py                 # ← Start here! Working examples
├── QUICKSTART.md           # ← 5-min setup guide
├── tool_adapters.py        # ← How to wrap external tools
│
├── docker-compose.yml      # Local dev environment (ES + PostgreSQL)
├── init_db.sql            # PostgreSQL schema for KG
│
├── main.py                # Main RetrievalAgent class
├── config.py              # Configuration and feature flags
│
└── [Other implementation files...]
```

## Support

If you hit issues integrating:
1. Check `demo.py` examples
2. Look at `tool_adapters.py` for adapter patterns
3. See `ARCHITECTURE.md` for technical details
4. Enable `DEBUG_MODE=True` to see what's happening

The system is designed to be flexible - it can adapt to whatever infrastructure you have.
