## For Mentor: Integration Guide

### TL;DR - What You Get

A production-ready agentic retrieval system that:
- ✅ Works with **your existing infrastructure** (just provide connection details)
- ✅ Works **out-of-the-box** with built-in indexes (no setup needed)
- ✅ Can be tested **immediately** (takes 2 minutes)

### Quick Test (No Setup Required)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
export OPENAI_API_KEY='your-openai-key'

# 3. Run
python demo.py
```

That's it! The system works with built-in FAISS + BM25 + NetworkX.

### How to Connect Your Infrastructure

The agent is **backend-agnostic**. It works with whatever you have.

#### Option 1: You Have Elasticsearch

```python
from main import RetrievalAgent

agent = RetrievalAgent(es_config={
    "host": "your-es-host.com",
    "port": 9200,
    "index": "your_index_name"
})

# That's all! Agent now uses your ES
```

#### Option 2: You Have a Custom Retrieval System

Write a simple adapter (10 minutes):

```python
# your_adapter.py
from interfaces import SearchBackend
from data_structures import SearchResult

class YourSystemBackend(SearchBackend):
    def search(self, query, search_plan, negative_cache):
        # Call your existing system
        results = your_system.search(query)

        # Convert to our format
        return [
            SearchResult(
                id=r.doc_id,
                content=r.text,
                score=r.relevance,
                metadata=r.meta,
                source_index="your_system"
            )
            for r in results
        ]

    def index_documents(self, documents):
        # Optional: if you want to index new docs
        your_system.index(documents)

    def get_stats(self):
        return {"backend_type": "your_system"}
```

Then use it:

```python
from backends import BackendFactory
from your_adapter import YourSystemBackend

BackendFactory.register("your_system", YourSystemBackend)

agent = RetrievalAgent(backend_type="your_system")
```

#### Option 3: Just Use Built-in

```python
agent = RetrievalAgent()  # Uses FAISS + BM25 + NetworkX
```

### What Information Do You Need to Provide?

**Absolutely minimal:**

1. If using Elasticsearch:
   - Host, port, index name
   - (Optional) Username/password if auth enabled

2. If using custom system:
   - None! Just write the adapter (see Option 2 above)

3. If using built-in:
   - Nothing! Works out of the box

**That's literally all.**

### Test With Your Data

```python
# Load your documents
documents = [
    {
        "id": "doc_001",
        "content": "Your document text here...",
        "metadata": {"any": "fields", "you": "want"}
    },
    # ... more documents
]

agent.load_data(documents)

# Search
results = agent.search("your query")

# Results
for r in results:
    print(f"{r.id}: {r.score}")
```

### Integration With Your Test Scaffold

The agent returns results in any format you need:

```python
# Format 1: Doc IDs only
doc_ids = [r.id for r in results]

# Format 2: ID + Score
scored = [(r.id, r.score) for r in results]

# Format 3: TREC format
trec = [f"{query_id} Q0 {r.id} {i+1} {r.score} agent"
        for i, r in enumerate(results)]

# Format 4: Custom JSON
custom = {
    "query_id": "q001",
    "results": [{"doc_id": r.id, "score": r.score} for r in results]
}
```

Just tell us which format your eval scripts expect.

### Files to Look At

1. **`demo.py`** - Working examples (start here!)
2. **`QUICKSTART.md`** - 5-minute setup guide
3. **`PORTABILITY.md`** - How it works with any backend
4. **`INTEGRATION_GUIDE.md`** - Detailed integration instructions

### Architecture Highlights

- **DSPy**: Typed query understanding (not vibes-based prompting)
- **LangGraph**: Parallel search execution with deterministic control flow
- **Circuit Breaker**: Mathematical stopping criteria (no infinite loops)
- **Tri-Index**: Vector + BM25 + Graph for high recall

See `ARCHITECTURE.md` for technical deep-dive.

### Common Questions

**Q: Do I need to install Elasticsearch?**
A: No! Built-in indexes work perfectly. ES is optional for production scale.

**Q: Do I need to set up a vector database (FAISS/Pinecone)?**
A: No! FAISS is built-in. No external vector DB needed.

**Q: What if my documents have different metadata fields?**
A: Doesn't matter. The system is schema-agnostic. Just pass your documents as-is.

**Q: What if my eval scripts expect a different output format?**
A: Easy to adapt. See "Integration With Your Test Scaffold" above.

**Q: Can I test without connecting to any external services?**
A: Yes! Just run `python demo.py` - uses in-memory indexes.

**Q: How do I verify it works?**
A: Run `python test_portability.py` - runs 6 tests proving backend-agnostic design.

### Performance Expectations

- **Latency**: ~2-3 seconds per query (parallel search + reranking)
- **Iterations**: Typically 2-4 loops before convergence
- **Scalability**: Handles millions of docs (with Elasticsearch backend)

### Support

If you hit any issues:
1. Check `demo.py` for examples
2. See `PORTABILITY.md` for backend options
3. Run `python test_portability.py` to verify setup

The system is designed to be flexible - it adapts to whatever infrastructure you have.

### What Makes This Different

Most RAG systems are tightly coupled to specific databases. This one isn't.

**Traditional approach:**
```python
# Breaks if you switch from ES to Pinecone
es_client = Elasticsearch(...)
results = es_client.search(...)
```

**This approach:**
```python
# Works with ANY backend
agent = RetrievalAgent(backend_type="auto")
results = agent.search(...)
```

The agent code **never changes**, regardless of backend.

### Next Steps

1. Try the quick test: `python demo.py`
2. Review `demo.py` to see usage examples
3. Let us know what backend/format you need
4. We'll configure it to match your infrastructure

Questions? Just ask!
