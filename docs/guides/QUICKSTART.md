# Quick Start Guide

## For Your Mentor (5-Minute Setup)

### Option 1: Use In-Memory (No Setup Required)

```bash
# Install
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY='your-key-here'

# Run demo
python demo.py
```

That's it! The agent runs with built-in FAISS, BM25, and NetworkX.

---

### Option 2: Use Your Elasticsearch & Knowledge Graph

1. **Update connection details** in `demo.py`:

```python
ES_CONFIG = {
    "host": "your-es-host.com",
    "port": 9200,
    "index": "your_index_name"
}

KG_CONFIG = {
    "host": "your-kg-host.com",
    "database": "knowledge_graph",
    "user": "your_user",
    "password": "your_password"
}
```

2. **Run with production services**:

```bash
python demo.py --production
```

---

### Option 3: Use Local Docker (For Testing)

```bash
# Start services
docker-compose up -d elasticsearch postgres

# Wait for services to be ready (30 seconds)
sleep 30

# Run demo
python demo.py --local
```

---

## How to Use the Agent

### Basic Usage

```python
from main import RetrievalAgent

# Initialize
agent = RetrievalAgent(
    es_config={"host": "localhost", "port": 9200, "index": "docs"},
    kg_config={"host": "localhost", "database": "kg",
               "user": "user", "password": "pass"}
)

# Load documents (if not already in ES)
agent.load_data(documents)

# Search
results = agent.search("your query here")

# Results are ranked by relevance
for r in results:
    print(f"{r.id}: {r.score}")
```

### Integration with Test Scaffolds

```python
from demo import MentorTestInterface

# Initialize
interface = MentorTestInterface(es_config=..., kg_config=...)
interface.load_documents(documents)

# Run test queries
results = interface.search("query", top_k=10)

# Returns: [{"doc_id": ..., "score": ..., "rank": ...}]
```

---

## What You Need to Provide

To integrate with your infrastructure, we need:

1. **Elasticsearch Connection**:
   - Host, port, index name
   - Credentials (if authentication enabled)

2. **Knowledge Graph Connection**:
   - Host, database name, credentials
   - OR tell us it's embedded in ES metadata

That's all! The agent handles the rest.

---

## Output Formats

The agent can return results in any format:

```python
# Format 1: Doc IDs only
doc_ids = [r.id for r in results]

# Format 2: ID + Score
scored = [(r.id, r.score) for r in results]

# Format 3: Full metadata
full = [{"doc_id": r.id, "score": r.score,
         "metadata": r.metadata} for r in results]

# Format 4: TREC format
trec = f"{query_id} Q0 {r.id} {rank} {r.score} run_name"
```

Let us know which format your evaluation scripts expect.

---

## Testing Without Real Data

```bash
# Uses mock "Mickey Mouse" dataset
python demo.py

# Test with custom documents
python demo.py --custom

# See all output formats
python demo.py --format

# Test the integration interface
python demo.py --interface
```

---

## Architecture

The agent uses:
- **DSPy**: Typed signatures for query understanding
- **LangGraph**: Parallel map-reduce retrieval
- **Tri-Index**: Vector (FAISS) + BM25 + Graph (NetworkX)
- **Circuit Breaker**: Stops when novelty < 20%

See `ARCHITECTURE.md` for technical details.

---

## FAQ

**Q: Do I need FAISS or LlamaIndex?**
A: No! FAISS is already built-in for vector search. You don't need external vector databases.

**Q: Can it work without Elasticsearch?**
A: Yes! It uses in-memory FAISS + BM25 if ES is not available.

**Q: How do I debug?**
A: Set `DEBUG_MODE=True` in `config.py` to see all intermediate steps.

**Q: What if my documents have different metadata fields?**
A: The agent is schema-agnostic. Just pass your documents with whatever metadata fields you have.

---

## Next Steps

1. Try the in-memory demo: `python demo.py`
2. Update connection configs in `demo.py`
3. Run with your data: `python demo.py --production`
4. Let us know what output format you need

Questions? Check `demo.py` for examples or `ARCHITECTURE.md` for details.
