## Debugging Guide: When to Use What

### Decision Tree

```
Are you testing...

├─ Algorithm logic (DSPy, circuit breaker, memory)?
│  └─ NO SERVICES NEEDED
│     └─ Use: python demo.py (built-in FAISS + BM25)
│
├─ Realistic queries on small dataset?
│  └─ NO SERVICES NEEDED
│     └─ Use: python demo.py with custom documents
│
├─ Realistic queries on large corpus (1000+ docs)?
│  └─ YES, USE ELASTICSEARCH
│     └─ Use: docker-compose -f docker-compose.legal.yml up -d
│
└─ Case-based organization with relationships?
   └─ YES, USE ES + POSTGRESQL
      └─ Use: docker-compose -f docker-compose.legal.yml up -d
```

---

## Scenario 1: Debugging Core Logic (No Services)

**What you're testing:**
- DSPy query understanding
- Circuit breaker math
- Memory evolution
- Negative constraint learning

**Setup:**
```bash
export OPENAI_API_KEY='your-key'
python demo.py
```

**Why no services:**
- Built-in FAISS is fast
- Small dataset (50 docs) loads instantly
- No network latency
- Easy to add print statements and debug

**When to use:**
- Developing algorithm improvements
- Testing new DSPy signatures
- Verifying circuit breaker logic
- Debugging state machine

---

## Scenario 2: Realistic Testing with Benchmarks (No Services)

**What you're testing:**
- Performance on standard IR benchmarks (BEIR, MS MARCO)
- Precision, Recall, NDCG metrics
- Comparison with baselines

**Setup:**
```python
from datasets import load_dataset

# Load benchmark
dataset = load_dataset("BeIR/scifact")

# Convert and test
agent = RetrievalAgent()  # Built-in backend
agent.load_data(convert_dataset(dataset))

# Run evaluation
for query in dataset["queries"]:
    results = agent.search(query["text"])
    score = evaluate_ndcg(results, query["relevant_docs"])
```

**Why no services:**
- Benchmarks are usually < 10K docs
- Built-in FAISS handles this easily
- Reproducible (no external dependencies)

**When to use:**
- Writing research paper
- Comparing with baselines
- Showing improvement metrics

---

## Scenario 3: Legal Corpus with Case Structure (USE SERVICES)

**What you're testing:**
- Large corpus (thousands of documents)
- Case-based organization (multiple docs per case)
- Complex filters (injury_type + jurisdiction + settlement_range)
- Entity relationships (plaintiff → injury → expert → case)

**Setup:**
```bash
# Start services
docker-compose -f docker-compose.legal.yml up -d

# Wait for services (30 seconds)
sleep 30

# Load corpus
python load_legal_corpus.py

# Use with agent
python demo.py --local
```

**Why you need services:**
- **Elasticsearch**: Fast full-text search on thousands of docs
- **PostgreSQL**: Store case metadata + knowledge graph
- **Organization**: Cases → Documents → Entities → Relations

**When to use:**
- Testing with mentor's real data
- Debugging case-based queries
- Testing graph traversal (find similar cases)
- Production-like environment

---

## Service Requirements by Use Case

| Use Case | ES | PostgreSQL | Why |
|----------|----|-----------|----|
| **Algorithm development** | ❌ | ❌ | Built-in FAISS fast enough |
| **Benchmark testing** | ❌ | ❌ | Datasets usually small |
| **Small custom corpus (< 1K docs)** | ❌ | ❌ | Built-in works fine |
| **Large corpus (1K-100K docs)** | ✅ | ❌ | Need ES performance |
| **Case-based organization** | ✅ | ✅ | Need structured metadata |
| **Knowledge graph reasoning** | ✅ | ✅ | Need relationship queries |
| **Production deployment** | ✅ | ✅ | Need scalability |

---

## Quick Start Commands

### No Services (Fastest)
```bash
# Works immediately, no setup
export OPENAI_API_KEY='key'
python demo.py
```

### With Docker Services
```bash
# One-time setup (2 minutes)
docker-compose -f docker-compose.legal.yml up -d
sleep 30

# Load sample legal corpus
python load_legal_corpus.py

# Test with agent
python demo.py --local
```

### Stop Services When Done
```bash
docker-compose -f docker-compose.legal.yml down
```

---

## What Each Service Provides

### Elasticsearch
```python
# What you get:
- Fast full-text search (BM25)
- Metadata filtering
- Fuzzy matching
- Aggregations (group by injury_type, jurisdiction)
- Scalability (millions of docs)

# Example query:
GET /legal_documents/_search
{
  "query": {
    "bool": {
      "must": [{"match": {"content": "spinal injury"}}],
      "filter": [
        {"term": {"metadata.jurisdiction": "California"}},
        {"range": {"metadata.settlement_amount": {"gte": 500000}}}
      ]
    }
  }
}
```

### PostgreSQL
```python
# What you get:
- Case metadata storage
- Knowledge graph (entities + relations)
- Complex joins (find all docs for cases with similar injuries)
- Transactional integrity

# Example query:
SELECT d.document_id, d.content, c.settlement_amount
FROM documents d
JOIN cases c ON d.case_id = c.case_id
JOIN document_entities de ON d.document_id = de.document_id
JOIN entities e ON de.entity_id = e.entity_id
WHERE c.injury_type = 'spinal_injury'
  AND e.entity_type = 'medical_expert'
```

---

## Storage Estimates

### Built-in FAISS + BM25
- **Max documents**: ~100K (comfortable)
- **Memory usage**: ~500MB for 10K docs
- **Index time**: ~1 second per 1K docs
- **Search time**: ~50ms per query

### Elasticsearch
- **Max documents**: Millions
- **Disk usage**: ~1KB per doc (compressed)
- **Index time**: ~5 seconds per 10K docs (bulk)
- **Search time**: ~10-50ms per query

### PostgreSQL
- **Max cases**: Millions
- **Max entities**: Millions
- **Disk usage**: ~1KB per row
- **Graph traversal**: ~10ms for 2-hop queries

---

## Recommended Debugging Flow

### Phase 1: Algorithm Development (Week 1)
```bash
# No services needed
python demo.py
python test_portability.py

# Iterate on:
- DSPy signatures
- Circuit breaker logic
- Memory evolution
```

### Phase 2: Benchmark Validation (Week 2)
```python
# Still no services
# Load BEIR or MS MARCO
agent = RetrievalAgent()
agent.load_data(benchmark_docs)

# Measure NDCG, MRR, Precision@K
# Compare with baselines
```

### Phase 3: Realistic Testing (Week 3)
```bash
# Start services
docker-compose -f docker-compose.legal.yml up -d

# Load mentor's sample data (or synthetic legal corpus)
python load_legal_corpus.py

# Test case-based queries:
# - "Find spinal injury cases in California with settlements > $500K"
# - "Find cases similar to case_12345"
# - "Find all documents where Dr. Smith testified"
```

### Phase 4: Production Deployment (Week 4)
```bash
# Connect to mentor's real infrastructure
agent = RetrievalAgent(es_config={
    "host": "production-es.company.com",
    "index": "legal_documents_2024"
})
```

---

## When You ABSOLUTELY Need Services

You need Docker services if:

1. **Testing with > 1000 documents**
   - Built-in FAISS slows down
   - Need ES performance

2. **Testing case-based queries**
   - "Find all docs in case X"
   - "Find cases similar to case Y"
   - Need PostgreSQL case table

3. **Testing knowledge graph queries**
   - "Find entities connected to plaintiff X"
   - "Find path from injury A to expert B"
   - Need PostgreSQL graph tables

4. **Testing with mentor's data schema**
   - If they use ES in production
   - Need to match their index structure

5. **Preparing for production deployment**
   - Verify it works with real infrastructure
   - Test at scale

---

## TL;DR

**For 90% of debugging:** NO SERVICES NEEDED
```bash
python demo.py  # Uses built-in FAISS + BM25
```

**For realistic legal corpus testing:** USE SERVICES
```bash
docker-compose -f docker-compose.legal.yml up -d
python load_legal_corpus.py
```

**Choose based on:**
- Built-in: Fast iteration, algorithm development
- Services: Realistic testing, production validation

Start with built-in. Add services when you need scale or structure.
