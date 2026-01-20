# Legal Corpus Setup for Debugging

## Quick Start (2 Commands)

```bash
# 1. Start services
docker-compose -f docker-compose.legal.yml up -d

# 2. Load sample corpus and test
python load_legal_corpus.py
```

**That's it!** You now have:
- ✅ Elasticsearch with legal documents indexed
- ✅ PostgreSQL with cases, entities, and knowledge graph
- ✅ Sample data (cases with spinal injuries, brain injuries, etc.)

---

## What You Get

### Elasticsearch Index: `legal_documents`
- **Full-text search** on document content
- **Metadata filtering**: injury_type, jurisdiction, settlement_amount
- **Optimized for**: Legal document retrieval

### PostgreSQL Database: `legal_corpus`
**Tables:**
- `cases` - Case metadata (plaintiff, defendant, injury, settlement)
- `documents` - Documents linked to cases
- `entities` - People, injuries, treatments, experts
- `relations` - Knowledge graph edges (plaintiff → sustained → injury)
- `document_entities` - Which entities appear in which docs

**Sample data included:**
- 3 cases (spinal injury, brain injury)
- Multiple document types (depositions, demand letters, medical records)
- Entity relationships (plaintiff ↔ injury ↔ expert)

---

## Using With Your Agent

### Basic Usage

```python
from main import RetrievalAgent

# Agent automatically uses ES + PostgreSQL
agent = RetrievalAgent(
    es_config={
        "host": "localhost",
        "port": 9200,
        "index": "legal_documents"
    },
    kg_config={
        "host": "localhost",
        "database": "legal_corpus",
        "user": "legal",
        "password": "legal123"
    }
)

# No need to load data - it's already in ES/PostgreSQL!

# Search
results = agent.search("spinal injury cases in California with high settlements")

for r in results:
    print(f"Case: {r.metadata['case_number']}")
    print(f"Injury: {r.metadata['injury_type']}")
    print(f"Settlement: ${r.metadata['settlement_amount']:,.2f}")
```

### Advanced Queries

```python
from load_legal_corpus import LegalCorpusLoader

loader = LegalCorpusLoader()

# Query 1: Search with filters
results = loader.search_documents(
    query="spinal injury surgery",
    injury_type="spinal_injury",
    jurisdiction="California",
    top_k=10
)

# Query 2: Get all documents for a case
case_docs = loader.get_case_documents("case_12345")

# Query 3: Direct PostgreSQL queries
with loader.pg_conn.cursor() as cur:
    # Find cases with similar injuries
    cur.execute("""
        SELECT c1.case_id, c1.plaintiff, c1.settlement_amount
        FROM cases c1
        JOIN cases c2 ON c1.injury_type = c2.injury_type
        WHERE c2.case_id = %s
          AND c1.case_id != %s
        ORDER BY ABS(c1.settlement_amount - c2.settlement_amount)
        LIMIT 5
    """, ["case_12345", "case_12345"])

    similar_cases = cur.fetchall()
```

---

## Loading Your Own Corpus

### Option 1: Load Cases Programmatically

```python
from load_legal_corpus import LegalCorpusLoader

loader = LegalCorpusLoader()

# Define your case
my_case = {
    "case_id": "case_custom_1",
    "case_number": "CV-2024-100",
    "plaintiff": "Your Plaintiff",
    "defendant": "Your Defendant",
    "injury_type": "spinal_injury",  # or brain_injury, burn_injury, etc.
    "jurisdiction": "California",
    "settlement_amount": 750000.00,
    "status": "active"
}

# Define documents
my_documents = [
    {
        "id": "case_custom_1_doc_1",
        "content": "Your document text here...",
        "metadata": {
            "doc_type": "deposition",
            "title": "Plaintiff Deposition",
            "entities": [
                {"id": "plaintiff_1", "name": "Your Plaintiff", "type": "plaintiff"},
                {"id": "injury_1", "name": "L5-S1 Herniation", "type": "injury"}
            ],
            "relations": [
                {"source": "plaintiff_1", "target": "injury_1", "type": "sustained"}
            ]
        }
    }
]

# Load
loader.load_case_with_documents(my_case, my_documents)
```

### Option 2: Load from CSV/JSON Files

```python
import pandas as pd
from load_legal_corpus import LegalCorpusLoader

# Load from CSV
df = pd.read_csv("your_cases.csv")

loader = LegalCorpusLoader()

for _, row in df.iterrows():
    case_data = {
        "case_id": row["case_id"],
        "case_number": row["case_number"],
        "plaintiff": row["plaintiff"],
        "injury_type": row["injury_type"],
        # ... other fields
    }

    # Load documents for this case
    doc_df = pd.read_csv(f"documents_{row['case_id']}.csv")
    documents = doc_df.to_dict('records')

    loader.load_case_with_documents(case_data, documents)
```

---

## Database Schema Reference

### Cases Table
```sql
case_id             VARCHAR(255) PRIMARY KEY
case_number         VARCHAR(100)
plaintiff           VARCHAR(500)
defendant           VARCHAR(500)
injury_type         VARCHAR(100)    -- spinal_injury, brain_injury, etc.
jurisdiction        VARCHAR(100)    -- California, New York, etc.
settlement_amount   DECIMAL(12, 2)
status              VARCHAR(50)     -- active, settled, dismissed
```

### Documents Table
```sql
document_id         VARCHAR(255) PRIMARY KEY
case_id             VARCHAR(255) REFERENCES cases(case_id)
doc_type            VARCHAR(100)    -- deposition, demand_letter, medical_record, etc.
title               VARCHAR(500)
content             TEXT
```

### Entities Table
```sql
entity_id           VARCHAR(255) PRIMARY KEY
entity_type         VARCHAR(100)    -- plaintiff, defendant, injury, expert, treatment
name                VARCHAR(500)
attributes          JSONB           -- Additional metadata
```

### Relations Table
```sql
source_id           VARCHAR(255) REFERENCES entities(entity_id)
target_id           VARCHAR(255) REFERENCES entities(entity_id)
relation_type       VARCHAR(100)    -- sustained, examined, treated_by, etc.
case_id             VARCHAR(255) REFERENCES cases(case_id)
```

---

## Common Queries

### Find Similar Cases
```sql
-- Cases with same injury type and similar settlement
SELECT c.*
FROM cases c
WHERE injury_type = 'spinal_injury'
  AND jurisdiction = 'California'
  AND settlement_amount BETWEEN 400000 AND 600000
ORDER BY settlement_amount DESC
```

### Find Documents by Entity
```sql
-- All documents mentioning a specific expert
SELECT d.*
FROM documents d
JOIN document_entities de ON d.document_id = de.document_id
JOIN entities e ON de.entity_id = e.entity_id
WHERE e.name = 'Dr. Sarah Smith'
  AND e.entity_type = 'medical_expert'
```

### Graph Traversal
```sql
-- Find all entities connected to a plaintiff
WITH RECURSIVE entity_graph AS (
    SELECT entity_id, 0 as depth
    FROM entities
    WHERE name = 'John Doe' AND entity_type = 'plaintiff'

    UNION

    SELECT r.target_id, eg.depth + 1
    FROM relations r
    JOIN entity_graph eg ON r.source_id = eg.entity_id
    WHERE eg.depth < 2
)
SELECT e.*
FROM entity_graph eg
JOIN entities e ON eg.entity_id = e.entity_id
```

---

## Elasticsearch Queries

### Full-Text Search with Filters
```bash
curl -X GET "localhost:9200/legal_documents/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {"match": {"content": "spinal injury settlement"}}
      ],
      "filter": [
        {"term": {"metadata.injury_type": "spinal_injury"}},
        {"term": {"metadata.jurisdiction": "California"}},
        {"range": {"metadata.settlement_amount": {"gte": 500000}}}
      ]
    }
  },
  "size": 10
}'
```

### Aggregations
```bash
# Count cases by injury type
curl -X GET "localhost:9200/legal_documents/_search" -H 'Content-Type: application/json' -d'
{
  "size": 0,
  "aggs": {
    "by_injury": {
      "terms": {"field": "metadata.injury_type"}
    },
    "avg_settlement": {
      "avg": {"field": "metadata.settlement_amount"}
    }
  }
}'
```

---

## Troubleshooting

### Services Not Starting
```bash
# Check status
docker-compose -f docker-compose.legal.yml ps

# View logs
docker-compose -f docker-compose.legal.yml logs elasticsearch
docker-compose -f docker-compose.legal.yml logs postgres

# Restart
docker-compose -f docker-compose.legal.yml restart
```

### Connection Errors
```python
# Test ES connection
from elasticsearch import Elasticsearch
es = Elasticsearch(["http://localhost:9200"])
print(es.ping())  # Should return True

# Test PostgreSQL connection
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    database="legal_corpus",
    user="legal",
    password="legal123"
)
print("Connected!")
```

### Reset Everything
```bash
# Stop and remove all data
docker-compose -f docker-compose.legal.yml down -v

# Start fresh
docker-compose -f docker-compose.legal.yml up -d
sleep 30
python load_legal_corpus.py
```

---

## Disk Space Requirements

- **Elasticsearch**: ~1KB per document (compressed)
- **PostgreSQL**: ~1KB per row
- **Total for 10,000 documents**: ~20MB
- **Total for 100,000 documents**: ~200MB

---

## Performance Benchmarks

| Operation | Time (10K docs) | Time (100K docs) |
|-----------|----------------|------------------|
| Load corpus | ~30 seconds | ~5 minutes |
| Search (ES) | ~10-20ms | ~20-50ms |
| Graph query (2-hop) | ~10ms | ~50ms |
| Case aggregation | ~5ms | ~20ms |

---

## When to Use This Setup

✅ **Use when:**
- Testing with large corpus (> 1000 docs)
- Need case-based organization
- Need knowledge graph queries
- Preparing for production
- Testing with mentor's data schema

❌ **Don't use when:**
- Just developing algorithms
- Testing with small datasets
- Running quick experiments
- Don't need case structure

For simple debugging, use: `python demo.py` (no services needed)

---

## Next Steps

1. **Start services**: `docker-compose -f docker-compose.legal.yml up -d`
2. **Load sample data**: `python load_legal_corpus.py`
3. **Test retrieval**: `python demo.py --local`
4. **Load your data**: Modify `load_legal_corpus.py` with your documents
5. **Integrate with agent**: Use the examples above

For questions, see `DEBUGGING_GUIDE.md`.
