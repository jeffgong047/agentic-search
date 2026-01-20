# LegalBench Integration Quick Start

This guide shows you how to set up and run the LegalBench evaluation for testing your agentic search system.

## What We've Built

✅ **docker-compose.legalbench.yml** - Dedicated benchmark environment
✅ **legalbench/init_db.sql** - PostgreSQL schema for tasks, documents, qrels, and results
✅ **legalbench/download_tasks.py** - Task downloader (generates 162 synthetic legal reasoning queries)
✅ **legalbench/__init__.py** - Python package structure
✅ **LEGALBENCH_IMPLEMENTATION_PLAN.md** - Complete implementation roadmap

## Architecture Overview

```
User Query (LegalBench Task)
         ↓
DSPy Orchestrator (orchestrator.py)
         ↓
SearchPlan Generation
         ↓
Elasticsearch Hybrid Search
         ↓
Results + Metrics Tracking
         ↓
Evaluation Report
```

## Quick Start (3 Steps)

### Step 1: Start Infrastructure

```bash
cd /Users/jeffgong/paper-agent/agentic_search

# Start Elasticsearch + PostgreSQL
docker-compose -f docker-compose.legalbench.yml up -d

# Wait for services to be healthy (~30 seconds)
docker-compose -f docker-compose.legalbench.yml ps
```

Expected output:
```
NAME                        STATUS              PORTS
legalbench_elasticsearch    Up (healthy)        0.0.0.0:9200->9200/tcp
legalbench_postgres         Up (healthy)        0.0.0.0:5433->5432/tcp
```

### Step 2: Generate Synthetic Tasks

```bash
# Generate 162 LegalBench-style tasks
python -m legalbench.download_tasks

# This creates:
# - legalbench/data/tasks.jsonl (162 legal reasoning queries)
# - 6 categories: issue_spotting, rule_recall, rule_application, etc.
# - 6 domains: contract_law, tort_law, criminal_law, etc.
```

### Step 3: Test the Setup

```bash
# Verify database
docker exec -it legalbench_postgres psql -U benchuser -d legalbench_db -c "\dt"

# Should show tables: tasks, documents, qrels, evaluation_results, run_metrics

# Verify Elasticsearch
curl http://localhost:9200/_cluster/health

# Should return: {"status":"green" or "yellow"}
```

## What's Next

The implementation plan in `LEGALBENCH_IMPLEMENTATION_PLAN.md` outlines the remaining components to build:

### Remaining Components (from TODO list)

1. **load_corpus.py** - Load or generate legal document corpus
   - Option A: Download CUAD dataset from HuggingFace
   - Option B: Generate 500 synthetic legal documents

2. **create_qrels.py** - Generate relevance judgments
   - Automatic pseudo-relevance (BM25 + similarity)
   - LLM-as-judge (GPT-4 for quality)
   - Manual annotation (gold standard)

3. **build_index.py** - Build semantic index
   - Uses `llama_index_ingestion.py` (already exists!)
   - Free embedding: BAAI/bge-small-en-v1.5
   - Indexes to Elasticsearch with hybrid config

4. **run_evaluation.py** - Main evaluation loop
   - For each task: DSPy → Search → Track metrics
   - Calculate: Recall@K, Precision@K, NDCG, latency
   - Track circuit breaker efficiency

5. **metrics.py** - Metrics tracking
   - Retrieval metrics calculator
   - Latency statistics (mean, P95, P99)
   - Circuit breaker stats

6. **report.py** - Report generator
   - Generate summary.md with aggregate metrics
   - Create visualizations (latency distribution, recall by category)

## File Structure

```
agentic_search/
├── docker-compose.legalbench.yml    ✅ Created
├── Dockerfile.benchmark              ⏳ TODO
├── legalbench/
│   ├── __init__.py                   ✅ Created
│   ├── __main__.py                   ⏳ TODO
│   ├── download_tasks.py             ✅ Created
│   ├── load_corpus.py                ⏳ TODO
│   ├── create_qrels.py               ⏳ TODO
│   ├── build_index.py                ⏳ TODO
│   ├── run_evaluation.py             ⏳ TODO
│   ├── metrics.py                    ⏳ TODO
│   ├── report.py                     ⏳ TODO
│   ├── init_db.sql                   ✅ Created
│   ├── data/                         ✅ Created (empty)
│   ├── results/                      ✅ Created (empty)
│   ├── cache/                        ✅ Created (empty)
│   └── plots/                        ✅ Created (empty)
└── LEGALBENCH_IMPLEMENTATION_PLAN.md ✅ Created
```

## Integration with Existing Code

The LegalBench module **reuses** existing components:

✅ **orchestrator.py** - DSPy brain (no changes needed)
✅ **elasticsearch_hybrid_backend.py** - Hybrid search (no changes needed)
✅ **llama_index_ingestion.py** - Document chunking and indexing (reuse as-is)
✅ **data_structures.py** - SearchPlan, SearchResult, AgentState (reuse)

New code is **isolated** in the `legalbench/` module.

## Why Three Docker Compose Files?

You asked about the three docker compose files:

1. **docker-compose.yml** (Production)
   - Full stack: API + ES + PostgreSQL + Neo4j
   - For production deployment
   - Includes API service

2. **docker-compose.dev.yml** (Development)
   - Lightweight: API only, in-memory backends
   - For rapid development without Docker dependencies
   - Fastest startup

3. **docker-compose.legal.yml** (Legal Corpus)
   - Specialized: ES (1GB heap) + PostgreSQL
   - For legal document retrieval development
   - Optimized for case metadata

4. **docker-compose.legalbench.yml** (Benchmarking) ← NEW!
   - Evaluation: ES (1GB heap) + PostgreSQL + optional benchmark runner
   - Isolated environment for performance testing
   - Persistent data for reproducible results

Each serves a different purpose. Use `docker-compose.legalbench.yml` for LegalBench evaluation.

## Expected Evaluation Flow (Once Complete)

```bash
# 1. Download data
python -m legalbench download

# 2. Generate corpus (500 documents)
python -m legalbench.load_corpus --source synthetic --num-docs 500

# 3. Create qrels (pseudo-relevance)
python -m legalbench.create_qrels --method pseudo

# 4. Build index
python -m legalbench build-index \
  --corpus legalbench/data/corpus.jsonl \
  --embedding-model BAAI/bge-small-en-v1.5

# 5. Run evaluation
python -m legalbench evaluate \
  --tasks all \
  --output legalbench/results

# 6. View results
cat legalbench/results/summary.md
```

## Metrics You'll Get

After running evaluation, you'll have:

### Retrieval Metrics
- Recall@5, Recall@10, Recall@20
- Precision@5, Precision@10, Precision@20
- NDCG@10

### Performance Metrics
- Mean/Median/P95/P99 latency (ms)
- Throughput (queries per second)

### Circuit Breaker Metrics
- Early stop rate
- Mean novelty score
- Median novelty score

### Breakdowns
- By category (6 reasoning types)
- By domain (6 legal domains)
- By difficulty (easy/medium/hard)

## Next Steps

Continue implementing the remaining components in this order:

1. **load_corpus.py** - Get documents into the system
2. **create_qrels.py** - Generate ground truth
3. **build_index.py** - Index documents with embeddings
4. **run_evaluation.py** - Execute evaluation loop
5. **metrics.py** & **report.py** - Calculate and report results

Refer to `LEGALBENCH_IMPLEMENTATION_PLAN.md` for detailed implementation guidance for each component.

## Questions?

- **Why synthetic tasks?** - LegalBench paper data might not be publicly available yet. Synthetic tasks based on the paper's taxonomy are a good proxy.
- **Why synthetic documents?** - Real legal datasets (CUAD, CaseHOLD) can be integrated later. Synthetic docs let you test the pipeline quickly.
- **How accurate are synthetic qrels?** - Use pseudo-relevance (BM25 + similarity) initially, then upgrade to LLM-as-judge or manual annotation.
- **Can I use real data?** - Yes! The pipeline supports real datasets. Just implement the loaders in `load_corpus.py` and `create_qrels.py`.
