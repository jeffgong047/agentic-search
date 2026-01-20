# LegalBench Integration Implementation Plan

## Overview

This plan integrates the LegalBench benchmark (162 legal reasoning tasks from arxiv:2308.11462) into the agentic search system to evaluate retrieval performance, latency/throughput, and circuit breaker efficiency.

**Key Challenge**: LegalBench provides *queries* (reasoning tasks), not a document corpus. We need both queries and documents with relevance judgments for proper evaluation.

---

## Architecture Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEGALBENCH TEST ENVIRONMENT                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Data Acquisition                                            │
│     ├─ Download LegalBench tasks (162 queries)                  │
│     ├─ Download legal document corpus (CUAD/CaseHOLD/LexGLUE)  │
│     └─ Generate/load relevance judgments (qrels)                │
│                                                                 │
│  2. Indexing (LlamaIndex + Elasticsearch)                       │
│     ├─ Chunk documents (LlamaIndex pipeline)                    │
│     ├─ Generate embeddings (free HuggingFace models)            │
│     └─ Build semantic index in Elasticsearch                    │
│                                                                 │
│  3. Evaluation Loop                                             │
│     ├─ For each LegalBench query:                               │
│     │   ├─ DSPy Orchestrator generates SearchPlan              │
│     │   ├─ Execute retrieval (hybrid search)                   │
│     │   ├─ Track latency, novelty, circuit breaker             │
│     │   └─ Compare to ground truth (qrels)                     │
│     └─ Compute metrics: Recall@K, Precision@K, NDCG, Latency   │
│                                                                 │
│  4. Results & Reporting                                         │
│     └─ Generate evaluation report with visualizations           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Infrastructure Setup

#### 1.1 Docker Compose for LegalBench (`docker-compose.legalbench.yml`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/docker-compose.legalbench.yml`

**Purpose**: Optimized environment for benchmarking with metrics collection and data volumes.

**Services**:
```yaml
services:
  # Elasticsearch for hybrid search (vector + BM25)
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"  # 1GB heap for benchmark corpus
    ports:
      - "9200:9200"
    volumes:
      - legalbench_es_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9200/_cluster/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # PostgreSQL for document metadata and qrels storage
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: legalbench_db
      POSTGRES_USER: benchuser
      POSTGRES_PASSWORD: benchpass
    ports:
      - "5432:5432"
    volumes:
      - legalbench_pg_data:/var/lib/postgresql/data
      - ./legalbench/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U benchuser -d legalbench_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Benchmark Runner (evaluation script)
  benchmark:
    build:
      context: .
      dockerfile: Dockerfile.benchmark
    depends_on:
      elasticsearch:
        condition: service_healthy
      postgres:
        condition: service_healthy
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ES_URL=http://elasticsearch:9200
      - ES_INDEX=legalbench_documents
      - PG_HOST=postgres
      - PG_DATABASE=legalbench_db
      - PG_USER=benchuser
      - PG_PASSWORD=benchpass
      - EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
    volumes:
      - ./legalbench/data:/app/legalbench/data  # Input data
      - ./legalbench/results:/app/legalbench/results  # Output metrics
      - ./legalbench/cache:/app/legalbench/cache  # Model cache
    command: python -m legalbench.run_evaluation --tasks all --metrics all

volumes:
  legalbench_es_data:
  legalbench_pg_data:
```

**Key Features**:
- Health checks ensure services are ready before benchmarking
- Volume mounts for persistent data and results export
- Separate database for benchmark isolation
- Configurable embedding model via environment variable

---

### Phase 2: Data Acquisition & Loading

#### 2.1 LegalBench Task Downloader (`legalbench/download_tasks.py`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/download_tasks.py`

**Purpose**: Download LegalBench tasks from the official repository or arxiv supplementary materials.

**Functionality**:
```python
class LegalBenchDownloader:
    """Download LegalBench tasks and metadata"""

    def download_tasks(self, output_dir: str) -> List[Dict]:
        """
        Download 162 LegalBench tasks

        Returns:
            List of task dicts with:
            - task_id: Unique identifier
            - query: Legal reasoning question
            - category: One of 6 reasoning types
            - difficulty: Easy/Medium/Hard
            - domain: Contract law, tort law, etc.
        """

    def parse_task_categories(self) -> Dict[str, List[str]]:
        """
        Map tasks to 6 LegalBench categories:
        1. Issue-spotting
        2. Rule-recall
        3. Rule-application
        4. Rule-conclusion
        5. Interpretation
        6. Rhetorical-understanding
        """

    def save_tasks_jsonl(self, tasks: List[Dict], filepath: str):
        """Save tasks in JSONL format for evaluation"""
```

**Data Source Options**:
1. **Official LegalBench repo** (if publicly available on GitHub)
2. **Arxiv supplementary materials** (download from arxiv.org)
3. **Manually curated subset** (start with representative samples)

**Output Format** (`legalbench/data/tasks.jsonl`):
```json
{"task_id": "lb_001", "query": "Does this contract contain a non-compete clause?", "category": "issue_spotting", "domain": "contract_law"}
{"task_id": "lb_002", "query": "What is the statute of limitations for medical malpractice in this jurisdiction?", "category": "rule_recall", "domain": "tort_law"}
```

---

#### 2.2 Legal Document Corpus Loader (`legalbench/load_corpus.py`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/load_corpus.py`

**Purpose**: Load or generate a legal document corpus for retrieval testing.

**Strategy Options**:

**Option A: Use Existing Legal Dataset**
- **CUAD** (Contract Understanding Atticus Dataset): 500+ contracts with annotations
- **CaseHOLD**: Legal case citations and holdings
- **LexGLUE**: Multi-task legal NLP benchmark with documents
- **MultiLegalPile**: Large-scale legal text corpus

**Option B: Generate Synthetic Documents** (faster for prototype)
- Create documents covering LegalBench's 6 reasoning categories
- Use GPT to generate realistic legal scenarios
- Ensure diversity across contract law, tort law, criminal law, etc.

**Implementation** (Hybrid Approach):
```python
class LegalCorpusManager:
    """Manage legal document corpus for LegalBench evaluation"""

    def load_or_generate_corpus(self, source: str = "synthetic") -> List[Dict]:
        """
        Load corpus from source or generate synthetic documents

        Args:
            source: "cuad" | "casehold" | "synthetic"

        Returns:
            List of document dicts with:
            - doc_id: Unique identifier
            - content: Full document text
            - metadata: {domain, doc_type, jurisdiction, year}
        """

    def generate_synthetic_documents(self, num_docs: int = 500) -> List[Dict]:
        """
        Generate synthetic legal documents using templates

        Coverage:
        - Contract law: 150 docs (employment, NDA, sales, lease)
        - Tort law: 100 docs (negligence, malpractice, injury)
        - Criminal law: 100 docs (statutes, case law)
        - Corporate law: 100 docs (bylaws, shareholder agreements)
        - IP law: 50 docs (patents, trademarks)
        """

    def download_cuad_dataset(self) -> List[Dict]:
        """Download CUAD from HuggingFace datasets"""
        # from datasets import load_dataset
        # dataset = load_dataset("cuad")

    def chunk_documents(self, documents: List[Dict], chunk_size: int = 512) -> List[Dict]:
        """Chunk documents for indexing (reuse llama_index_ingestion.py)"""
```

**Output** (`legalbench/data/corpus.jsonl`):
```json
{"doc_id": "doc_001", "content": "EMPLOYMENT AGREEMENT\n\nThis Employment Agreement...", "metadata": {"domain": "contract_law", "doc_type": "employment_agreement", "jurisdiction": "CA"}}
```

---

#### 2.3 Ground Truth / Relevance Judgments (`legalbench/create_qrels.py`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/create_qrels.py`

**Purpose**: Create query-document relevance judgments (qrels) for metric calculation.

**Challenge**: LegalBench doesn't come with pre-labeled relevant documents.

**Solution Strategies**:

**Strategy 1: Automatic Pseudo-Relevance** (Quick Start)
```python
def create_pseudo_qrels(queries: List[Dict], documents: List[Dict]) -> pd.DataFrame:
    """
    Use BM25 + embedding similarity to create pseudo-qrels

    For each query:
    1. Retrieve top-100 documents using BM25 + vector search
    2. Label top-10 as relevant (grade=1), rest as non-relevant (grade=0)
    3. Manually verify a sample for sanity check
    """
```

**Strategy 2: LLM-as-Judge** (Better Quality)
```python
def create_llm_judged_qrels(queries: List[Dict], documents: List[Dict]) -> pd.DataFrame:
    """
    Use GPT-4 to judge relevance

    For each query-document pair:
    1. Prompt: "Given query: {query}, is this document relevant? (0-3 scale)"
    2. 0 = Not relevant, 1 = Marginally relevant, 2 = Relevant, 3 = Highly relevant
    3. Store judgments for evaluation
    """
```

**Strategy 3: Manual Annotation** (Gold Standard, Time-Intensive)
- Sample 50-100 queries from LegalBench
- For each query, manually label top-50 retrieved docs
- Use for final evaluation

**Output Format** (`legalbench/data/qrels.tsv`):
```
task_id	doc_id	relevance_grade
lb_001	doc_042	2
lb_001	doc_143	3
lb_001	doc_287	1
lb_002	doc_051	3
```

---

### Phase 3: Semantic Index Building

#### 3.1 Index Builder (`legalbench/build_index.py`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/build_index.py`

**Purpose**: Build Elasticsearch index using LlamaIndex with free embedding models.

**Integration**: Reuse existing `llama_index_ingestion.py` pipeline.

**Embedding Models** (Free HuggingFace Models via LlamaIndex):
1. **BAAI/bge-small-en-v1.5** (current default in llama_index_ingestion.py)
   - 384 dimensions
   - Good for general semantic search

2. **sentence-transformers/all-MiniLM-L6-v2** (lightweight)
   - 384 dimensions
   - Faster inference

3. **BAAI/bge-base-en-v1.5** (better quality)
   - 768 dimensions
   - Higher accuracy, slower

**Implementation**:
```python
from llama_index_ingestion import LlamaIndexPipeline
from elasticsearch import Elasticsearch

class LegalBenchIndexBuilder:
    """Build semantic index for LegalBench evaluation"""

    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.pipeline = LlamaIndexPipeline(embedding_model=embedding_model)
        self.es = Elasticsearch("http://localhost:9200")

    def build_index(self, corpus_path: str, index_name: str = "legalbench_documents"):
        """
        Build complete index from corpus

        Steps:
        1. Load documents from corpus.jsonl
        2. Chunk documents (512 chars, 50 overlap)
        3. Generate embeddings using LlamaIndex
        4. Index to Elasticsearch with hybrid config (vector + BM25)
        5. Create metadata mappings for filtering
        """
        # Load corpus
        documents = self._load_corpus(corpus_path)

        # Chunk and embed using LlamaIndex pipeline
        chunks = self.pipeline.chunk_documents(documents, chunk_size=512, chunk_overlap=50)

        # Index to Elasticsearch
        indexed_count = self.pipeline.index_to_backend(
            chunks,
            backend_config={"es_url": "http://localhost:9200", "index_name": index_name}
        )

        print(f"✓ Indexed {indexed_count} chunks from {len(documents)} documents")

    def verify_index(self, index_name: str):
        """Verify index was created correctly"""
        stats = self.es.indices.stats(index=index_name)
        doc_count = stats['indices'][index_name]['total']['docs']['count']
        print(f"Index '{index_name}' contains {doc_count} documents")

        # Test search
        test_query = "non-compete agreement"
        results = self.es.search(index=index_name, body={
            "query": {"match": {"content": test_query}},
            "size": 5
        })
        print(f"Test search for '{test_query}' returned {len(results['hits']['hits'])} results")
```

**Usage**:
```bash
# Build index
python -m legalbench.build_index \
  --corpus legalbench/data/corpus.jsonl \
  --index legalbench_documents \
  --embedding-model BAAI/bge-small-en-v1.5
```

---

### Phase 4: Evaluation Script

#### 4.1 Main Evaluation Runner (`legalbench/run_evaluation.py`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/run_evaluation.py`

**Purpose**: Execute full evaluation loop and compute all metrics.

**Metrics to Compute**:
1. **Retrieval Metrics**: Recall@K, Precision@K, NDCG@K (K=5,10,20)
2. **Latency Metrics**: Mean, median, P95, P99 query latency
3. **Throughput**: Queries per second
4. **Circuit Breaker Efficiency**: Novelty scores, early stopping rate

**Architecture**:
```python
class LegalBenchEvaluator:
    """Main evaluation orchestrator"""

    def __init__(self, config: Dict):
        self.orchestrator = LegalOrchestrator()  # DSPy brain
        self.backend = ElasticsearchHybridBackend(config["es_url"], config["es_index"])
        self.tasks = self._load_tasks(config["tasks_path"])
        self.qrels = self._load_qrels(config["qrels_path"])
        self.metrics_tracker = MetricsTracker()

    def run_evaluation(self, max_queries: int = None) -> Dict:
        """
        Run full evaluation loop

        For each LegalBench task:
        1. Generate SearchPlan using DSPy orchestrator
        2. Execute hybrid search
        3. Track latency, novelty, circuit breaker
        4. Record retrieved doc IDs
        5. Compare to qrels for metric calculation

        Returns:
            Evaluation results with all metrics
        """
        results = []

        for task in tqdm(self.tasks[:max_queries], desc="Evaluating"):
            start_time = time.time()

            # Step 1: DSPy generates SearchPlan
            state = create_initial_state(task["query"])
            search_plan = self.orchestrator.forward(state)

            # Step 2: Execute retrieval
            search_query = SearchQuery(
                text=task["query"],
                vector_query=search_plan.hyde_passage,
                keyword_query=search_plan.search_queries[0],
                filters=search_plan.filter_constraints,
                negative_constraints=search_plan.negative_constraints,
                top_k=20
            )
            retrieved_docs = self.backend.hybrid_search(search_query)

            # Step 3: Track metrics
            latency = time.time() - start_time
            retrieved_ids = [doc.id for doc in retrieved_docs]

            # Step 4: Calculate retrieval metrics
            relevant_docs = self._get_relevant_docs(task["task_id"], self.qrels)
            metrics = self._calculate_metrics(retrieved_ids, relevant_docs)

            # Step 5: Track circuit breaker (if doing iterative retrieval)
            # novelty_score = self._calculate_novelty(retrieved_docs, state["known_doc_ids"])

            results.append({
                "task_id": task["task_id"],
                "query": task["query"],
                "category": task["category"],
                "latency_ms": latency * 1000,
                "num_retrieved": len(retrieved_docs),
                "recall@5": metrics["recall@5"],
                "recall@10": metrics["recall@10"],
                "precision@5": metrics["precision@5"],
                "ndcg@10": metrics["ndcg@10"],
                "retrieved_ids": retrieved_ids[:10]  # Top 10 for analysis
            })

        # Aggregate metrics
        aggregate_metrics = self._aggregate_results(results)

        return {
            "per_query_results": results,
            "aggregate_metrics": aggregate_metrics
        }

    def _calculate_metrics(self, retrieved_ids: List[str], relevant_docs: Set[str]) -> Dict:
        """Calculate Recall@K, Precision@K, NDCG@K"""
        metrics = {}

        for k in [5, 10, 20]:
            retrieved_at_k = set(retrieved_ids[:k])
            relevant_retrieved = retrieved_at_k & relevant_docs

            # Recall@K
            recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
            metrics[f"recall@{k}"] = recall

            # Precision@K
            precision = len(relevant_retrieved) / k if k > 0 else 0.0
            metrics[f"precision@{k}"] = precision

        # NDCG@10
        metrics["ndcg@10"] = self._calculate_ndcg(retrieved_ids[:10], relevant_docs)

        return metrics

    def _calculate_ndcg(self, retrieved_ids: List[str], relevant_docs: Set[str]) -> float:
        """Calculate NDCG@K using binary relevance"""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids, start=1):
            relevance = 1.0 if doc_id in relevant_docs else 0.0
            dcg += relevance / np.log2(i + 1)

        # Ideal DCG (all relevant docs at top)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant_docs) + 1, len(retrieved_ids) + 1)))

        return dcg / idcg if idcg > 0 else 0.0
```

---

#### 4.2 Metrics Tracker (`legalbench/metrics.py`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/metrics.py`

**Purpose**: Track and aggregate all evaluation metrics.

```python
class MetricsTracker:
    """Track retrieval, latency, and circuit breaker metrics"""

    def __init__(self):
        self.latencies = []
        self.novelty_scores = []
        self.early_stops = 0
        self.total_queries = 0

    def track_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def track_novelty(self, novelty_score: float, stopped_early: bool):
        self.novelty_scores.append(novelty_score)
        if stopped_early:
            self.early_stops += 1

    def get_latency_stats(self) -> Dict:
        """Calculate latency statistics"""
        return {
            "mean_ms": np.mean(self.latencies),
            "median_ms": np.median(self.latencies),
            "p95_ms": np.percentile(self.latencies, 95),
            "p99_ms": np.percentile(self.latencies, 99),
            "min_ms": np.min(self.latencies),
            "max_ms": np.max(self.latencies)
        }

    def get_throughput(self, total_time_sec: float) -> float:
        """Calculate queries per second"""
        return len(self.latencies) / total_time_sec

    def get_circuit_breaker_stats(self) -> Dict:
        """Calculate circuit breaker efficiency"""
        return {
            "early_stop_rate": self.early_stops / len(self.novelty_scores) if self.novelty_scores else 0.0,
            "mean_novelty": np.mean(self.novelty_scores) if self.novelty_scores else 0.0,
            "median_novelty": np.median(self.novelty_scores) if self.novelty_scores else 0.0
        }
```

---

#### 4.3 Report Generator (`legalbench/report.py`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/report.py`

**Purpose**: Generate evaluation report with visualizations.

```python
class EvaluationReporter:
    """Generate evaluation reports and visualizations"""

    def generate_report(self, results: Dict, output_dir: str):
        """
        Generate comprehensive evaluation report

        Output:
        - results.json: Raw evaluation data
        - summary.md: Markdown report with tables
        - plots/: Visualizations (latency distribution, recall by category, etc.)
        """

    def create_summary_markdown(self, results: Dict) -> str:
        """Create markdown summary with aggregate metrics"""

    def plot_latency_distribution(self, latencies: List[float], output_path: str):
        """Plot latency histogram and CDF"""

    def plot_recall_by_category(self, results: List[Dict], output_path: str):
        """Plot Recall@10 grouped by LegalBench task category"""

    def plot_precision_recall_curve(self, results: List[Dict], output_path: str):
        """Plot precision-recall curve across all queries"""
```

---

### Phase 5: Integration & Execution

#### 5.1 Database Schema (`legalbench/init_db.sql`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/init_db.sql`

```sql
-- LegalBench evaluation database schema

-- Tasks table
CREATE TABLE tasks (
    task_id VARCHAR(50) PRIMARY KEY,
    query TEXT NOT NULL,
    category VARCHAR(50),
    domain VARCHAR(50),
    difficulty VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents table
CREATE TABLE documents (
    doc_id VARCHAR(100) PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Relevance judgments (qrels)
CREATE TABLE qrels (
    task_id VARCHAR(50) REFERENCES tasks(task_id),
    doc_id VARCHAR(100) REFERENCES documents(doc_id),
    relevance_grade INTEGER CHECK (relevance_grade >= 0 AND relevance_grade <= 3),
    PRIMARY KEY (task_id, doc_id)
);

-- Evaluation results
CREATE TABLE evaluation_results (
    eval_id SERIAL PRIMARY KEY,
    task_id VARCHAR(50) REFERENCES tasks(task_id),
    retrieved_ids TEXT[],
    latency_ms FLOAT,
    recall_at_5 FLOAT,
    recall_at_10 FLOAT,
    precision_at_5 FLOAT,
    ndcg_at_10 FLOAT,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookups
CREATE INDEX idx_qrels_task ON qrels(task_id);
CREATE INDEX idx_qrels_doc ON qrels(doc_id);
CREATE INDEX idx_eval_results_task ON evaluation_results(task_id);
```

---

#### 5.2 Dockerfile for Benchmark Runner (`Dockerfile.benchmark`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/Dockerfile.benchmark`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# Install additional benchmark dependencies
RUN pip install --no-cache-dir \
    datasets \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    scikit-learn

# Copy codebase
COPY . .

# Create directories for data and results
RUN mkdir -p legalbench/data legalbench/results legalbench/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/legalbench/cache

CMD ["python", "-m", "legalbench.run_evaluation"]
```

---

#### 5.3 Main Entry Point (`legalbench/__main__.py`)

**File**: `/Users/jeffgong/paper-agent/agentic_search/legalbench/__main__.py`

```python
"""
LegalBench Evaluation Entry Point

Usage:
    python -m legalbench --help
    python -m legalbench download --output legalbench/data
    python -m legalbench build-index --corpus legalbench/data/corpus.jsonl
    python -m legalbench evaluate --tasks all --output legalbench/results
"""

import argparse
from legalbench.download_tasks import LegalBenchDownloader
from legalbench.load_corpus import LegalCorpusManager
from legalbench.build_index import LegalBenchIndexBuilder
from legalbench.run_evaluation import LegalBenchEvaluator

def main():
    parser = argparse.ArgumentParser(description="LegalBench Evaluation for Agentic Search")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download LegalBench tasks and corpus")
    download_parser.add_argument("--output", default="legalbench/data", help="Output directory")

    # Build index command
    build_parser = subparsers.add_parser("build-index", help="Build semantic index")
    build_parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    build_parser.add_argument("--index", default="legalbench_documents", help="Index name")
    build_parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--tasks", default="all", help="Tasks to evaluate (all | category | task_id)")
    eval_parser.add_argument("--output", default="legalbench/results", help="Output directory")
    eval_parser.add_argument("--max-queries", type=int, help="Limit number of queries")

    args = parser.parse_args()

    if args.command == "download":
        print("Downloading LegalBench tasks and corpus...")
        downloader = LegalBenchDownloader()
        corpus_manager = LegalCorpusManager()

        # Download tasks
        tasks = downloader.download_tasks(args.output)
        print(f"✓ Downloaded {len(tasks)} tasks")

        # Generate corpus
        corpus = corpus_manager.load_or_generate_corpus(source="synthetic")
        print(f"✓ Generated {len(corpus)} documents")

    elif args.command == "build-index":
        print(f"Building index from {args.corpus}...")
        builder = LegalBenchIndexBuilder(embedding_model=args.embedding_model)
        builder.build_index(args.corpus, args.index)
        builder.verify_index(args.index)

    elif args.command == "evaluate":
        print(f"Running evaluation on {args.tasks}...")
        evaluator = LegalBenchEvaluator({
            "es_url": os.getenv("ES_URL", "http://localhost:9200"),
            "es_index": os.getenv("ES_INDEX", "legalbench_documents"),
            "tasks_path": "legalbench/data/tasks.jsonl",
            "qrels_path": "legalbench/data/qrels.tsv"
        })
        results = evaluator.run_evaluation(max_queries=args.max_queries)

        # Generate report
        reporter = EvaluationReporter()
        reporter.generate_report(results, args.output)
        print(f"✓ Results saved to {args.output}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

---

## Execution Workflow

### Step 1: Setup Environment

```bash
# Start services
docker-compose -f docker-compose.legalbench.yml up -d

# Wait for health checks
docker-compose -f docker-compose.legalbench.yml ps
```

### Step 2: Download Data

```bash
# Download LegalBench tasks and generate corpus
python -m legalbench download --output legalbench/data

# Files created:
# - legalbench/data/tasks.jsonl (162 tasks)
# - legalbench/data/corpus.jsonl (500 documents)
# - legalbench/data/qrels.tsv (relevance judgments)
```

### Step 3: Build Index

```bash
# Build semantic index using LlamaIndex
python -m legalbench build-index \
  --corpus legalbench/data/corpus.jsonl \
  --index legalbench_documents \
  --embedding-model BAAI/bge-small-en-v1.5

# This will:
# 1. Chunk 500 documents → ~2000 chunks
# 2. Generate embeddings (384-dim vectors)
# 3. Index to Elasticsearch with hybrid config
```

### Step 4: Run Evaluation

```bash
# Full evaluation (all 162 tasks)
python -m legalbench evaluate \
  --tasks all \
  --output legalbench/results

# Or test with subset
python -m legalbench evaluate \
  --tasks issue_spotting \
  --max-queries 50 \
  --output legalbench/results/subset
```

### Step 5: View Results

```bash
# Results directory structure:
legalbench/results/
├── results.json           # Raw per-query results
├── summary.md            # Aggregate metrics report
├── plots/
│   ├── latency_distribution.png
│   ├── recall_by_category.png
│   └── precision_recall_curve.png
└── evaluation_log.txt    # Detailed logs
```

---

## Expected Output

### Aggregate Metrics (`summary.md`)

```markdown
# LegalBench Evaluation Summary

**Evaluation Date**: 2026-01-07
**Total Queries**: 162
**Corpus Size**: 500 documents (2000 chunks)
**Embedding Model**: BAAI/bge-small-en-v1.5

## Retrieval Metrics

| Metric       | Value  |
|--------------|--------|
| Recall@5     | 0.523  |
| Recall@10    | 0.687  |
| Recall@20    | 0.812  |
| Precision@5  | 0.614  |
| Precision@10 | 0.521  |
| NDCG@10      | 0.642  |

## Latency Metrics

| Metric    | Value (ms) |
|-----------|------------|
| Mean      | 142.3      |
| Median    | 128.5      |
| P95       | 256.8      |
| P99       | 312.1      |
| Throughput| 7.02 QPS   |

## Circuit Breaker Efficiency

| Metric           | Value |
|------------------|-------|
| Early Stop Rate  | 0.34  |
| Mean Novelty     | 0.42  |
| Median Novelty   | 0.38  |

## Breakdown by Category

| Category              | Queries | Recall@10 |
|-----------------------|---------|-----------|
| Issue Spotting        | 35      | 0.712     |
| Rule Recall           | 28      | 0.693     |
| Rule Application      | 32      | 0.658     |
| Rule Conclusion       | 25      | 0.671     |
| Interpretation        | 22      | 0.702     |
| Rhetorical Understand | 20      | 0.645     |
```

---

## File Structure Summary

```
agentic_search/
├── docker-compose.legalbench.yml    # Benchmark environment
├── Dockerfile.benchmark              # Benchmark runner image
├── legalbench/
│   ├── __init__.py
│   ├── __main__.py                  # CLI entry point
│   ├── download_tasks.py            # Download LegalBench tasks
│   ├── load_corpus.py               # Load/generate document corpus
│   ├── create_qrels.py              # Generate relevance judgments
│   ├── build_index.py               # Build semantic index
│   ├── run_evaluation.py            # Main evaluation loop
│   ├── metrics.py                   # Metrics tracking
│   ├── report.py                    # Report generation
│   ├── init_db.sql                  # Database schema
│   ├── data/                        # Input data (generated)
│   │   ├── tasks.jsonl
│   │   ├── corpus.jsonl
│   │   └── qrels.tsv
│   ├── results/                     # Output results
│   └── cache/                       # Model cache
└── (existing files...)
```

---

## Integration with Existing Architecture

### Reuse Existing Components

✅ **DSPy Orchestrator** (`orchestrator.py`)
- No changes needed
- Use `LegalOrchestrator` to generate `SearchPlan` for each query

✅ **Elasticsearch Hybrid Backend** (`elasticsearch_hybrid_backend.py`)
- No changes needed
- Use `ElasticsearchHybridBackend.hybrid_search()` for retrieval

✅ **LlamaIndex Ingestion** (`llama_index_ingestion.py`)
- Reuse `LlamaIndexPipeline` for chunking and indexing
- Configure embedding model via constructor

✅ **Data Structures** (`data_structures.py`)
- Use `SearchPlan`, `SearchResult`, `AgentState`

### New Components

🆕 **LegalBench Module** (`legalbench/`)
- Self-contained evaluation suite
- Doesn't modify existing codebase
- Can be run independently

---

## Success Criteria

✅ **Infrastructure**: Docker Compose starts all services with health checks
✅ **Data Loading**: Successfully load 162 tasks + corpus + qrels
✅ **Indexing**: Build semantic index with >95% success rate
✅ **Evaluation**: Run all 162 queries and compute metrics
✅ **Reporting**: Generate summary with visualizations
✅ **Performance**: <200ms median latency, >5 QPS throughput

---

## Next Steps After Implementation

1. **Baseline Evaluation**: Run with current embedding model
2. **Model Comparison**: Test BAAI/bge-small vs all-MiniLM vs bge-base
3. **Prompt Optimization**: Use DSPy to optimize orchestrator prompts
4. **Error Analysis**: Investigate low-recall queries
5. **Production Deployment**: Integrate best-performing config into main system
