# Quick Start: Production Demo

This guide shows you how to run the production demo that demonstrates:
- **DSPy Brain**: Intent classification and query generation
- **Swappable Plumbing**: Toggle between LlamaIndex and Raw ES
- **Elasticsearch Backend**: Single system for vector + BM25 + RRF

---

## Prerequisites

1. **Python 3.9+**
2. **OpenAI API Key**
3. **Elasticsearch 8.0+** (Docker recommended)

---

## Setup Steps

### 1. Install Dependencies

```bash
cd /Users/jeffgong/paper-agent/agentic_search

# Install production requirements
pip install -r requirements_production.txt
```

**Key dependencies**:
- `dspy-ai` - The "Brain" (intent classification)
- `llama-index` - Rapid prototyping pipeline (Option 1)
- `elasticsearch` - Backend + production pipeline (Option 2)
- `sentence-transformers` - Local embeddings

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-key-here'
```

Or create a `.env` file:
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 3. Start Elasticsearch (Docker)

```bash
# Pull and run Elasticsearch 8.9+ (for native RRF support)
docker run -d \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  --name elasticsearch \
  elasticsearch:8.9.0

# Verify it's running
curl http://localhost:9200
```

---

## Running the Production Demo

The `production_demo.py` showcases 4 key demos:

### Demo 1: LlamaIndex for Rapid Prototyping

```bash
python production_demo.py llamaindex
```

**What it shows**:
- Uses `LlamaIndexPipeline` for document chunking
- DSPy orchestrator generates search plan
- Elasticsearch executes hybrid search
- Fast to prototype and prove concept

### Demo 2: Raw ES for Production

```bash
python production_demo.py raw
```

**What it shows**:
- Uses `RawElasticsearchPipeline` with direct ES calls
- Same DSPy orchestrator (no change)
- Same Elasticsearch backend (no change)
- **Only the data plumbing changed**

### Demo 3: Swappability Test

```bash
python production_demo.py swap
```

**What it shows**:
- Creates agent with LlamaIndex pipeline
- Creates agent with Raw ES pipeline
- **Both work with the same agent code**
- Proves the interface-based design

### Demo 4: Why No FAISS?

```bash
python production_demo.py why
```

**What it shows**:
- Comparison: FAISS + BM25 vs Elasticsearch
- Why Elasticsearch is sufficient
- When to use FAISS (research) vs ES (production)

### Run All Demos

```bash
python production_demo.py
```

This runs all 4 demos in sequence.

---

## Code Walkthrough

### The Agent (production_demo.py)

```python
from production_demo import ProductionRetrievalAgent

# Create agent - toggle between LlamaIndex and Raw ES
agent = ProductionRetrievalAgent(
    use_llamaindex=True,  # False for production
    es_url="http://localhost:9200",
    es_index="legal_documents"
)

# Ingest documents
documents = [{
    "id": "meta_qian_chen_001",
    "content": "Employment Agreement - Meta Platforms Inc. ...",
    "metadata": {"org": "Meta", "year": 2023}
}]
agent.ingest_documents(documents)

# Search
results = agent.search("Did Qian Chen at Meta sign a non-compete?")
```

### What Happens Under the Hood

1. **DSPy Brain** (`orchestrator.py`):
   ```python
   orchestrator = LegalOrchestrator()
   search_plan = orchestrator.forward(state)
   # Output: SearchPlan with intent, filters, queries
   ```

2. **Data Plumbing** (swappable):
   ```python
   # Option 1: LlamaIndex
   pipeline = LlamaIndexPipeline()
   chunks = pipeline.chunk_documents(documents)

   # Option 2: Raw ES
   pipeline = RawElasticsearchPipeline()
   chunks = pipeline.chunk_documents(documents)
   ```

3. **Backend** (`elasticsearch_hybrid_backend.py`):
   ```python
   backend = ElasticsearchHybridBackend()
   results = backend.hybrid_search(query)
   # One API call: vector + BM25 + RRF + filtering
   ```

---

## Customization for Your Mentor's Infrastructure

### Step 1: Adapt Raw ES Pipeline

Edit `raw_es_ingestion.py` to match your mentor's chunking logic:

```python
class RawElasticsearchPipeline(DocumentIngestionPipeline):
    def chunk_documents(self, documents, chunk_size=512, chunk_overlap=50):
        # Replace with mentor's chunking logic
        return mentor_specific_chunking(documents)

    def extract_metadata(self, document):
        # Replace with mentor's NER/entity extraction
        return mentor_ner_service(document)
```

### Step 2: Use Mentor's ES Configuration

```python
backend_config = {
    "es_url": "https://mentor-es-cluster:9200",
    "index_name": "mentor_legal_index",
    "es_credentials": ("username", "password")
}
```

### Step 3: Agent Code Doesn't Change!

```python
# Just toggle the flag
agent = ProductionRetrievalAgent(
    use_llamaindex=False,  # Use mentor's Raw ES
    es_url=backend_config["es_url"],
    es_index=backend_config["index_name"]
)
```

---

## Next Steps

1. ✅ **Run Demos**: See the swappability in action
2. ✅ **Read Architecture**: Check `ARCHITECTURE_SUMMARY.md`
3. **Customize**: Adapt Raw ES pipeline for mentor's infrastructure
4. **Evaluate**: Compare LlamaIndex vs Raw ES performance
5. **Optimize**: Use DSPy's optimization features to tune prompts
6. **Deploy**: Show mentor the production-ready system

---

## Key Takeaways

- **DSPy = Brain**: Intent classification, query generation
- **LlamaIndex = Fast Prototyping**: Prove concept quickly
- **Raw ES = Production**: Mentor's infrastructure
- **Swappable**: Interface-based design, no agent code changes
- **Single Backend**: Elasticsearch does everything (no FAISS)

**The Pitch**: "I used LlamaIndex to move fast and prove the concept. But the system is designed with clean interfaces. Here's how we swap to your Elasticsearch infrastructure—just one flag change, and the agent code stays the same."
