# High-SNR Agentic RAG System

A modular, measurable RAG agent that solves entity collision and infinite loop problems using **DSPy** (Logic Compiler) + **LangGraph** (State Orchestration) + deterministic circuit breakers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DSPy Orchestrator                        │
│              (Typed Signatures, HyDE)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  LangGraph State      │
         │  Machine (Parallel    │
         │  Map-Reduce)          │
         └───────┬───────────────┘
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   ┌──────┐ ┌──────┐ ┌──────┐
   │Vector│ │ BM25 │ │Graph │
   │Search│ │Search│ │Search│
   └──┬───┘ └──┬───┘ └──┬───┘
      │        │        │
      └────────┼────────┘
               ▼
      ┌─────────────────┐
      │ Cascade Funnel  │
      │ (Filter+Rerank) │
      └────────┬────────┘
               ▼
      ┌─────────────────┐
      │ Novelty Circuit │
      │    Breaker      │
      └────────┬────────┘
               ▼
      ┌─────────────────┐
      │ Memory Evolution│
      │  (Reflection)   │
      └─────────────────┘
```

## Documentation

- **[Quick Start](docs/guides/QUICKSTART.md)**: Get up and running in 5 minutes.
- **[Architecture](docs/architecture/ARCHITECTURE.md)**: Deep dive into DSPy + Graph architecture.
- **[Scripts & Tools](scripts/README.md)**: Guide to ingestion and analysis scripts.

## Key Features

1. **Typed Schema Compilation (DSPy)**: No more "vibes-based" prompting. Uses mathematical optimization to generate search plans.

2. **Deterministic Circuit Breaker**: Stops loops based on set-theory novelty calculation, not LLM guessing.

3. **Tri-Index Retrieval**: Combines Vector (semantic), BM25 (lexical), and Graph (relational) search.

4. **Negative Constraint Learning**: Learns from failed paths to avoid checking wrong entities twice.

5. **Counterfactual Ablation**: Feature flags enable A/B testing to prove component value.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'
```

## Quick Start

```python
from main import HighSNRAgent

# Initialize the agent
agent = HighSNRAgent()

# Load mock data (or provide your own documents)
agent.load_data()

# Run a search
results = agent.search("Did Qian Chen at Meta sign a non-compete agreement?")

# Print results
for i, result in enumerate(results):
    print(f"\n[Result {i+1}]")
    print(f"Organization: {result.metadata.get('org')}")
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content[:200]}...")
```

## Running the Qian Chen Test

The canonical test case for entity disambiguation:

```bash
python main.py
```

Expected behavior:
- Query: "Did Qian Chen at Meta sign a non-compete agreement?"
- Should retrieve ONLY documents about Qian Chen (Meta Researcher)
- Should AVOID Qian Chen (Shanghai Lawyer) and Qian Chen (Student)

## Running Ablation Tests

To prove the value of each component:

## Running Ablation Tests / Benchmarks

To run the full evaluation suite:

```bash
# Run using the unified docker-compose profile
docker compose --profile benchmark up --build
```

This runs 4 tests:
1. **Baseline**: Full system
2. **No DSPy**: Raw LLM prompts instead of typed signatures
3. **No Circuit Breaker**: Fixed iteration count instead of novelty-based stopping
4. **No Memory**: Clears negative constraints each loop

## Configuration

Edit `config.py` to customize:

```python
# Feature flags (ablation testing)
USE_DSPY_SIGNATURES = True    # Use typed DSPy signatures
USE_NOVELTY_CIRCUIT = True    # Use mathematical circuit breaker
USE_NEGATIVE_MEMORY = True    # Use constraint learning
USE_CASCADE_RECALL = True     # Use tri-index cascade

# Retrieval parameters
VECTOR_TOP_K = 20
BM25_TOP_K = 20
FINAL_TOP_K = 5

# Circuit breaker
NOVELTY_EPSILON = 0.2  # 20% minimum novelty threshold
MAX_ITERATIONS = 5

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gpt-4o"
```

## Module Structure

```
agentic_search/
├── core/                      # Main Agent Logic
│   ├── orchestrator.py        # DSPy "Brain"
│   ├── backends.py            # Retrieval Interfaces
│   ├── graph_engine.py        # LangGraph State Machine
│   ├── verifier.py            # Novelty Circuit Breaker
│   ├── memory.py              # Reflection & Learning
│   └── config.py              # Configuration
│
├── legalbench/                # Evaluation & Benchmarks
│
├── scripts/                   # Utilities
│   ├── ingestion/             # Data loading
│   ├── analysis/              # Trace analysis
│   └── debug/                 # Debugging tools
│
├── docs/                      # Documentation
│   ├── architecture/          # System design
│   └── guides/                # Quickstarts
│
├── api.py                     # FastAPI entrypoint
└── main.py                    # CLI entrypoint
```

## Using Elasticsearch (Optional)

To use Elasticsearch instead of in-memory indexes:

1. Start Elasticsearch:
```bash
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.0.0
```

2. Enable in code:
```python
agent = HighSNRAgent(use_elasticsearch=True)
```

## Custom Data Ingestion

```python
from main import HighSNRAgent

# Your documents
documents = [
    {
        "id": "doc1",
        "content": "Document text here...",
        "metadata": {
            "org": "Company Name",
            "year": 2024,
            "type": "Legal Document",
            "entities": [
                {"id": "person1", "name": "John Doe", "type": "person"}
            ],
            "relations": [
                {"source": "person1", "target": "company1", "type": "employed_by"}
            ]
        }
    }
]

# Load your data
agent = HighSNRAgent()
agent.load_data(documents)

# Search
results = agent.search("Your query here")
```

## Extending the System

### Add a New Retrieval Index

```python
# In retrieval/ folder, create new_search.py
class NewSearchEngine:
    def search(self, query, top_k, filter_constraints):
        # Your implementation
        return results
```

### Add a New DSPy Signature

```python
# In orchestrator.py
class CustomSignature(dspy.Signature):
    input_field = dspy.InputField(desc="...")
    output_field = dspy.OutputField(desc="...")
```

### Modify the State Machine

Edit `graph_engine.py` to add new nodes or edges to the LangGraph.

## Performance Metrics

The system tracks:
- **Novelty Score**: Information gain per iteration
- **Precision**: % of relevant results
- **Iterations**: Number of search loops before convergence
- **Latency**: Time per search (with parallel map-reduce)

## Research Applications

This architecture is designed for PhD/research contexts:

1. **Counterfactual Analysis**: Feature flags enable controlled experiments
2. **Metrics**: All decisions are logged and measurable
3. **Modularity**: Each component can be swapped for comparison
4. **Reproducibility**: Deterministic circuit breaker ensures consistent behavior

## License

MIT

## Citation

If you use this system in research:

```bibtex
@software{high_snr_agentic_rag,
  title={High-SNR Agentic RAG: Deterministic Multi-Index Search with DSPy and LangGraph},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/high-snr-agentic-rag}
}
```

## Troubleshooting

### OpenAI API Key Error
```bash
export OPENAI_API_KEY='sk-...'
```

### Import Errors
```bash
pip install --upgrade dspy-ai langgraph langchain
```

### Elasticsearch Connection Failed
Check that ES is running: `curl http://localhost:9200`

## References

- [DSPy: Compiling Declarative Language Model Calls](https://github.com/stanfordnlp/dspy)
- [LangGraph: Multi-Agent Workflows](https://github.com/langchain-ai/langgraph)
- [Map-Reduce with Send() API](https://www.youtube.com/watch?v=5iYV0q6eKbM)
