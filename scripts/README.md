# Agentic Search Scripts

This directory contains utility scripts for the Agentic Search project, organized by function.

## Directory Structure

### `ingestion/`
Scripts handling document processing and indexing.
- `load_legal_corpus.py`: Main ingestion script for legal documents (ES + PostgreSQL).
- `raw_es_ingestion.py`: Production-grade ingestion using raw Elasticsearch calls (No LlamaIndex).
- `llama_index_ingestion.py`: Prototype ingestion using LlamaIndex abstractions.

### `analysis/`
Tools for evaluating agent performance and behavior.
- `legalbench_analysis.py`: Standalone evaluation runner (Memory Phase evaluation).
- `analyze_traces.py`: Analyzes `jsonl` trace logs from DSPy runs.
- `analyze_agent_traces.py`: Visualizes quantitative results (latency, retrieval counts) from `results.json`.
- `audit_metric_validity.py`: Deep dive audit tool for verifying metric calculations.
- `run_es_baseline.py`: Baseline evaluation runner (Search sans Agent).

### `debug/`
Utilities for debugging and testing.
- `debug_trace.py`: Visualizes DSPy Chain-of-Thought traces in terminal.
- `debug_es.py`: Tests Elasticsearch connectivity.
- `check_model_access.py`: Verifies LLM API access.

## Usage

When running scripts from this directory, they are patched to find the project root `agentic_search` (via `sys.path.append`).

Example:
```bash
# Run ingestion
python scripts/ingestion/load_legal_corpus.py

# Run analysis
python scripts/analysis/analyze_traces.py
```
