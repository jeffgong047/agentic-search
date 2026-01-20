"""
LegalBench Evaluation Module

This module provides tools for evaluating agentic search performance using
the LegalBench benchmark (162 legal reasoning tasks from arxiv:2308.11462).

Components:
- download_tasks: Download/generate LegalBench tasks
- load_corpus: Load or generate legal document corpus
- create_qrels: Generate relevance judgments
- build_index: Build semantic index using LlamaIndex
- run_evaluation: Execute full evaluation and compute metrics
- metrics: Track retrieval, latency, and circuit breaker metrics
- report: Generate evaluation reports and visualizations

Usage:
    # Download data
    python -m legalbench download --output legalbench/data

    # Build index
    python -m legalbench build-index --corpus legalbench/data/corpus.jsonl

    # Run evaluation
    python -m legalbench evaluate --tasks all --output legalbench/results
"""

__version__ = "0.1.0"
__author__ = "Agentic Search Team"

from .download_tasks import LegalBenchDownloader
from .load_corpus import LegalCorpusManager
from .create_qrels import QrelsGenerator
from .build_index import LegalBenchIndexBuilder

__all__ = [
    "LegalBenchDownloader",
    "LegalCorpusManager",
    "QrelsGenerator",
    "LegalBenchIndexBuilder",
]
