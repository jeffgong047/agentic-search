"""
Configuration and Feature Flags for High-SNR Agentic RAG
Enables counterfactual ablation testing
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class SystemConfig:
    """Global configuration with feature toggles for ablation testing"""

    # === FEATURE FLAGS (Ablation Toggles) ===
    USE_DSPY_SIGNATURES: bool = True
    """If False: Replace DSPy with raw LLM calls (proves value of typed schemas)"""

    USE_NOVELTY_CIRCUIT: bool = True
    """If False: Run for fixed N=5 steps (proves value of circuit breaker)"""

    USE_NEGATIVE_MEMORY: bool = True
    """If False: Clear constraints each loop (proves value of memory evolution)"""

    USE_CASCADE_RECALL: bool = True
    """If False: Use only vector search (proves value of tri-index)"""

    # === RETRIEVAL PARAMETERS ===
    VECTOR_TOP_K: int = 20
    BM25_TOP_K: int = 20
    GRAPH_MAX_DEPTH: int = 2
    FINAL_TOP_K: int = 5

    # === CIRCUIT BREAKER PARAMETERS ===
    NOVELTY_EPSILON: float = 0.2
    """Minimum novelty threshold (20%) to continue looping"""

    MAX_ITERATIONS: int = 5
    """Hard limit on search iterations"""

    # === ELASTICSEARCH CONFIG ===
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    ES_INDEX_NAME: str = "legal_docs"

    # === MODEL CONFIG ===
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL: str = "claude-3-opus-20240229"
    EVAL_MODEL: str = "claude-3-haiku-20240307"  # Default to Haiku for evaluation
    
    # Model Mappings for convenience
    MODEL_OPUS = "claude-3-opus-20240229"
    MODEL_SONNET = "claude-3-5-sonnet-20240620"
    MODEL_HAIKU = "claude-3-haiku-20240307"

    # === LOGGING ===
    DEBUG_MODE: bool = True
    LOG_RETRIEVAL_STEPS: bool = True


# Global config instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs) -> None:
    """Update configuration parameters"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
