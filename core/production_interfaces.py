"""
Production Interfaces: Industry Meta Design

Key Principles:
1. DSPy = The Brain (logic, intent classification)
2. LlamaIndex = Data Plumbing (swappable)
3. Elasticsearch = Single Backend (vector + BM25 + metadata)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


# ====================================================================
# INTERFACE 1: Document Ingestion (The "Plumbing")
# ====================================================================

class DocumentIngestionPipeline(ABC):
    """
    Abstract interface for document chunking and indexing.

    Implementations:
    - LlamaIndexPipeline (rapid prototyping)
    - RawElasticsearchPipeline (production, mentor's infrastructure)
    """

    @abstractmethod
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents into smaller pieces.

        Args:
            documents: Raw documents
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunks with metadata
        """
        pass

    @abstractmethod
    def extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from document.

        In production, this might call an NER service or LLM.
        """
        pass

    @abstractmethod
    def index_to_backend(
        self,
        chunks: List[Dict[str, Any]],
        backend_config: Dict[str, Any]
    ) -> int:
        """
        Index chunks to backend (Elasticsearch).

        Args:
            chunks: Preprocessed chunks
            backend_config: ES connection details

        Returns:
            Number of successfully indexed chunks
        """
        pass


# ====================================================================
# INTERFACE 2: Hybrid Retrieval (The "Search")
# ====================================================================

@dataclass
class SearchQuery:
    """
    Structured search query from DSPy orchestrator.
    """
    text: str  # Original query
    vector_query: str  # For semantic search (might be HyDE-expanded)
    keyword_query: str  # For BM25 search
    filters: Dict[str, Any]  # Metadata filters (org, jurisdiction, etc.)
    negative_constraints: List[str]  # Terms to exclude
    top_k: int = 10


@dataclass
class SearchResult:
    """Standard search result"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # "vector", "bm25", "hybrid"


class HybridSearchBackend(ABC):
    """
    Abstract interface for hybrid search.

    Elasticsearch implements this natively (vector + BM25 + RRF).
    This interface allows swapping ES for other systems later.
    """

    @abstractmethod
    def hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Execute hybrid search (vector + BM25 + metadata filtering).

        Elasticsearch does this in ONE API call using:
        - k-NN for vector search
        - BM25 for keyword search
        - RRF (Reciprocal Rank Fusion) for combining scores
        - bool query for metadata filtering

        Args:
            query: Structured search query

        Returns:
            Ranked search results
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        pass


# ====================================================================
# WHY THIS DESIGN WINS
# ====================================================================

"""
COMPARISON: What We're AVOIDING

❌ WRONG APPROACH (Complex, Fragile):
┌─────────┐  ┌─────────┐  ┌─────────┐
│  FAISS  │  │  BM25   │  │   ES    │
│ Vector  │  │Keyword  │  │Metadata │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┼────────────┘
                  │
          Manual RRF Fusion
          Manual Sync Issues
          3x the infrastructure


✅ RIGHT APPROACH (Simple, Production):
┌──────────────────────────────────┐
│      Elasticsearch               │
│  - k-NN vector search            │
│  - BM25 keyword search           │
│  - Metadata filtering            │
│  - Built-in RRF fusion           │
│  - Single API call               │
└──────────────────────────────────┘


PRODUCTION BENEFITS:

1. Day 1 Integration: Mentor already has ES
2. No Sync Issues: Everything in one system
3. Native RRF: ES combines vector + BM25 automatically
4. Metadata Filtering: Can't do this easily with FAISS
5. Scalability: ES proven at EvenUp scale
"""


# ====================================================================
# INTERFACE 3: The Complete Stack
# ====================================================================

class ProductionRAGStack:
    """
    The complete stack showing interface boundaries.
    """

    def __init__(
        self,
        ingestion_pipeline: DocumentIngestionPipeline,
        search_backend: HybridSearchBackend,
        orchestrator: Any  # DSPy orchestrator
    ):
        """
        Initialize the stack.

        Args:
            ingestion_pipeline: LlamaIndex or Raw ES
            search_backend: Elasticsearch
            orchestrator: DSPy brain
        """
        self.ingestion = ingestion_pipeline
        self.backend = search_backend
        self.orchestrator = orchestrator

    def ingest_documents(self, documents: List[Dict[str, Any]]):
        """
        Ingest documents through the pipeline.

        This shows the swappable boundary:
        - Start with LlamaIndex for rapid prototyping
        - Swap to raw ES when mentor needs it
        """
        # Step 1: Chunk (LlamaIndex does this well)
        chunks = self.ingestion.chunk_documents(documents)

        # Step 2: Extract metadata (LlamaIndex or custom)
        for chunk in chunks:
            chunk["metadata"] = self.ingestion.extract_metadata(chunk)

        # Step 3: Index to ES (via LlamaIndex or direct)
        self.ingestion.index_to_backend(chunks, {})

    def search(self, query: str) -> List[SearchResult]:
        """
        Search using DSPy orchestrator + ES backend.

        DSPy decides:
        - What vector query to use (HyDE expansion)
        - What keyword query to use
        - What filters to apply

        ES executes:
        - Vector search (k-NN)
        - Keyword search (BM25)
        - Combines with RRF
        - Filters by metadata
        """
        # DSPy: Generate search strategy
        search_query = self.orchestrator.generate_query(query)

        # ES: Execute hybrid search
        results = self.backend.hybrid_search(search_query)

        return results


# ====================================================================
# SWAPPABILITY DEMONSTRATION
# ====================================================================

class SwappableIngestionDemo:
    """
    Shows how to swap LlamaIndex → Raw ES without changing agent code.
    """

    @staticmethod
    def with_llamaindex():
        """Option 1: Use LlamaIndex for rapid prototyping"""
        from llama_index_ingestion import LlamaIndexPipeline

        pipeline = LlamaIndexPipeline()
        backend = ElasticsearchBackend()
        orchestrator = DSPyOrchestrator()

        stack = ProductionRAGStack(pipeline, backend, orchestrator)
        return stack

    @staticmethod
    def with_raw_elasticsearch():
        """Option 2: Use raw ES for production (mentor's preference)"""
        from raw_es_ingestion import RawElasticsearchPipeline

        pipeline = RawElasticsearchPipeline()
        backend = ElasticsearchBackend()  # Same backend!
        orchestrator = DSPyOrchestrator()  # Same orchestrator!

        stack = ProductionRAGStack(pipeline, backend, orchestrator)
        return stack

    @staticmethod
    def demo():
        """
        Show mentor: "Here's LlamaIndex for prototyping, but we can
        swap to your infrastructure with zero changes to the agent."
        """
        # Start with LlamaIndex
        stack_v1 = SwappableIngestionDemo.with_llamaindex()
        results_v1 = stack_v1.search("Qian Chen Meta")

        # Swap to raw ES (mentor's system)
        stack_v2 = SwappableIngestionDemo.with_raw_elasticsearch()
        results_v2 = stack_v2.search("Qian Chen Meta")

        # Agent code didn't change!
        # Only swapped the ingestion pipeline
        assert len(results_v1) > 0
        assert len(results_v2) > 0


# ====================================================================
# THE PITCH TO YOUR MENTOR
# ====================================================================

"""
MENTOR CONVERSATION:

You: "I used LlamaIndex to rapidly prove the concept and get chunks working.
     But the core retrieval logic is clean Python interfaces.
     Here's how we plug in your Elasticsearch infrastructure."

Mentor: "We already have ES with k-NN and our own chunking pipeline."

You: "Perfect! I just swap this:"

    # Prototype
    pipeline = LlamaIndexPipeline()

    # Production
    pipeline = YourElasticsearchPipeline()

"The agent code doesn't change. DSPy still does the intent classification.
ES still does the hybrid search. We just changed the data plumbing."

Mentor: "And the knowledge graph?"

You: "ES nested documents for simple relationships, or we can query your
PostgreSQL graph via the same interface."

Mentor: 🤝 "Ship it."
"""
