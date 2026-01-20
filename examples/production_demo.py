"""
Production Demo: Industry Meta Stack

DSPy (Brain) + LlamaIndex (Plumbing) + Elasticsearch (Backend)

This shows:
1. How to use LlamaIndex for rapid prototyping
2. How to swap to raw ES for production
3. How DSPy orchestrates the search
4. Why Elasticsearch is sufficient (no FAISS needed)
"""

import os
from typing import List, Dict, Any

# The Brain: DSPy
from orchestrator import LegalOrchestrator

# Data Plumbing: LlamaIndex (swappable)
from llama_index_ingestion import LlamaIndexPipeline
from raw_es_ingestion import RawElasticsearchPipeline

# Backend: Elasticsearch (does everything)
from elasticsearch_hybrid_backend import ElasticsearchHybridBackend

# Interfaces
from production_interfaces import SearchQuery, SearchResult


class ProductionRetrievalAgent:
    """
    Production agent using Industry Meta stack.

    Components:
    - DSPy: Intent classification, query decomposition
    - LlamaIndex/RawES: Document chunking (swappable)
    - Elasticsearch: Vector + BM25 + RRF (single system)
    """

    def __init__(
        self,
        use_llamaindex: bool = True,
        es_url: str = "http://localhost:9200",
        es_index: str = "legal_documents"
    ):
        """
        Initialize production agent.

        Args:
            use_llamaindex: True = LlamaIndex (prototyping), False = Raw ES (production)
            es_url: Elasticsearch URL
            es_index: ES index name
        """
        # The Brain: DSPy
        self.orchestrator = LegalOrchestrator()

        # Data Plumbing: LlamaIndex or Raw ES (swappable!)
        if use_llamaindex:
            self.ingestion = LlamaIndexPipeline()
            print("[Agent] Using LlamaIndex for data plumbing")
        else:
            self.ingestion = RawElasticsearchPipeline()
            print("[Agent] Using Raw ES for data plumbing")

        # Backend: Elasticsearch (does everything)
        self.backend = ElasticsearchHybridBackend(
            es_url=es_url,
            index_name=es_index
        )

        print("[Agent] Production agent initialized")
        print(f"  DSPy: ✓ (intent classification)")
        print(f"  Ingestion: ✓ ({'LlamaIndex' if use_llamaindex else 'Raw ES'})")
        print(f"  Backend: ✓ (Elasticsearch hybrid)")

    def ingest_documents(self, documents: List[Dict[str, Any]]):
        """
        Ingest documents through the pipeline.

        Shows the swappable boundary:
        - LlamaIndex: Rapid prototyping
        - Raw ES: Production deployment
        """
        print(f"\n[Ingest] Processing {len(documents)} documents...")

        # Step 1: Chunk
        chunks = self.ingestion.chunk_documents(documents)

        # Step 2: Extract metadata
        for chunk in chunks:
            enhanced_metadata = self.ingestion.extract_metadata(chunk)
            chunk["metadata"].update(enhanced_metadata)

        # Step 3: Index to ES
        backend_config = {
            "es_url": self.backend.es.transport.hosts[0],
            "index_name": self.backend.index_name
        }
        indexed = self.ingestion.index_to_backend(chunks, backend_config)

        print(f"[Ingest] ✓ Indexed {indexed} chunks")

    def search(self, query: str) -> List[SearchResult]:
        """
        Execute search using DSPy + ES.

        Flow:
        1. DSPy: Classify intent, decompose query, apply HyDE
        2. ES: Execute hybrid search (vector + BM25 + RRF)
        3. Return: Ranked, fused results
        """
        print(f"\n[Search] Query: {query}")

        # DSPy: Generate search strategy
        # This returns SearchPlan with intent, filters, negative constraints
        from data_structures import create_initial_state
        state = create_initial_state(query)
        search_plan = self.orchestrator.forward(state)

        print(f"[DSPy] Intent: {search_plan.primary_intent}")
        print(f"[DSPy] Filters: {search_plan.filter_constraints}")
        print(f"[DSPy] Negative: {search_plan.negative_constraints}")

        # Convert SearchPlan to SearchQuery
        search_query = SearchQuery(
            text=query,
            vector_query=search_plan.hyde_passage,  # HyDE-expanded for vector search
            keyword_query=search_plan.search_queries[0] if search_plan.search_queries else query,
            filters=search_plan.filter_constraints,
            negative_constraints=search_plan.negative_constraints,
            top_k=5
        )

        # ES: Execute hybrid search
        results = self.backend.hybrid_search(search_query)

        print(f"[ES] Retrieved {len(results)} results")

        return results


# ====================================================================
# DEMO 1: LlamaIndex for Prototyping
# ====================================================================

def demo_with_llamaindex():
    """
    Demo showing LlamaIndex for rapid prototyping.
    """
    print("\n" + "="*70)
    print(" DEMO 1: LlamaIndex (Rapid Prototyping)".center(70))
    print("="*70)

    # Sample legal documents
    documents = [
        {
            "id": "meta_qian_chen_001",
            "content": """
            Employment Agreement - Meta Platforms Inc.
            Employee: Qian Chen
            Position: Senior Research Scientist
            Department: AI Research Lab
            Start Date: January 15, 2023

            Non-Compete Clause: Employee agrees not to engage in competitive
            activities with Meta for 12 months following termination. This
            agreement is governed by California law, which generally prohibits
            non-compete agreements except in limited circumstances.
            """,
            "metadata": {
                "org": "Meta",
                "year": 2023,
                "doc_type": "employment_agreement"
            }
        },
        {
            "id": "shanghai_qian_chen_001",
            "content": """
            Legal Opinion - Shanghai Financial District Law Firm
            Attorney: Qian Chen
            Practice Area: Corporate Finance & Securities Law

            Re: Non-Compete Enforcement in Chinese Employment Law

            In China, non-compete agreements are generally enforceable if they
            meet statutory requirements. This differs from California law.
            """,
            "metadata": {
                "org": "Shanghai Law Firm",
                "year": 2024,
                "doc_type": "legal_opinion",
                "location": "Shanghai"
            }
        }
    ]

    # Initialize agent with LlamaIndex
    agent = ProductionRetrievalAgent(use_llamaindex=True)

    # Ingest documents
    agent.ingest_documents(documents)

    # Search
    query = "Did Qian Chen at Meta sign a non-compete agreement?"
    results = agent.search(query)

    # Display results
    print(f"\n{'='*70}")
    print(" RESULTS".center(70))
    print("="*70)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.3f} | Org: {result.metadata.get('org', 'N/A')}")
        print(f"    {result.content[:150]}...")


# ====================================================================
# DEMO 2: Raw ES for Production
# ====================================================================

def demo_with_raw_es():
    """
    Demo showing raw ES for production deployment.
    """
    print("\n" + "="*70)
    print(" DEMO 2: Raw Elasticsearch (Production)".center(70))
    print("="*70)

    documents = [
        {
            "id": "meta_doc_001",
            "content": "Meta Platforms employment agreement for Qian Chen...",
            "metadata": {"org": "Meta", "year": 2024}
        }
    ]

    # Initialize agent with Raw ES
    agent = ProductionRetrievalAgent(use_llamaindex=False)

    # Ingest documents
    agent.ingest_documents(documents)

    # Search
    results = agent.search("Qian Chen Meta")

    print(f"\n✓ Retrieved {len(results)} results using Raw ES pipeline")


# ====================================================================
# DEMO 3: Swappability
# ====================================================================

def demo_swappability():
    """
    Show how easy it is to swap ingestion pipelines.
    """
    print("\n" + "="*70)
    print(" DEMO 3: Swappability Test".center(70))
    print("="*70)

    documents = [{"id": "test", "content": "Test document", "metadata": {}}]

    # Version 1: LlamaIndex
    print("\n[Version 1] With LlamaIndex:")
    agent_v1 = ProductionRetrievalAgent(use_llamaindex=True)
    agent_v1.ingest_documents(documents)

    # Version 2: Raw ES
    print("\n[Version 2] With Raw ES:")
    agent_v2 = ProductionRetrievalAgent(use_llamaindex=False)
    agent_v2.ingest_documents(documents)

    print("\n✓ Both versions work!")
    print("✓ Agent code didn't change")
    print("✓ Only swapped ingestion pipeline")
    print("\n→ This is what you show your mentor:")
    print("  'I used LlamaIndex to prove the concept,")
    print("   but here's how we swap to your ES infrastructure.'")


# ====================================================================
# DEMO 4: Why No FAISS Needed
# ====================================================================

def demo_why_no_faiss():
    """
    Explain why Elasticsearch is sufficient.
    """
    print("\n" + "="*70)
    print(" WHY NO FAISS? Elasticsearch Does It All".center(70))
    print("="*70)

    print("""
┌──────────────────────────────────────────────────────────────┐
│                  TRADITIONAL APPROACH (Complex)              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  FAISS                    BM25                  Elasticsearch│
│  (Vector Search)          (Keyword)             (Metadata)  │
│      │                       │                      │       │
│      └───────────────────────┼──────────────────────┘       │
│                              │                              │
│                    Manual RRF Fusion                        │
│                    Sync Issues                              │
│                    Complex Deployment                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  INDUSTRY META APPROACH (Simple)             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                   ELASTICSEARCH                              │
│                                                              │
│   ✓ Vector Search (k-NN/HNSW)                               │
│   ✓ BM25 Keyword Search                                     │
│   ✓ Metadata Filtering                                      │
│   ✓ Native RRF Fusion                                       │
│   ✓ Single API Call                                         │
│   ✓ No Sync Issues                                          │
│   ✓ Production Proven                                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘

KEY INSIGHT:
    Elasticsearch 8.0+ has built-in k-NN using the same HNSW algorithm
    that FAISS uses. Adding FAISS creates redundancy, not value.

WHEN TO USE FAISS:
    - Research on specialized vector algorithms (IVF, PQ)
    - Ultra-fast local testing without Docker
    - Benchmarking specific vector indexing methods

WHEN TO USE ELASTICSEARCH (Your Case):
    - Production deployment ✓
    - Need metadata filtering ✓
    - Need keyword + vector fusion ✓
    - Mentor already has ES ✓
    - "Industry Meta" architecture ✓
    """)


# ====================================================================
# MAIN
# ====================================================================

def main():
    """Run production demos"""
    import sys

    # Check API key
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️  Set OPENAI_API_KEY environment variable")
        print("Usage: export OPENAI_API_KEY='your-key'")
        return

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "llamaindex" or mode == "all":
        demo_with_llamaindex()

    if mode == "raw" or mode == "all":
        demo_with_raw_es()

    if mode == "swap" or mode == "all":
        demo_swappability()

    if mode == "why" or mode == "all":
        demo_why_no_faiss()

    print("\n" + "="*70)
    print(" PRODUCTION DEMOS COMPLETE".center(70))
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. DSPy = The Brain (intent classification)")
    print("  2. LlamaIndex/Raw ES = Swappable data plumbing")
    print("  3. Elasticsearch = Single backend (vector + BM25 + RRF)")
    print("  4. NO FAISS needed - ES does it all!")
    print("\n→ Show your mentor: production_demo.py")


if __name__ == "__main__":
    main()
