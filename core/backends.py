"""
Concrete Backend Implementations

This file contains all the concrete implementations of the interfaces.
Each backend is self-contained and can be swapped independently.
"""

from typing import List, Dict, Any, Set
from .interfaces import (
    SearchBackend,
    VectorSearchInterface,
    KeywordSearchInterface,
    KnowledgeGraphInterface,
    BackendFactory
)
from .data_structures import SearchResult, SearchPlan
from retrieval import VectorSearchEngine, BM25SearchEngine, GraphSearchEngine, CascadeRecallFunnel
from indexing import KnowledgeGraphBuilder
from .config import get_config


# ====================================================================
# BACKEND 1: Tri-Index Backend (Built-in, Always Works)
# ====================================================================

class TriIndexBackend(SearchBackend):
    """
    Built-in tri-index backend using FAISS + BM25 + NetworkX.

    This ALWAYS works and requires no external dependencies.
    Perfect for development, testing, and when no other backend is available.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tri-index backend.

        Args:
            config: Configuration dict (optional parameters)
        """
        self.config = config
        self.vector_engine = VectorSearchEngine(
            embedding_model=config.get("vector_model")
        )
        self.bm25_engine = BM25SearchEngine()
        self.graph_engine = GraphSearchEngine()
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.cascade = CascadeRecallFunnel(
            reranker_model=config.get("reranker_model")
        )

        print("[Backend] Tri-Index initialized (FAISS + BM25 + NetworkX)")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents into all three indexes"""
        # Vector index
        self.vector_engine.add_documents(documents)

        # BM25 index
        self.bm25_engine.add_documents(documents)

        # Knowledge graph
        for doc in documents:
            entities = doc.get("metadata", {}).get("entities", [])
            relations = doc.get("metadata", {}).get("relations", [])
            self.knowledge_graph.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                entities=entities,
                relations=relations
            )

        # Update graph engine
        self.graph_engine.graph = self.knowledge_graph.graph
        self.graph_engine.entity_to_docs = self.knowledge_graph.entity_to_docs
        self.graph_engine.doc_store = {
            doc["id"]: {
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            }
            for doc in documents
        }

    def search(
        self,
        query: str,
        search_plan: SearchPlan,
        negative_cache: List[Dict[str, str]] | None = None
    ) -> List[SearchResult]:
        """Execute tri-index search"""
        # Search vector index
        vector_results = self.vector_engine.search(
            query=search_plan.hyde_passage,
            filter_constraints=search_plan.filter_constraints
        )

        # Search BM25 index
        lexical_query = search_plan.search_queries[0] if search_plan.search_queries else query
        bm25_results = self.bm25_engine.search(
            query=lexical_query,
            filter_constraints=search_plan.filter_constraints
        )

        # Search graph
        relational_query = search_plan.search_queries[2] if len(search_plan.search_queries) > 2 else query
        query_entities = self._extract_entities(relational_query)
        graph_results = self.graph_engine.search(
            query_entities=query_entities,
            depth=search_plan.entity_graph_depth,
            filter_constraints=search_plan.filter_constraints
        )

        # Aggregate and filter
        filtered = self.cascade.aggregate_and_filter(
            vector_results,
            bm25_results,
            graph_results,
            search_plan,
            negative_cache
        )

        # Rerank
        reranked = self.cascade.rerank(
            filtered,
            query,
            search_plan.hyde_passage
        )

        return reranked

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "backend_type": "tri-index",
            "vector_size": self.vector_engine.get_index_size(),
            "bm25_size": self.bm25_engine.get_index_size(),
            "graph_stats": self.knowledge_graph.get_graph_stats()
        }

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction"""
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return words if words else [text]


# ====================================================================
# BACKEND 2: Elasticsearch-Only Backend
# ====================================================================

class ElasticsearchBackend(SearchBackend):
    """
    Pure Elasticsearch backend.

    Uses ES for everything: keyword search, vector search (k-NN), and metadata.
    No external knowledge graph.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Elasticsearch backend.

        Args:
            config: Must contain "host", "port", "index"
        """
        from indexing import ElasticsearchManager

        self.config = config
        self.es_manager = ElasticsearchManager(
            host=config.get("host", "localhost"),
            port=config.get("port", 9200),
            index_name=config.get("index", "documents")
        )

        # Connect
        if not self.es_manager.connect():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {config['host']}")

        # Create index
        self.es_manager.create_index()

        print(f"[Backend] Elasticsearch initialized ({config['host']}:{config['port']})")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents into Elasticsearch"""
        self.es_manager.index_documents(documents)

    def search(
        self,
        query: str,
        search_plan: SearchPlan,
        negative_cache: List[Dict[str, str]] | None = None
    ) -> List[SearchResult]:
        """Search using Elasticsearch"""
        # Build negative constraints
        negative_terms = search_plan.negative_constraints.copy()
        if negative_cache:
            for item in negative_cache:
                entity = item.get("entity", "")
                if entity:
                    negative_terms.append(entity)

        # Search ES
        results = self.es_manager.search(
            query=query,
            top_k=get_config().FINAL_TOP_K,
            filter_constraints=search_plan.filter_constraints,
            negative_constraints=negative_terms
        )

        # Convert to SearchResult objects
        search_results = [
            SearchResult(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"],
                source_index="elasticsearch"
            )
            for r in results
        ]

        return search_results

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "backend_type": "elasticsearch",
            "document_count": self.es_manager.get_document_count(),
            "host": self.config["host"],
            "index": self.config["index"]
        }


# ====================================================================
# BACKEND 3: Hybrid Backend (ES + NetworkX Graph)
# ====================================================================

class HybridBackend(SearchBackend):
    """
    Hybrid backend: Elasticsearch for search + NetworkX for graph.

    Good balance between production scalability and simplicity.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hybrid backend.

        Args:
            config: Must contain ES config and optionally graph config
        """
        from indexing import ElasticsearchManager

        # Initialize ES
        self.es_manager = ElasticsearchManager(
            host=config.get("host", "localhost"),
            port=config.get("port", 9200),
            index_name=config.get("index", "documents")
        )
        self.es_manager.connect()
        self.es_manager.create_index()

        # Initialize graph
        self.graph_engine = GraphSearchEngine()
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.cascade = CascadeRecallFunnel()

        print(f"[Backend] Hybrid initialized (ES + NetworkX)")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index into both ES and graph"""
        # Index to ES
        self.es_manager.index_documents(documents)

        # Index to graph
        for doc in documents:
            entities = doc.get("metadata", {}).get("entities", [])
            relations = doc.get("metadata", {}).get("relations", [])
            self.knowledge_graph.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                entities=entities,
                relations=relations
            )

        # Update graph engine
        self.graph_engine.graph = self.knowledge_graph.graph
        self.graph_engine.entity_to_docs = self.knowledge_graph.entity_to_docs
        self.graph_engine.doc_store = {
            doc["id"]: {
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            }
            for doc in documents
        }

    def search(
        self,
        query: str,
        search_plan: SearchPlan,
        negative_cache: List[Dict[str, str]] | None = None
    ) -> List[SearchResult]:
        """Search using ES + graph"""
        # Search ES (combines vector + BM25)
        negative_terms = search_plan.negative_constraints.copy()
        if negative_cache:
            for item in negative_cache:
                if item.get("entity"):
                    negative_terms.append(item["entity"])

        es_results = self.es_manager.search(
            query=query,
            top_k=20,
            filter_constraints=search_plan.filter_constraints,
            negative_constraints=negative_terms
        )

        # Search graph
        query_entities = self._extract_entities(query)
        graph_results = self.graph_engine.search(
            query_entities=query_entities,
            depth=search_plan.entity_graph_depth,
            filter_constraints=search_plan.filter_constraints
        )

        # Convert ES results to SearchResult
        es_search_results = [
            SearchResult(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"],
                source_index="elasticsearch"
            )
            for r in es_results
        ]

        # Combine and rerank
        all_results = es_search_results + graph_results

        # Simple dedup by ID
        seen = {}
        for r in all_results:
            if r.id not in seen or r.score > seen[r.id].score:
                seen[r.id] = r

        final = list(seen.values())
        final.sort(key=lambda x: x.score, reverse=True)

        return final[:get_config().FINAL_TOP_K]

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "backend_type": "hybrid",
            "es_document_count": self.es_manager.get_document_count(),
            "graph_stats": self.knowledge_graph.get_graph_stats()
        }

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction"""
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return words if words else [text]


# ====================================================================
# Register Backends
# ====================================================================

# Register all built-in backends
BackendFactory.register("tri-index", TriIndexBackend)
BackendFactory.register("elasticsearch", ElasticsearchBackend)
BackendFactory.register("hybrid", HybridBackend)


# ====================================================================
# Helper: Auto-select best backend
# ====================================================================

def create_best_backend(
    es_config: Dict[str, Any] | None = None,
    kg_config: Dict[str, Any] | None = None,
    backend_type: str | None = None
) -> SearchBackend:
    """
    Automatically select and create the best backend.

    Args:
        es_config: Elasticsearch configuration
        kg_config: Knowledge graph configuration
        backend_type: Force specific backend (overrides auto-detection)

    Returns:
        SearchBackend instance

    Example:
        # Auto-detect
        backend = create_best_backend(es_config={...})

        # Force specific
        backend = create_best_backend(backend_type="tri-index")
    """
    # If user specified backend type, use that
    if backend_type:
        if backend_type == "elasticsearch" and es_config:
            return BackendFactory.create("elasticsearch", es_config)
        elif backend_type == "hybrid" and es_config:
            return BackendFactory.create("hybrid", es_config)
        else:
            return BackendFactory.create(backend_type, {})

    # Auto-detect based on what's configured
    if es_config:
        try:
            # Try hybrid backend
            return BackendFactory.create("hybrid", es_config)
        except Exception as e:
            print(f"[Backend] Could not create hybrid backend: {e}")
            print("[Backend] Falling back to tri-index")

    # Default: tri-index (always works)
    return BackendFactory.create("tri-index", {})


if __name__ == "__main__":
    # Test backend factory
    print("Available backends:", BackendFactory.list_backends())

    # Test tri-index
    backend = BackendFactory.create("tri-index", {})
    print(f"\nCreated: {backend.get_stats()}")
