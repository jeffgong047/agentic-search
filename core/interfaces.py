"""
Abstract Interfaces for Retrieval Components

This defines the contracts that ALL implementations must follow.
The agent only depends on these interfaces, not specific implementations.

This ensures the code works regardless of what database/retrieval system is used.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set


# ====================================================================
# INTERFACE 1: Document Store
# ====================================================================

class DocumentStore(ABC):
    """
    Abstract interface for document storage and retrieval.

    Implementations:
    - ElasticsearchStore (production)
    - InMemoryStore (development/testing)
    - CustomStore (user-provided)
    """

    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Index documents into the store.

        Args:
            documents: List of documents with keys: id, content, metadata

        Returns:
            Number of successfully indexed documents
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Dict[str, Any] | None = None,
        negative_constraints: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (must match)
            negative_constraints: Terms that must NOT appear

        Returns:
            List of dicts with keys: id, content, metadata, score
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Dict[str, Any] | None:
        """Get a document by ID"""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of documents"""
        pass


# ====================================================================
# INTERFACE 2: Vector Search Engine
# ====================================================================

class VectorSearchInterface(ABC):
    """
    Abstract interface for vector/semantic search.

    Implementations:
    - FAISSVectorSearch (built-in)
    - ElasticsearchVectorSearch (ES k-NN)
    - PineconeVectorSearch (external)
    - CustomVectorSearch (user-provided)
    """

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to vector index"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_constraints: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search.

        Returns:
            List of dicts with keys: id, content, score, metadata, source_index='vector'
        """
        pass

    @abstractmethod
    def get_index_size(self) -> int:
        """Return number of documents in index"""
        pass


# ====================================================================
# INTERFACE 3: Keyword Search Engine
# ====================================================================

class KeywordSearchInterface(ABC):
    """
    Abstract interface for keyword/lexical search.

    Implementations:
    - BM25Search (built-in)
    - ElasticsearchBM25 (ES BM25)
    - CustomKeywordSearch (user-provided)
    """

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to keyword index"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_constraints: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Keyword-based search.

        Returns:
            List of dicts with keys: id, content, score, metadata, source_index='bm25'
        """
        pass

    @abstractmethod
    def get_index_size(self) -> int:
        """Return number of documents in index"""
        pass


# ====================================================================
# INTERFACE 4: Knowledge Graph
# ====================================================================

class KnowledgeGraphInterface(ABC):
    """
    Abstract interface for knowledge graph operations.

    Implementations:
    - NetworkXGraph (built-in)
    - PostgreSQLGraph (relational)
    - Neo4jGraph (native graph)
    - CustomGraph (user-provided)
    """

    @abstractmethod
    def add_document(
        self,
        doc_id: str,
        content: str,
        entities: List[Dict[str, str]],
        relations: List[Dict[str, str]] | None = None
    ) -> None:
        """
        Add a document and its entities/relations to the graph.

        Args:
            doc_id: Document ID
            content: Document content
            entities: List of {"id": ..., "name": ..., "type": ...}
            relations: List of {"source": ..., "target": ..., "type": ...}
        """
        pass

    @abstractmethod
    def find_related_entities(
        self,
        entity_query: str,
        max_depth: int = 2,
        entity_types: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to query entity.

        Returns:
            List of dicts with keys: entity_id, distance, entity_data
        """
        pass

    @abstractmethod
    def get_entity_documents(self, entity_id: str) -> Set[str]:
        """Get all documents mentioning an entity"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        pass


# ====================================================================
# INTERFACE 5: Search Backend (High-Level)
# ====================================================================

class SearchBackend(ABC):
    """
    High-level search backend interface.

    This allows swapping entire search systems (not just components).

    Implementations:
    - TriIndexBackend (Vector + BM25 + Graph)
    - ElasticsearchOnlyBackend (ES-only)
    - CustomBackend (user-provided)
    """

    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        search_plan: Any,  # SearchPlan from data_structures
        negative_cache: List[Dict[str, str]] | None = None
    ) -> List[Any]:  # List[SearchResult]
        """
        Execute search based on search plan.

        Args:
            query: User query
            search_plan: SearchPlan with strategy, filters, etc.
            negative_cache: Failed paths to avoid

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        pass


# ====================================================================
# FACTORY PATTERN
# ====================================================================

class BackendFactory:
    """
    Factory for creating search backends.

    This is the magic that makes everything work regardless of infrastructure.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, backend_class: type):
        """Register a backend implementation"""
        cls._registry[name] = backend_class

    @classmethod
    def create(cls, backend_type: str, config: Dict[str, Any]) -> SearchBackend:
        """
        Create a search backend.

        Args:
            backend_type: "tri-index", "elasticsearch", "custom", etc.
            config: Backend-specific configuration

        Returns:
            SearchBackend instance

        Example:
            backend = BackendFactory.create(
                "tri-index",
                {"vector_model": "all-MiniLM-L6-v2"}
            )

            # OR with custom backend
            BackendFactory.register("mybackend", MyCustomBackend)
            backend = BackendFactory.create("mybackend", {...})
        """
        if backend_type not in cls._registry:
            raise ValueError(f"Unknown backend type: {backend_type}. "
                           f"Available: {list(cls._registry.keys())}")

        backend_class = cls._registry[backend_type]
        return backend_class(config)

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backends"""
        return list(cls._registry.keys())


# ====================================================================
# INTERFACE 6: External Tool Interface (Generic)
# ====================================================================

class ExternalTool(ABC):
    """
    Generic interface for external tools/services.

    This allows wrapping ANY external service the agent might need.
    """

    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """
        Connect to external service.

        Args:
            config: Connection configuration

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the external tool.

        Args:
            request: Tool-specific request

        Returns:
            Tool-specific response
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if tool is available"""
        pass


# ====================================================================
# HELPER: Auto-detect best available backend
# ====================================================================

def auto_detect_backend(
    es_config: Dict[str, Any] | None = None,
    kg_config: Dict[str, Any] | None = None
) -> str:
    """
    Auto-detect the best available backend based on what's configured.

    Args:
        es_config: Elasticsearch configuration
        kg_config: Knowledge graph configuration

    Returns:
        Backend type name

    Example:
        backend_type = auto_detect_backend(es_config={"host": "..."})
        # Returns: "elasticsearch" if ES is available, else "tri-index"
    """
    # Try Elasticsearch
    if es_config:
        try:
            from elasticsearch import Elasticsearch
            es = Elasticsearch([f"{es_config.get('host', 'localhost')}:{es_config.get('port', 9200)}"])
            if es.ping():
                return "elasticsearch"
        except Exception:
            pass

    # Fallback to tri-index (always available)
    return "tri-index"


# ====================================================================
# USAGE EXAMPLE
# ====================================================================

"""
Example of how this ensures portability:

# In the agent code:
class RetrievalAgent:
    def __init__(self, backend: SearchBackend):
        self.backend = backend  # ← Depends on interface, not implementation

    def search(self, query):
        return self.backend.search(query, search_plan, negative_cache)


# User with Elasticsearch:
backend = BackendFactory.create("elasticsearch", {
    "host": "es.company.com",
    "index": "docs"
})
agent = RetrievalAgent(backend)

# User with tri-index:
backend = BackendFactory.create("tri-index", {})
agent = RetrievalAgent(backend)

# User with custom system:
class MyCustomBackend(SearchBackend):
    def search(self, query, search_plan, negative_cache):
        # Call my custom system
        return results

BackendFactory.register("mybackend", MyCustomBackend)
backend = BackendFactory.create("mybackend", {...})
agent = RetrievalAgent(backend)

# Agent code doesn't change! It just calls backend.search()
"""
