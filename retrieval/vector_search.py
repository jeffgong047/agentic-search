"""
Vector Search Engine: Semantic retrieval using embeddings
"""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from data_structures import SearchResult
from config import get_config


class VectorSearchEngine:
    """
    Dense retrieval using sentence embeddings and FAISS.
    Handles HyDE (Hypothetical Document Embeddings) for query expansion.
    """

    def __init__(self, embedding_model: str | None = None):
        """
        Initialize the vector search engine.

        Args:
            embedding_model: Sentence transformer model name (uses config default if None)
        """
        config = get_config()
        model_name = embedding_model or config.EMBEDDING_MODEL

        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # FAISS index (will be populated by ingestion)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.doc_store: Dict[int, Dict[str, Any]] = {}  # idx -> document metadata

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector index.

        Args:
            documents: List of dicts with keys: id, content, metadata
        """
        if not documents:
            return

        # Extract text content
        texts = [doc["content"] for doc in documents]

        # Encode to embeddings
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Add to FAISS index
        start_idx = len(self.doc_store)
        self.index.add(embeddings)

        # Store document metadata
        for i, doc in enumerate(documents):
            self.doc_store[start_idx + i] = {
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            }

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_constraints: Dict[str, Any] | None = None
    ) -> List[SearchResult]:
        """
        Perform semantic vector search.

        Args:
            query: Search query (can be raw query or HyDE passage)
            top_k: Number of results to return
            filter_constraints: Metadata filters to apply

        Returns:
            List of SearchResult objects
        """
        config = get_config()
        top_k = top_k or config.VECTOR_TOP_K

        if self.index.ntotal == 0:
            return []

        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)

        # Search FAISS index (retrieve more than top_k for filtering)
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k=search_k)

        # Convert to SearchResult objects
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            doc = self.doc_store.get(idx)
            if doc is None:
                continue

            # Apply metadata filters
            if filter_constraints and not self._matches_filters(doc["metadata"], filter_constraints):
                continue

            # Convert L2 distance to similarity score (lower is better, so invert)
            score = 1.0 / (1.0 + float(dist))

            results.append(SearchResult(
                id=doc["id"],
                content=doc["content"],
                score=score,
                metadata=doc["metadata"],
                source_index="vector"
            ))

            if len(results) >= top_k:
                break

        return results

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches filter constraints"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def get_index_size(self) -> int:
        """Return number of documents in the index"""
        return self.index.ntotal
