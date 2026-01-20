"""
BM25 Search Engine: Keyword-based retrieval
Handles exact token matching for entities, case IDs, statute numbers
"""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from data_structures import SearchResult
from config import get_config
import re


class BM25SearchEngine:
    """
    Sparse keyword retrieval using BM25 algorithm.
    Optimized for exact entity names, legal citations, and specific terms.
    """

    def __init__(self):
        """Initialize the BM25 search engine"""
        self.bm25: BM25Okapi | None = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the BM25 index.

        Args:
            documents: List of dicts with keys: id, content, metadata
        """
        if not documents:
            return

        self.documents.extend(documents)

        # Tokenize new documents
        new_tokenized = [self._tokenize(doc["content"]) for doc in documents]
        self.tokenized_corpus.extend(new_tokenized)

        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_constraints: Dict[str, Any] | None = None
    ) -> List[SearchResult]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_constraints: Metadata filters to apply

        Returns:
            List of SearchResult objects
        """
        config = get_config()
        top_k = top_k or config.BM25_TOP_K

        if self.bm25 is None or len(self.documents) == 0:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Sort by score (descending)
        ranked_indices = scores.argsort()[::-1]

        # Build results
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:  # Skip zero-score results
                break

            doc = self.documents[idx]

            # Apply metadata filters
            if filter_constraints and not self._matches_filters(doc.get("metadata", {}), filter_constraints):
                continue

            results.append(SearchResult(
                id=doc["id"],
                content=doc["content"],
                score=score,
                metadata=doc.get("metadata", {}),
                source_index="bm25"
            ))

            if len(results) >= top_k:
                break

        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        Preserves legal citations and entity names.
        """
        # Convert to lowercase
        text = text.lower()

        # Simple whitespace + punctuation tokenization
        # Preserve hyphens and periods in legal citations
        tokens = re.findall(r'\b[\w\-\.]+\b', text)

        return tokens

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
        return len(self.documents)
