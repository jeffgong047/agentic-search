"""
Reranker Module for Agentic Search
Implements Cross-Encoder reranking to improve precision.
"""

from typing import List
from sentence_transformers import CrossEncoder
from production_interfaces import SearchResult

class CrossEncoderReranker:
    """
    Reranks a list of SearchResults using a Cross-Encoder model.
    Benefits: Higher precision ("finding the needle") at the cost of latency.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model ID for the Cross-Encoder.
            device: 'cpu', 'cuda', or 'mps' (for Mac).
        """
        print(f"[Reranker] Loading model: {model_name}...")
        self.model = CrossEncoder(model_name, device=device)
        print("[Reranker] Model loaded.")

    def rank(self, query: str, docs: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """
        Rerank a list of documents against a query.

        Args:
            query: The user query string.
            docs: List of candidate SearchResult objects.
            top_k: Number of results to return after sorting.

        Returns:
            List of the top_k SearchResults, sorted by relevance score.
        """
        if not docs:
            return []

        # Prepare pairs for scoring: (query, doc_content)
        # We rely on 'content' being populated in SearchResult.
        # Fallback to empty string if missing.
        pairs = [(query, doc.content or "") for doc in docs]

        # Predict scores (logits or probabilities)
        # show_progress_bar=False to reduce noise in logs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores to documents (optional, for debugging)
        for i, doc in enumerate(docs):
            # Store the rerank score in metadata for analysis
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["rerank_score"] = float(scores[i])

        # Zip docs with scores and sort
        doc_score_pairs = list(zip(docs, scores))
        # Sort desc by score
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k docs
        ranked_docs = [pair[0] for pair in doc_score_pairs]
        return ranked_docs[:top_k]
