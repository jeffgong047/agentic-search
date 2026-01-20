"""
Cascade Recall Funnel: Combines tri-index results with filtering and reranking
Implements the 3-tier architecture: Broad Sweep -> Hard Filter -> Rerank
"""

from typing import List, Dict, Any, Set
from sentence_transformers import CrossEncoder
from data_structures import SearchResult, SearchPlan
from config import get_config


class CascadeRecallFunnel:
    """
    Multi-stage retrieval funnel:
    Tier 1: Aggregate results from Vector, BM25, Graph
    Tier 2: Apply metadata filters and negative constraints
    Tier 3: Rerank using cross-encoder
    """

    def __init__(self, reranker_model: str | None = None):
        """
        Initialize the cascade funnel.

        Args:
            reranker_model: Cross-encoder model for reranking (uses config default if None)
        """
        config = get_config()
        model_name = reranker_model or config.RERANKER_MODEL

        self.reranker = CrossEncoder(model_name)

    def aggregate_and_filter(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        graph_results: List[SearchResult],
        search_plan: SearchPlan,
        negative_cache: List[Dict[str, str]] | None = None
    ) -> List[SearchResult]:
        """
        Tier 1 & 2: Aggregate results and apply filters.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            graph_results: Results from graph traversal
            search_plan: The search plan with filter constraints
            negative_cache: List of failed paths to exclude

        Returns:
            Filtered list of unique SearchResult objects
        """
        config = get_config()

        # Tier 1: Merge results (deduplication by ID)
        all_results: Dict[str, SearchResult] = {}

        for result in vector_results + bm25_results + graph_results:
            if result.id in all_results:
                # Keep the one with higher score
                if result.score > all_results[result.id].score:
                    all_results[result.id] = result
            else:
                all_results[result.id] = result

        results = list(all_results.values())

        if config.DEBUG_MODE and config.LOG_RETRIEVAL_STEPS:
            print(f"[Cascade] Tier 1 - Aggregated {len(results)} unique documents")
            print(f"  Vector: {len(vector_results)}, BM25: {len(bm25_results)}, Graph: {len(graph_results)}")

        # Tier 2: Apply hard filters
        filtered_results = []

        for result in results:
            # Apply metadata filters from search plan
            if not self._matches_filters(result.metadata, search_plan.filter_constraints):
                continue

            # Apply negative constraints (exclude wrong entities)
            if negative_cache and self._matches_negative_constraints(result, negative_cache):
                if config.DEBUG_MODE:
                    print(f"[Cascade] Filtered out doc {result.id[:20]}... (negative constraint match)")
                continue

            # Apply search plan negative constraints
            if self._contains_negative_terms(result.content, search_plan.negative_constraints):
                if config.DEBUG_MODE:
                    print(f"[Cascade] Filtered out doc {result.id[:20]}... (negative term match)")
                continue

            filtered_results.append(result)

        if config.DEBUG_MODE and config.LOG_RETRIEVAL_STEPS:
            print(f"[Cascade] Tier 2 - Filtered to {len(filtered_results)} documents")

        return filtered_results

    def rerank(
        self,
        results: List[SearchResult],
        query: str,
        hyde_passage: str,
        top_k: int | None = None
    ) -> List[SearchResult]:
        """
        Tier 3: Rerank using cross-encoder.

        Args:
            results: Filtered results from Tier 2
            query: Original user query
            hyde_passage: Hypothetical ideal passage (HyDE)
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        config = get_config()
        top_k = top_k or config.FINAL_TOP_K

        if not results:
            return []

        if not config.USE_CASCADE_RECALL:
            # Ablation: Skip reranking, just return top-k by original score
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        # Prepare pairs for cross-encoder
        # Use HyDE passage for better semantic matching
        query_text = hyde_passage if hyde_passage else query
        pairs = [(query_text, result.content) for result in results]

        # Score with cross-encoder
        scores = self.reranker.predict(pairs)

        # Update scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        if config.DEBUG_MODE and config.LOG_RETRIEVAL_STEPS:
            print(f"[Cascade] Tier 3 - Reranked to top {len(reranked)} documents")

        return reranked

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filter constraints"""
        if not filters:
            return True

        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False

        return True

    def _matches_negative_constraints(
        self,
        result: SearchResult,
        negative_cache: List[Dict[str, str]]
    ) -> bool:
        """Check if result matches any negative constraint"""
        for constraint in negative_cache:
            entity = constraint.get("entity", "").lower()
            reason = constraint.get("reason", "").lower()

            # Check if the negative entity appears in content or metadata
            content_lower = result.content.lower()

            if entity and entity in content_lower:
                # Additional check: Does the reason also match?
                if reason:
                    # Check metadata for reason indicators
                    for meta_value in result.metadata.values():
                        if isinstance(meta_value, str) and reason in meta_value.lower():
                            return True
                else:
                    return True

        return False

    def _contains_negative_terms(self, content: str, negative_terms: List[str]) -> bool:
        """Check if content contains any negative terms"""
        if not negative_terms:
            return False

        content_lower = content.lower()
        for term in negative_terms:
            if term.lower() in content_lower:
                return True

        return False
