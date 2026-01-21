"""
Feedback Integration: Wraps the search engine with automatic outcome logging.

This module provides:
1. FeedbackEnabledSearch - Wrapper that logs all search outcomes
2. Integration helpers for the graph engine
3. Success signal computation

Usage:
    from core.feedback_integration import FeedbackEnabledSearch
    from core.feedback_storage import get_feedback_storage

    storage = get_feedback_storage("sqlite")
    storage.initialize()

    search_engine = FeedbackEnabledSearch(
        base_engine=your_agentic_search_engine,
        storage=storage
    )

    # Search with automatic feedback logging
    result = search_engine.search("your query")

    # Optionally add user feedback
    search_engine.record_user_feedback(session_id, query_id, score=0.8)
"""

import time
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

from .feedback import FeedbackLogger, StrategySelector, StrategyOutcome
from .feedback_storage import FeedbackStorage, get_feedback_storage
from .data_structures import AgentState, SearchResult
from .config import get_config


class FeedbackEnabledSearch:
    """
    Wrapper that adds feedback logging to any search engine.

    Automatically logs:
    - Query intents
    - Strategy outcomes (docs retrieved, latency, etc.)
    - Implicit success signals (novelty, rerank scores)
    """

    def __init__(
        self,
        base_engine: Any,  # AgenticSearchEngine
        storage: FeedbackStorage,
        enable_strategy_selection: bool = True
    ):
        """
        Initialize feedback-enabled search.

        Args:
            base_engine: The underlying AgenticSearchEngine
            storage: Feedback storage backend
            enable_strategy_selection: Use historical data for strategy hints
        """
        self.base_engine = base_engine
        self.storage = storage
        self.logger = FeedbackLogger(storage)
        self.selector = StrategySelector(storage) if enable_strategy_selection else None

        self._current_session_id: Optional[str] = None
        self._query_feedback_map: Dict[str, str] = {}  # query_id -> session_id

    def search(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> AgentState:
        """
        Execute search with automatic feedback logging.

        Args:
            query: User's search query
            session_id: Optional session ID (generated if not provided)

        Returns:
            Final agent state with results
        """
        config = get_config()

        # Generate IDs
        session_id = session_id or self._current_session_id or str(uuid.uuid4())[:8]
        query_id = str(uuid.uuid4())[:8]
        self._current_session_id = session_id
        self._query_feedback_map[query_id] = session_id

        # Start timing
        start_time = time.time()

        # Execute search
        final_state = self.base_engine.search(query)

        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000

        # Extract intent from the search plan
        intent = "Broad_Research"  # default
        if final_state.get("search_plan"):
            intent = final_state["search_plan"].primary_intent

        # Start logging
        self.logger.start_query(session_id, query_id, query, intent)

        # Log aggregated outcome (simplified - could be per-strategy in more advanced version)
        self._log_search_outcome(final_state, query_id, total_latency_ms)

        # Compute implicit success signal
        implicit_success = self._compute_implicit_success(final_state)

        # Finalize logging
        self.logger.finalize(implicit_success=implicit_success)

        # Attach query_id to state for later feedback
        final_state["_feedback_query_id"] = query_id
        final_state["_feedback_session_id"] = session_id

        if config.DEBUG_MODE:
            print(f"[Feedback] Logged outcome for query {query_id}")
            print(f"  Intent: {intent}")
            print(f"  Implicit success: {implicit_success:.3f}")

        return final_state

    def record_user_feedback(
        self,
        query_id: str,
        score: float,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Record explicit user feedback for a query.

        Args:
            query_id: The query ID (from state["_feedback_query_id"])
            score: User satisfaction score (0-1)
            metadata: Optional additional metadata
        """
        # Update the outcome with user feedback
        # Note: In a production system, you'd update the existing record
        # For simplicity, we log a new outcome with user_feedback set

        session_id = self._query_feedback_map.get(query_id, "unknown")

        outcome = StrategyOutcome(
            session_id=session_id,
            query_id=query_id,
            query_text="[feedback update]",
            query_intent="[feedback update]",
            strategy_type="aggregate",
            user_feedback=score,
            metadata=metadata or {}
        )

        self.storage.log_outcome(outcome)

    def get_strategy_recommendations(self, intent: str) -> Dict[str, float]:
        """
        Get strategy weight recommendations based on historical data.

        Args:
            intent: Query intent type

        Returns:
            Dict of strategy -> weight
        """
        if self.selector:
            return self.selector.get_strategy_weights(intent)
        return {"vector": 0.4, "lexical": 0.3, "graph": 0.3}

    def _log_search_outcome(
        self,
        state: AgentState,
        query_id: str,
        latency_ms: float
    ) -> None:
        """Log the search outcome details."""
        results = state.get("retrieved_docs", [])

        # Count by source
        source_counts = {}
        for r in results:
            source = getattr(r, "source_index", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        # Get top rerank score
        top_score = max((r.score for r in results), default=0.0)

        # Log aggregate outcome
        self.logger.log_strategy_result(
            strategy_type="aggregate",
            docs_retrieved=len(results),
            docs_after_filter=len(results),
            novelty_score=state.get("novelty_score", 0.0),
            rerank_top_score=top_score,
            latency_ms=latency_ms,
            metadata={
                "source_distribution": source_counts,
                "iterations": state.get("step_count", 0),
                "verified_facts_count": len(state.get("verified_facts", [])),
                "negative_cache_size": len(state.get("negative_cache", []))
            }
        )

        # Also log per-source outcomes for finer analysis
        for source, count in source_counts.items():
            source_results = [r for r in results if getattr(r, "source_index", "unknown") == source]
            source_top_score = max((r.score for r in source_results), default=0.0)

            self.logger.log_strategy_result(
                strategy_type=source,
                docs_retrieved=count,
                docs_after_filter=count,
                rerank_top_score=source_top_score,
                latency_ms=0,  # We don't have per-strategy timing here
                metadata={"is_component": True}
            )

    def _compute_implicit_success(self, state: AgentState) -> float:
        """
        Compute implicit success signal from search outcome.

        Combines multiple signals:
        - Number of results (did we find anything?)
        - Novelty score (did we find new information?)
        - Top rerank score (how relevant is the best result?)
        - Iterations needed (fewer is better)
        """
        results = state.get("retrieved_docs", [])
        config = get_config()

        if not results:
            return 0.0

        # Signal 1: Did we get results? (0 or 1)
        has_results = 1.0 if len(results) > 0 else 0.0

        # Signal 2: Top rerank score (already 0-1 range typically)
        top_score = max((r.score for r in results), default=0.0)
        # Normalize if needed (cross-encoder scores can be negative)
        top_score_norm = max(0.0, min(1.0, (top_score + 1) / 2)) if top_score < 0 else min(1.0, top_score)

        # Signal 3: Novelty (we want high novelty = finding new info)
        novelty = state.get("novelty_score", 0.5)

        # Signal 4: Efficiency (fewer iterations = better)
        iterations = state.get("step_count", 1)
        efficiency = 1.0 / iterations if iterations > 0 else 0.0

        # Weighted combination
        implicit_success = (
            0.2 * has_results +
            0.4 * top_score_norm +
            0.2 * novelty +
            0.2 * efficiency
        )

        return min(1.0, max(0.0, implicit_success))


# =============================================================================
# Factory Function
# =============================================================================

def create_feedback_enabled_search(
    base_engine: Any,
    storage_backend: str = "sqlite",
    **storage_kwargs
) -> FeedbackEnabledSearch:
    """
    Factory function to create a feedback-enabled search engine.

    Args:
        base_engine: The underlying AgenticSearchEngine
        storage_backend: "sqlite" or "postgres"
        **storage_kwargs: Backend-specific configuration

    Returns:
        FeedbackEnabledSearch instance

    Usage:
        # Local development
        engine = create_feedback_enabled_search(
            base_engine,
            storage_backend="sqlite",
            db_path="./feedback.db"
        )

        # Production
        engine = create_feedback_enabled_search(
            base_engine,
            storage_backend="postgres",
            host="db.evenup.com",
            database="prod_db"
        )
    """
    storage = get_feedback_storage(storage_backend, **storage_kwargs)
    storage.initialize()

    return FeedbackEnabledSearch(base_engine, storage)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    # Quick test with SQLite
    from .feedback_storage import SQLiteStorage

    print("Testing Feedback System...")

    # Create storage
    storage = SQLiteStorage("test_feedback.db")
    storage.initialize()

    # Create logger
    logger = FeedbackLogger(storage)

    # Simulate a search
    logger.start_query("test_session", "query_001", "test query", "Entity_Resolution")
    logger.log_strategy_result("vector", docs_retrieved=20, latency_ms=150)
    logger.log_strategy_result("lexical", docs_retrieved=15, latency_ms=80)
    logger.log_strategy_result("graph", docs_retrieved=5, latency_ms=200)
    logger.finalize(implicit_success=0.75)

    # Check stats
    stats = storage.get_strategy_stats()
    print(f"\nStrategy Stats:")
    for s in stats:
        print(f"  {s.intent} + {s.strategy}: {s.total_uses} uses, {s.avg_success:.2f} avg success")

    # Test selector
    selector = StrategySelector(storage)
    weights = selector.get_strategy_weights("Entity_Resolution")
    print(f"\nRecommended weights for Entity_Resolution: {weights}")

    print("\nTest complete!")
