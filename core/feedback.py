"""
Strategy Feedback Loop: Track retrieval outcomes to learn what works.

This module implements a simple feedback system that:
1. Logs which strategies (vector, lexical, graph) succeed for which query intents
2. Stores outcomes in SQLite (local) or PostgreSQL (production)
3. Uses historical data to inform strategy selection

Start small: Log first, optimize later.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import json


# =============================================================================
# Data Structures
# =============================================================================

class QueryIntent(str, Enum):
    """Types of query intents the orchestrator identifies."""
    ENTITY_RESOLUTION = "Entity_Resolution"
    STATUTE_LOOKUP = "Statute_Lookup"
    CASE_LAW_DRILLDOWN = "Case_Law_Drilldown"
    BROAD_RESEARCH = "Broad_Research"
    CLARIFICATION_SEARCH = "Clarification_Search"


class StrategyType(str, Enum):
    """Types of retrieval strategies."""
    VECTOR = "vector"
    LEXICAL = "lexical"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class StrategyOutcome:
    """
    Records the outcome of a retrieval strategy for learning.

    This is the core data structure for the feedback loop.
    """
    # Identifiers
    session_id: str
    query_id: str

    # Query context
    query_text: str
    query_intent: str  # From orchestrator's primary_intent

    # Strategy used
    strategy_type: str  # vector, lexical, graph, hybrid
    strategy_config: Dict[str, Any] = field(default_factory=dict)

    # Outcome metrics
    docs_retrieved: int = 0
    docs_after_filter: int = 0
    novelty_score: float = 0.0
    rerank_top_score: float = 0.0

    # Success signal (can be from multiple sources)
    user_feedback: Optional[float] = None  # 0-1 from explicit feedback
    implicit_success: Optional[float] = None  # Derived from behavior

    # Timing
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_intent": self.query_intent,
            "strategy_type": self.strategy_type,
            "strategy_config": json.dumps(self.strategy_config),
            "docs_retrieved": self.docs_retrieved,
            "docs_after_filter": self.docs_after_filter,
            "novelty_score": self.novelty_score,
            "rerank_top_score": self.rerank_top_score,
            "user_feedback": self.user_feedback,
            "implicit_success": self.implicit_success,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyOutcome":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            query_id=data["query_id"],
            query_text=data["query_text"],
            query_intent=data["query_intent"],
            strategy_type=data["strategy_type"],
            strategy_config=json.loads(data.get("strategy_config", "{}")),
            docs_retrieved=data.get("docs_retrieved", 0),
            docs_after_filter=data.get("docs_after_filter", 0),
            novelty_score=data.get("novelty_score", 0.0),
            rerank_top_score=data.get("rerank_top_score", 0.0),
            user_feedback=data.get("user_feedback"),
            implicit_success=data.get("implicit_success"),
            latency_ms=data.get("latency_ms", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.utcnow()),
            metadata=json.loads(data.get("metadata", "{}")),
        )


@dataclass
class StrategyStats:
    """Aggregated statistics for a strategy-intent combination."""
    intent: str
    strategy: str
    total_uses: int
    avg_success: float
    avg_latency_ms: float
    avg_docs_retrieved: float
    last_used: datetime


# =============================================================================
# Storage Interface
# =============================================================================

class FeedbackStorage(ABC):
    """Abstract interface for feedback storage backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize storage (create tables, etc.)."""
        pass

    @abstractmethod
    def log_outcome(self, outcome: StrategyOutcome) -> None:
        """Store a strategy outcome."""
        pass

    @abstractmethod
    def get_strategy_stats(
        self,
        intent: Optional[str] = None,
        strategy: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[StrategyStats]:
        """Get aggregated statistics for strategies."""
        pass

    @abstractmethod
    def get_recent_outcomes(
        self,
        limit: int = 100,
        intent: Optional[str] = None
    ) -> List[StrategyOutcome]:
        """Get recent outcomes for analysis."""
        pass

    @abstractmethod
    def get_best_strategy(self, intent: str) -> Optional[str]:
        """Get the historically best-performing strategy for an intent."""
        pass


# =============================================================================
# Strategy Selector
# =============================================================================

class StrategySelector:
    """
    Selects retrieval strategies based on historical performance.

    Starts with simple rules, can be upgraded to ML later.
    """

    def __init__(self, storage: FeedbackStorage, min_samples: int = 10):
        """
        Initialize strategy selector.

        Args:
            storage: Feedback storage backend
            min_samples: Minimum samples before using historical data
        """
        self.storage = storage
        self.min_samples = min_samples

        # Default strategy weights when no data available
        self.default_weights = {
            QueryIntent.ENTITY_RESOLUTION.value: {"graph": 0.4, "lexical": 0.4, "vector": 0.2},
            QueryIntent.STATUTE_LOOKUP.value: {"lexical": 0.5, "vector": 0.3, "graph": 0.2},
            QueryIntent.CASE_LAW_DRILLDOWN.value: {"vector": 0.4, "lexical": 0.3, "graph": 0.3},
            QueryIntent.BROAD_RESEARCH.value: {"vector": 0.5, "lexical": 0.3, "graph": 0.2},
            QueryIntent.CLARIFICATION_SEARCH.value: {"vector": 0.4, "lexical": 0.4, "graph": 0.2},
        }

    def get_strategy_weights(self, intent: str) -> Dict[str, float]:
        """
        Get recommended strategy weights for a query intent.

        Returns dict like {"vector": 0.4, "lexical": 0.3, "graph": 0.3}
        """
        # Get historical stats
        stats = self.storage.get_strategy_stats(intent=intent)

        # Check if we have enough data
        total_samples = sum(s.total_uses for s in stats)

        if total_samples < self.min_samples:
            # Fall back to defaults
            return self.default_weights.get(intent, {"vector": 0.4, "lexical": 0.3, "graph": 0.3})

        # Calculate weights based on success rates
        weights = {}
        total_success = sum(s.avg_success * s.total_uses for s in stats)

        if total_success == 0:
            # No successes recorded, use defaults
            return self.default_weights.get(intent, {"vector": 0.4, "lexical": 0.3, "graph": 0.3})

        for stat in stats:
            # Weight by success rate and usage
            weights[stat.strategy] = (stat.avg_success * stat.total_uses) / total_success

        # Normalize
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v / weight_sum for k, v in weights.items()}

        return weights

    def get_recommended_strategies(self, intent: str, top_k: int = 3) -> List[str]:
        """
        Get ranked list of recommended strategies for an intent.

        Returns list like ["graph", "lexical", "vector"]
        """
        weights = self.get_strategy_weights(intent)
        sorted_strategies = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_strategies[:top_k]]

    def should_explore(self, intent: str, exploration_rate: float = 0.1) -> bool:
        """
        Decide whether to explore (try less-used strategies) vs exploit.

        Simple epsilon-greedy for now.
        """
        import random
        return random.random() < exploration_rate


# =============================================================================
# Feedback Logger (Convenience wrapper)
# =============================================================================

class FeedbackLogger:
    """
    Convenience class for logging feedback during agent execution.

    Usage:
        logger = FeedbackLogger(storage)
        logger.start_query(session_id, query_id, query_text, intent)
        logger.log_strategy_result("vector", docs_retrieved=20, latency_ms=150)
        logger.log_strategy_result("lexical", docs_retrieved=15, latency_ms=80)
        logger.finalize(user_feedback=0.8)
    """

    def __init__(self, storage: FeedbackStorage):
        self.storage = storage
        self._current_session: Optional[str] = None
        self._current_query: Optional[str] = None
        self._current_text: Optional[str] = None
        self._current_intent: Optional[str] = None
        self._pending_outcomes: List[StrategyOutcome] = []

    def start_query(
        self,
        session_id: str,
        query_id: str,
        query_text: str,
        intent: str
    ) -> None:
        """Start tracking a new query."""
        self._current_session = session_id
        self._current_query = query_id
        self._current_text = query_text
        self._current_intent = intent
        self._pending_outcomes = []

    def log_strategy_result(
        self,
        strategy_type: str,
        docs_retrieved: int = 0,
        docs_after_filter: int = 0,
        novelty_score: float = 0.0,
        rerank_top_score: float = 0.0,
        latency_ms: float = 0.0,
        config: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log the result of a single strategy."""
        if not self._current_query:
            raise RuntimeError("Call start_query() before logging results")

        outcome = StrategyOutcome(
            session_id=self._current_session,
            query_id=self._current_query,
            query_text=self._current_text,
            query_intent=self._current_intent,
            strategy_type=strategy_type,
            strategy_config=config or {},
            docs_retrieved=docs_retrieved,
            docs_after_filter=docs_after_filter,
            novelty_score=novelty_score,
            rerank_top_score=rerank_top_score,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._pending_outcomes.append(outcome)

    def finalize(
        self,
        user_feedback: Optional[float] = None,
        implicit_success: Optional[float] = None
    ) -> None:
        """Finalize and store all outcomes for this query."""
        for outcome in self._pending_outcomes:
            outcome.user_feedback = user_feedback
            outcome.implicit_success = implicit_success
            self.storage.log_outcome(outcome)

        self._pending_outcomes = []
        self._current_query = None
