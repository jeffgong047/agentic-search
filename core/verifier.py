"""
Novelty Circuit Breaker: Deterministic loop termination
Uses set theory to calculate information gain and prevent infinite loops
"""

from typing import Set, List
from .data_structures import SearchResult, AgentState
from .config import get_config


class NoveltyVerifier:
    """
    Implements the mathematical circuit breaker to stop agentic loops.

    Formula: Novelty = |R_new - R_total| / |R_total|

    If Novelty < epsilon, the search is yielding diminishing returns and should stop.
    """

    def __init__(self, epsilon: float | None = None):
        """
        Initialize the novelty verifier.

        Args:
            epsilon: Novelty threshold (uses config default if None)
        """
        config = get_config()
        self.epsilon = epsilon if epsilon is not None else config.NOVELTY_EPSILON

    def calculate_novelty(
        self,
        current_results: List[SearchResult],
        known_doc_ids: Set[str]
    ) -> tuple[float, Set[str]]:
        """
        Calculate novelty score for current retrieval batch.

        Args:
            current_results: Results from current iteration
            known_doc_ids: Document IDs seen in previous iterations

        Returns:
            Tuple of (novelty_score, new_doc_ids)
        """
        # Extract current document IDs
        current_ids = {result.id for result in current_results}

        # Calculate set difference (new documents)
        new_ids = current_ids - known_doc_ids

        # Calculate novelty ratio
        if len(known_doc_ids) == 0:
            # First iteration: all documents are novel
            novelty = 1.0
        else:
            # Cap at 1.0 to ensure consistent behavior when new > known
            novelty = min(1.0, len(new_ids) / len(known_doc_ids))

        config = get_config()
        if config.DEBUG_MODE:
            print(f"[Verifier] Novelty Calculation:")
            print(f"  Total docs in current batch: {len(current_ids)}")
            print(f"  Previously known docs: {len(known_doc_ids)}")
            print(f"  New docs found: {len(new_ids)}")
            print(f"  Novelty score: {novelty:.3f}")
            print(f"  Threshold (epsilon): {self.epsilon}")

        return novelty, new_ids

    def should_continue(self, state: AgentState) -> bool:
        """
        Determine if the agent should continue searching.

        Args:
            state: Current agent state

        Returns:
            True if should continue, False if should stop
        """
        config = get_config()

        # Check feature flag
        if not config.USE_NOVELTY_CIRCUIT:
            # Ablation: Use fixed iteration count
            should_continue = state["step_count"] < config.MAX_ITERATIONS
            if config.DEBUG_MODE:
                print(f"[Verifier] Novelty circuit DISABLED. Using fixed iterations: {state['step_count']}/{config.MAX_ITERATIONS}")
            return should_continue

        # Check max iterations hard limit
        if state["step_count"] >= config.MAX_ITERATIONS:
            if config.DEBUG_MODE:
                print(f"[Verifier] Max iterations reached ({config.MAX_ITERATIONS}). STOPPING.")
            return False

        # Check novelty score
        if state["novelty_score"] < self.epsilon:
            if config.DEBUG_MODE:
                print(f"[Verifier] Novelty below threshold ({state['novelty_score']:.3f} < {self.epsilon}). STOPPING.")
            return False

        # Continue searching
        if config.DEBUG_MODE:
            print(f"[Verifier] Novelty sufficient ({state['novelty_score']:.3f} >= {self.epsilon}). CONTINUING.")
        return True

    def verify_and_update_state(self, state: AgentState) -> AgentState:
        """
        Calculate novelty and update the state with circuit breaker decision.

        Args:
            state: Current agent state

        Returns:
            Updated agent state with novelty_score and should_continue
        """
        # Calculate novelty from current results
        novelty, new_ids = self.calculate_novelty(
            state["retrieved_docs"],
            state["known_doc_ids"]
        )

        # Update state
        state["novelty_score"] = novelty
        state["known_doc_ids"].update(new_ids)
        state["should_continue"] = self.should_continue(state)

        return state


class QualityFilter:
    """
    Additional quality checks for retrieved documents.
    Can be extended to check for hallucination, relevance, etc.
    """

    def __init__(self, min_score: float = 0.1, min_length: int = 50):
        """
        Initialize quality filter.

        Args:
            min_score: Minimum relevance score
            min_length: Minimum document length (characters)
        """
        self.min_score = min_score
        self.min_length = min_length

    def filter_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Filter results by quality criteria.

        Args:
            results: List of search results

        Returns:
            Filtered list of search results
        """
        filtered = []

        for result in results:
            # Check score
            if result.score < self.min_score:
                continue

            # Check length
            if len(result.content) < self.min_length:
                continue

            # Check for empty/placeholder content
            if self._is_placeholder(result.content):
                continue

            filtered.append(result)

        return filtered

    def _is_placeholder(self, content: str) -> bool:
        """Check if content is a placeholder or empty"""
        content = content.strip().lower()

        # Check for common placeholders
        placeholders = [
            "lorem ipsum",
            "placeholder",
            "todo",
            "tbd",
            "coming soon"
        ]

        for placeholder in placeholders:
            if placeholder in content:
                return True

        return False
