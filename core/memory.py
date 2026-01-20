"""
Memory Evolution and Reflection Module
Implements intra-session learning to avoid checking wrong paths twice
"""

from typing import List, Dict, Any
from .data_structures import SearchResult, AgentState, ReflectionOutput
from .config import get_config
import dspy


class MemoryReflectionSignature(dspy.Signature):
    """
    Signature for analyzing search results and updating memory.
    Identifies wrong paths to avoid in future iterations.
    """

    # Inputs
    query: str = dspy.InputField(desc="Original user query")
    current_results: str = dspy.InputField(desc="Summary of documents retrieved in this iteration")
    previous_constraints: str = dspy.InputField(desc="Negative constraints from previous iterations")
    previous_entity_profile: str = dspy.InputField(desc="人物建模: Current profile of the target entity")

    # Outputs
    verified_facts: str = dspy.OutputField(desc="New facts confirmed by this iteration (JSON list)")
    negative_entities: str = dspy.OutputField(
        desc="Entities that should be EXCLUDED in future searches (JSON list of dicts with 'entity' and 'reason')"
    )
    context_summary: str = dspy.OutputField(desc="Brief summary of what we've learned")
    updated_entity_profile: str = dspy.OutputField(desc="人物建模: Comprehensive updated profile of the target entity based on new findings")


class MemoryEvolutionModule:
    """
    Manages the agent's working memory and learning process.
    Updates negative constraints based on failed search paths.
    """

    def __init__(self):
        """Initialize the memory evolution module"""
        self.reflector = dspy.ChainOfThought(MemoryReflectionSignature)

    def reflect(self, state: AgentState) -> ReflectionOutput:
        """
        Analyze current search results and update memory.

        Args:
            state: Current agent state with results

        Returns:
            ReflectionOutput with new verified facts and negative constraints
        """
        config = get_config()

        if not config.USE_NEGATIVE_MEMORY:
            # Ablation: Skip memory evolution
            if config.DEBUG_MODE:
                print("[Memory] Memory evolution DISABLED. Skipping reflection.")
            return ReflectionOutput(
                new_verified_facts=[],
                new_negative_constraints=[],
                context_summary=""
            )

        # Summarize current results
        results_summary = self._summarize_results(state["retrieved_docs"])

        # Build previous constraints string
        previous_constraints = self._format_constraints(state["negative_cache"])

        # Run reflection
        try:
            reflection = self.reflector(
                query=state["query"],
                current_results=results_summary,
                previous_constraints=previous_constraints,
                previous_entity_profile=state.get("entity_profile", "Not yet established.")
            )

            # Parse outputs
            import json

            verified_facts = json.loads(reflection.verified_facts) if reflection.verified_facts else []
            negative_entities = json.loads(reflection.negative_entities) if reflection.negative_entities else []

            if config.DEBUG_MODE:
                print(f"[Memory] Reflection complete:")
                print(f"  New verified facts: {len(verified_facts)}")
                print(f"  New negative constraints: {len(negative_entities)}")

            return ReflectionOutput(
                new_verified_facts=verified_facts,
                new_negative_constraints=negative_entities,
                context_summary=reflection.context_summary,
                new_entity_profile=getattr(reflection, 'updated_entity_profile', state.get("entity_profile", ""))
            )

        except Exception as e:
            if config.DEBUG_MODE:
                print(f"[Memory] Reflection error: {e}. Using fallback.")

            # Fallback: Simple rule-based reflection
            return self._fallback_reflection(state)

    def update_state_memory(self, state: AgentState, reflection: ReflectionOutput) -> AgentState:
        """
        Update the agent state with reflection outputs.

        Args:
            state: Current agent state
            reflection: Reflection output

        Returns:
            Updated agent state
        """
        # Add verified facts
        state["verified_facts"].extend(reflection.new_verified_facts)

        # Add negative constraints
        state["negative_cache"].extend(reflection.new_negative_constraints)
        
        # Update character model (人物建模)
        if reflection.new_entity_profile and reflection.new_entity_profile != "Not yet established.":
            state["entity_profile"] = reflection.new_entity_profile

        return state

    def _summarize_results(self, results: List[SearchResult], max_results: int = 5) -> str:
        """Create a summary of search results"""
        if not results:
            return "No documents retrieved."

        summaries = []
        for i, result in enumerate(results[:max_results]):
            # Extract metadata
            metadata_str = ", ".join(f"{k}: {v}" for k, v in result.metadata.items())

            # Truncate content
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content

            summaries.append(
                f"Doc {i+1} (ID: {result.id[:20]}...):\n"
                f"  Metadata: {metadata_str}\n"
                f"  Content: {content_preview}\n"
                f"  Source: {result.source_index}, Score: {result.score:.3f}"
            )

        return "\n\n".join(summaries)

    def _format_constraints(self, negative_cache: List[Dict[str, str]]) -> str:
        """Format negative constraints for display"""
        if not negative_cache:
            return "No previous constraints."

        constraints = []
        for item in negative_cache:
            entity = item.get("entity", "Unknown")
            reason = item.get("reason", "N/A")
            constraints.append(f"- AVOID: {entity} (Reason: {reason})")

        return "\n".join(constraints)

    def _fallback_reflection(self, state: AgentState) -> ReflectionOutput:
        """
        Simple rule-based reflection when DSPy fails.
        Checks for obvious mismatches in metadata.
        """
        results = state["retrieved_docs"]

        if not results:
            return ReflectionOutput(
                new_verified_facts=[],
                new_negative_constraints=[],
                context_summary="No results to reflect on."
            )

        # Simple heuristic: If all results share a common attribute that seems wrong,
        # flag it as a negative constraint
        # (This is a placeholder - in production, you'd use more sophisticated checks)

        metadata_values: Dict[str, List[Any]] = {}
        for result in results:
            for key, value in result.metadata.items():
                if key not in metadata_values:
                    metadata_values[key] = []
                metadata_values[key].append(value)

        # Example: If all results are from a different org than expected
        negative_constraints = []
        if "org" in metadata_values:
            orgs = set(metadata_values["org"])
            if len(orgs) == 1:
                org = list(orgs)[0]
                # Simple check: Does this org appear in the query?
                if org.lower() not in state["query"].lower():
                    negative_constraints.append({
                        "entity": org,
                        "reason": "All results from this org, but not in query"
                    })

        return ReflectionOutput(
            new_verified_facts=[],
            new_negative_constraints=negative_constraints,
            context_summary=f"Retrieved {len(results)} documents. Applied heuristic checks."
        )


def initialize_memory_module() -> MemoryEvolutionModule:
    """Initialize the memory evolution module"""
    return MemoryEvolutionModule()
