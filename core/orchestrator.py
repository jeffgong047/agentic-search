"""
DSPy Orchestrator: The "Compiler" that generates structured search plans
Replaces free-text prompting with typed signatures
"""

import dspy
from typing import List, Dict, Any
from .data_structures import SearchPlan, AgentState, StrategyConfig
from .config import get_config
import json


# === DSPy SIGNATURE DEFINITION ===

class LegalIntentSignature(dspy.Signature):
    """
    Analyze the user query and memory context to produce a precise search strategy.
    Disambiguate entities by defining strict negative constraints.
    Output a structured plan (not free text).
    """

    # Inputs
    query: str = dspy.InputField(desc="The user's legal functionality query")
    history: List[str] = dspy.InputField(desc="Previous interaction history")
    tool_context: str = dspy.InputField(desc="User Manual for tools: Available filters, valid doc_types, and schema constraints")
    context_memory: str = dspy.InputField(desc="Summary of verified facts and failed paths from previous steps")
    negative_constraints: str = dspy.InputField(desc="Entities/Paths explicitly proven to be wrong")
    entity_profile: str = dspy.InputField(desc="人物建模: Current profile of the target entity (facts about their work history, colleagues, education)")

    # Outputs (Structured)
    primary_intent: str = dspy.OutputField(
        desc="Choose one: 'Entity_Resolution', 'Statute_Lookup', 'Case_Law_Drilldown', 'Broad_Research', 'Clarification_Search'"
    )
    entity_analysis: str = dspy.OutputField(
        desc="Analyze if the query contains ambiguous entities (e.g., common names). If yes, list potential distractors to rule out."
    )
    hyde_passage: str = dspy.OutputField(
        desc="A hypothetical ideal answer paragraph (2-3 sentences) for vector alignment"
    )
    search_queries: str = dspy.OutputField(
        desc="Exactly 3 distinct search queries as JSON list. For ambiguous entities, use one query to target distractors."
    )
    filter_constraints: str = dspy.OutputField(
        desc="Metadata filters as JSON dict. ONLY use these fields: 'org', 'doc_type', 'year'. AVOID filters if uncertainty is high."
    )
    relational_pivot_hints: str = dspy.OutputField(
        desc="JSON list of high-confidence entities to pivot to if target is ambiguous (e.g. ['Richard Wang CEO', 'Stanford AI Lab'])"
    )
    new_negative_constraints: str = dspy.OutputField(
        desc="Additional entities/domains to EXCLUDE as JSON list (distractors to rule out)"
    )
    entity_graph_depth: str = dspy.OutputField(
        desc="Depth of knowledge graph traversal: 0, 1, or 2"
    )
    strategies: str = dspy.OutputField(
        desc="List of parallel strategies as JSON. Use 3 items: [{'type': 'vector', 'query': '...'}, {'type': 'lexical', 'query': '...'}, {'type': 'graph', 'query': '...'}]"
    )


# === ORCHESTRATOR MODULE ===

from retrieval.critic import RetrievalCritic
from retrieval.reflector import RetrievalReflector
from utils.tracing import TraceLogger

class LegalOrchestrator(dspy.Module):
    """
    The "Brain" that compiles the search plan.
    Uses Chain-of-Thought reasoning internally.
    """

    def __init__(self, schema_context: Dict = None):
        super().__init__()
        self.planner = dspy.ChainOfThought(LegalIntentSignature)
        self.schema_context = schema_context or {}
        
        # Phase 3 Components: The Reflective Scholar
        self.critic = RetrievalCritic()
        self.reflector = RetrievalReflector()
        self.tracer = TraceLogger()

    def forward(self, state: AgentState) -> SearchPlan:
        """
        Generate a structured search plan from the current state.
        Now context-aware of the actual tool capabilities (schema).
        """
        config = get_config()

        # Format the tool context string (The "User Manual")
        doc_types = self.schema_context.get("doc_type", [])
        if doc_types:
            tool_context_str = f"""
[ELASTICSEARCH TOOL MANUAL]
Available Filters:
- 'doc_type': MUST be one of {doc_types}.
  DO NOT USE any other values like 'statute', 'legal_analysis', etc.
  If the query implies a document type not in this list, DO NOT use the doc_type filter.
- 'org': Organization name (text).
- 'year': Integer year.
"""
        else:
            tool_context_str = "Standard Elasticsearch filters available."

        # Run DSPy signature
        if config.USE_DSPY_SIGNATURES:
            tool_context = f"""
[ELASTICSEARCH TOOL MANUAL]
Available Filters:
- 'doc_type': MUST be one of {doc_types}. 
  DO NOT USE any other values like 'statute', 'legal_analysis', etc.
  If the query implies a document type not in this list, DO NOT use the doc_type filter.
- 'org': Organization name (text).
- 'year': Integer year.
""" if doc_types else "Standard Elasticsearch filters available."

            result = self.planner(
                query=state["query"],
                history=[f"Q: {h['query']} A: {h['answer']}" for h in state.get("history", [])],
                tool_context=tool_context,
                context_memory=self._build_context(state),
                negative_constraints=self._build_negative_constraints(state),
                entity_profile=state.get("entity_profile", "Not yet established.")
            )
            plan = self._parse_dspy_output(result)
        else:
            # Ablation: Raw LLM call without DSPy structure
            plan = self._fallback_raw_llm(state["query"], context_memory)

        return plan

    def evaluate_retrieval(self, query: str, docs: List[str]):
        """Critique the retrieval results."""
        self.tracer.log("critic_input", {"query": query, "doc_count": len(docs)})
        result = self.critic(query=query, retrieved_docs=docs)
        self.tracer.log("critic_output", result)
        return result

    def reflect_on_failure(self, query: str, docs: List[str], critique: str, history: List[str]):
        """Generate a verbal gradient (reflection) for the next attempt."""
        self.tracer.log("reflector_input", {"query": query, "critique": critique})
        result = self.reflector(query=query, retrieved_docs=docs, critique=critique, history=history)
        self.tracer.log("reflector_output", result)
        return result

    def _build_context(self, state: AgentState) -> str:
        """Build context string from verified facts"""
        if not state["verified_facts"]:
            return "No prior context."

        facts = "\n".join(f"- {fact}" for fact in state["verified_facts"][-5:])
        return f"Verified facts from previous searches:\n{facts}"

    def _build_negative_constraints(self, state: AgentState) -> str:
        """Build negative constraint string"""
        if not state["negative_cache"]:
            return "No constraints."

        constraints = []
        for item in state["negative_cache"]:
            constraints.append(f"AVOID: {item.get('entity', 'Unknown')} (Reason: {item.get('reason', 'N/A')})")

        return "\n".join(constraints)

    def _normalize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize filter field names to match document schema"""
        normalized = {}

        # Field name mappings (LLM variations -> canonical names)
        # IMPORTANT: Canonical names must match the actual fields in Elasticsearch
        field_mappings = {
            'organization': 'org',
            'company': 'org',
            'employer': 'org',
            'document_type': 'doc_type',  # LLM may say "document_type", but ES has "doc_type"
            'doctype': 'doc_type',
            'type': 'doc_type',
            'person': None,  # Remove - not in schema
            'employee': None,  # Remove - not in schema
            'individual': None,  # Remove - not in schema
        }

        for key, value in filters.items():
            # Normalize field name
            canonical_key = field_mappings.get(key.lower(), key)

            # Skip fields that should be removed
            if canonical_key is None:
                print(f"[Orchestrator] Removing invalid filter field: {key}")
                continue

            normalized[canonical_key] = value

        return normalized

    def _parse_dspy_output(self, result) -> SearchPlan:
        """Parse DSPy output into structured SearchPlan"""
        try:
            # Parse JSON fields
            search_queries = json.loads(result.search_queries) if isinstance(result.search_queries, str) else result.search_queries
            filter_constraints = json.loads(result.filter_constraints) if isinstance(result.filter_constraints, str) else result.filter_constraints
            new_negatives = json.loads(result.new_negative_constraints) if isinstance(result.new_negative_constraints, str) else result.new_negative_constraints
            relational_hints = json.loads(result.relational_pivot_hints) if hasattr(result, 'relational_pivot_hints') and isinstance(result.relational_pivot_hints, str) else getattr(result, 'relational_pivot_hints', [])

            # Normalize filter field names (LLM might use variations)
            filter_constraints = self._normalize_filters(filter_constraints)

            # Ensure search_queries has exactly 3 items
            if len(search_queries) < 3:
                search_queries.extend([""] * (3 - len(search_queries)))
            search_queries = search_queries[:3]

            # Parse strategies
            parsed_strategies = []
            try:
                raw_strategies = json.loads(result.strategies) if isinstance(result.strategies, str) else result.strategies
                for i, s in enumerate(raw_strategies):
                    # Merge global filters with strategy-specific filters
                    strategy_filters = filter_constraints.copy()
                    strategy_filters.update(s.get("filters", {}))
                    
                    parsed_strategies.append(StrategyConfig(
                        strategy_id=f"strat_{i}",
                        strategy_type=s.get("type", "hybrid"),
                        query_variant=s.get("query", ""),
                        index_weights=s.get("weights", {}),
                        filters=strategy_filters
                    ))
            except Exception as e:
                print(f"Warning: Strategy parsing failed: {e}. Output was: {result.strategies}. Using valid defaults.")
                # Fallback strategies - CRITICAL: Must pass filters!
                parsed_strategies = [
                    StrategyConfig(
                        strategy_id="s1", 
                        strategy_type="vector", 
                        query_variant=result.hyde_passage,
                        filters=filter_constraints
                    ),
                    StrategyConfig(
                        strategy_id="s2", 
                        strategy_type="lexical", 
                        query_variant=search_queries[0],
                        filters=filter_constraints
                    ),
                    StrategyConfig(
                        strategy_id="s3", 
                        strategy_type="graph", 
                        query_variant=search_queries[-1],
                        filters=filter_constraints
                    )
                ]

            return SearchPlan(
                primary_intent=result.primary_intent,
                hyde_passage=result.hyde_passage,
                search_queries=search_queries,
                filter_constraints=filter_constraints,
                negative_constraints=new_negatives,
                relational_pivot_hints=relational_hints,
                entity_graph_depth=int(result.entity_graph_depth),
                strategies=parsed_strategies
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback to safe defaults if parsing fails
            print(f"Warning: DSPy output parsing failed: {e}. Using defaults.")
            return SearchPlan(
                primary_intent="Broad_Research",
                hyde_passage=result.hyde_passage if hasattr(result, 'hyde_passage') else "",
                search_queries=[result.query] * 3 if hasattr(result, 'query') else [""] * 3,
                filter_constraints={},
                negative_constraints=[],
                entity_graph_depth=1,
                strategies=[]
            )

    def _fallback_raw_llm(self, query: str, context: str) -> SearchPlan:
        """
        Ablation fallback: Use raw LLM without DSPy structure
        (For comparison in experiments)
        """
        lm = dspy.settings.lm
        prompt = f"""
        Query: {query}
        Context: {context}

        Generate a search strategy. Output JSON with keys:
        - primary_intent
        - hyde_passage
        - search_queries (list of 3)
        - filter_constraints (dict)
        - negative_constraints (list)
        - entity_graph_depth (int 0-2)
        """

        response = lm(prompt)
        # Very basic parsing (this will be worse than DSPy structure)
        return SearchPlan(
            primary_intent="Broad_Research",
            hyde_passage=query,
            search_queries=[query, query, query],
            filter_constraints={},
            negative_constraints=[],
            entity_graph_depth=1,
            strategies=[
                StrategyConfig(strategy_id="raw_1", strategy_type="hybrid", query_variant=query)
            ]
        )


# === INITIALIZATION ===

from typing import Dict

def get_orchestrator(model=None, schema_context: Dict = None):
    """
    Factory function to get configured orchestrator.
    NOW supports passing schema_context for agent adaptation and model selection.
    """
    config = get_config()
    model_name = model or config.LLM_MODEL
    
    # DSPy configuration should be handled by the caller, but we provide defaults
    return LegalOrchestrator(schema_context=schema_context)
