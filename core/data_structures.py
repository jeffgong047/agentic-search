"""
Core Data Structures for the Agentic RAG System
Defines the typed state and result schemas
"""

import operator
from typing import Annotated, List, Dict, Set, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# === RETRIEVAL RESULTS ===

class SearchResult(BaseModel):
    """Individual search result from any retrieval index"""

    id: str = Field(description="Unique document ID")
    content: str = Field(description="Document text content")
    score: float = Field(default=0.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    source_index: Literal["vector", "bm25", "graph", "hybrid", "elasticsearch"] = Field(description="Which index this came from")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, SearchResult) and self.id == other.id



# === STRATEGY CONFIG (Parallel Execution) ===

class StrategyConfig(BaseModel):
    """Configuration for a single parallel search strategy"""
    
    strategy_id: str = Field(description="Unique ID for this strategy execution")
    strategy_type: Literal["vector", "lexical", "graph", "hybrid"] = Field(description="Type of search to execute")
    query_variant: str = Field(description="Specific query phrasing for this strategy")
    index_weights: Dict[str, float] = Field(default_factory=dict, description="Weights for fusion if hybrid")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific filters")


# === ORCHESTRATOR OUTPUT (DSPy Signature Output) ===

class SearchPlan(BaseModel):
    """Structured output from the DSPy orchestrator"""

    primary_intent: Literal["Entity_Resolution", "Statute_Lookup", "Case_Law_Drilldown", "Broad_Research", "Clarification_Search"] = Field(
        description="The primary search strategy"
    )
    relational_pivot_hints: List[str] = Field(
        default_factory=list,
        description="High-confidence entities to pivot to (e.g. ['Richard Wang CEO', 'Stanford AI Lab'])"
    )
    hyde_passage: str = Field(
        description="Hypothetical ideal answer paragraph for vector alignment"
    )
    search_queries: List[str] = Field(
        description="3 distinct queries: [Lexical, Semantic, Relational]",
        min_length=3,
        max_length=3
    )
    filter_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters (e.g., {'org': 'Meta', 'year': '2024'})"
    )
    negative_constraints: List[str] = Field(
        default_factory=list,
        description="Entities/domains to EXCLUDE"
    )
    entity_graph_depth: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Depth of knowledge graph traversal"
    )
    strategies: List[StrategyConfig] = Field(
        default_factory=list,
        description="List of parallel search strategies to execute"
    )


# === AGENT STATE (LangGraph State) ===

class AgentState(TypedDict):
    """The central state object that flows through the LangGraph"""

    # INPUT
    query: str
    """User's raw question"""

    # PLANNING
    search_plan: SearchPlan | None
    """Current search plan from orchestrator"""
    
    entity_profile: str
    """Character Model (人物建模): Consolidated profile of the target entity"""

    # MEMORY (Accumulated across iterations)
    verified_facts: List[str]
    """Confirmed information from previous steps"""

    negative_cache: List[Dict[str, str]]
    """Failed paths: [{"entity": "Qian Chen", "reason": "Wrong Industry"}]"""

    known_doc_ids: Set[str]
    """Document IDs seen in previous iterations (for novelty calculation)"""

    # RETRIEVAL RESULTS (Merged via operator.add for parallel map-reduce)
    retrieved_docs: Annotated[List[SearchResult], operator.add]
    """All documents retrieved in current iteration"""

    # CONTROL FLOW
    step_count: int
    """Current iteration number"""

    novelty_score: float
    """Last calculated novelty score"""

    should_continue: bool
    """Circuit breaker status"""


# === MEMORY REFLECTION OUTPUT ===

class ReflectionOutput(BaseModel):
    """Output from the memory reflection node"""

    new_verified_facts: List[str] = Field(
        default_factory=list,
        description="New facts confirmed in this iteration"
    )
    new_negative_constraints: List[Dict[str, str]] = Field(
        default_factory=list,
        description="New failed paths to avoid"
    )
    context_summary: str = Field(
        default="",
        description="Summary of what we've learned so far"
    )
    new_entity_profile: str = Field(
        default="",
        description="人物建模: Updated character profile of the target entity"
    )


# === HELPER FUNCTIONS ===

def create_initial_state(query: str) -> AgentState:
    """Create the initial state for a new query"""
    return AgentState(
        query=query,
        search_plan=None,
        entity_profile="Not yet established.",
        verified_facts=[],
        negative_cache=[],
        known_doc_ids=set(),
        retrieved_docs=[],
        step_count=0,
        novelty_score=1.0,
        should_continue=True
    )
