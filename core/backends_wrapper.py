"""
Backend Wrapper for LangGraph AgenticSearchEngine

This adapts the LangGraph state machine to work with ANY backend.
"""

from typing import List, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.types import Send
# from langgraph.graph.graph import CompiledGraph

from .data_structures import AgentState, SearchResult, create_initial_state, StrategyConfig, SearchPlan
from orchestrator import LegalOrchestrator
from .interfaces import SearchBackend
from .verifier import NoveltyVerifier
from .memory import MemoryEvolutionModule
from .config import get_config


class BackendAgenticSearchEngine:
    """
    LangGraph-based search engine that works with ANY backend.

    This is backend-agnostic - it works with tri-index, ES, or custom backends.
    """

    def __init__(self, backend: SearchBackend):
        """
        Initialize the agentic search engine.

        Args:
            backend: Any SearchBackend implementation
        """
        self.config = get_config()
        self.backend = backend

        # Components (backend-agnostic)
        self.orchestrator = LegalOrchestrator()
        self.verifier = NoveltyVerifier()
        self.memory = MemoryEvolutionModule()

        # Build the graph
        self.app = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine"""
        workflow = StateGraph(AgentState)

        # Node definitions
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("execute_strategy", self._execute_strategy_node)
        # We need a reducer/aggregator node if we want to dedup before verify, 
        # but verifies work on state["retrieved_docs"] which is already aggregated by operator.add.
        # However, a dedicated dedup step is good practice.
        workflow.add_node("dedup", self._dedup_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("reflect", self._reflect_node)

        # Edge definitions
        workflow.set_entry_point("orchestrator")
        
        # Conditional edge: Orchestrator -> Parallel Strategies
        workflow.add_conditional_edges(
            "orchestrator",
            self._trigger_strategies,
            ["execute_strategy"]
        )

        # Fan-in: All strategies -> Dedup -> Verify
        workflow.add_edge("execute_strategy", "dedup")
        workflow.add_edge("dedup", "verify")

        # Conditional edge from verify
        workflow.add_conditional_edges(
            "verify",
            self._should_continue,
            {
                "continue": "reflect",
                "stop": END
            }
        )

        # Loop back
        workflow.add_edge("reflect", "orchestrator")

        return workflow.compile()

    # === NODE IMPLEMENTATIONS ===

    def _orchestrator_node(self, state: AgentState) -> AgentState:
        """Generate search plan"""
        if self.config.DEBUG_MODE:
            print(f"\n[Orchestrator] Step {state['step_count'] + 1}")

        search_plan = self.orchestrator.forward(state)
        state["search_plan"] = search_plan
        state["step_count"] += 1

        if self.config.DEBUG_MODE:
            print(f"  Intent: {search_plan.primary_intent}")
            print(f"  Filters: {search_plan.filter_constraints}")

        return state

    def _execute_strategy_node(self, config: StrategyConfig) -> Dict[str, Any]:
        """
        Worker Node: Execute a single strategy.
        Accepts StrategyConfig, returns AgentState update.
        """
        if self.config.DEBUG_MODE:
            print(f"  [Strategy: {config.strategy_type}] Executing: {config.query_variant[:50]}...")

        # Adapt StrategyConfig to SearchPlan (for legacy backend compatibility)
        # This allows us to use existing SearchBackend.search() without changing interfaces
        mini_plan = SearchPlan(
            primary_intent="Broad_Research", # Ignored by most granular/hybrid backends if we craft queries right
            hyde_passage=config.query_variant, # Used by Vector
            search_queries=[config.query_variant] * 3, # Used by Lexical
            filter_constraints=config.filters,
            negative_constraints=[],
            entity_graph_depth=1,
            strategies=[] # Prevent recursion
        )

        # Execute
        # Note: Ideally backends should implement execute_strategy, but this adapter works for now.
        # We need to pass the query variants somewhat hackily via the plan or query arg.
        results = self.backend.search(
            query=config.query_variant, 
            search_plan=mini_plan,
            negative_cache=[] # Sub-strategies don't need neg cache, orchestrator handles it? Or pass it?
                              # Actually, filters handle neg constraints.
        )
        
        # Tag results with the strategy type for debugging/weighting
        for r in results:
            r.metadata["strategy_source"] = config.strategy_type

        return {"retrieved_docs": results}

    def _dedup_node(self, state: AgentState) -> AgentState:
        """Merge and deduplicate results from parallel branches"""
        raw_docs = state["retrieved_docs"]
        unique_docs = {}
        
        for doc in raw_docs:
            if doc.id not in unique_docs:
                unique_docs[doc.id] = doc
            else:
                # Keep the one with higher score? Or merge metadata?
                if doc.score > unique_docs[doc.id].score:
                    unique_docs[doc.id] = doc
        
        deduped = list(unique_docs.values())
        # Sort by score desc
        deduped.sort(key=lambda x: x.score, reverse=True)
        
        if self.config.DEBUG_MODE:
            print(f"[Dedup] Merged {len(raw_docs)} -> {len(deduped)} unique docs")
            
        state["retrieved_docs"] = deduped
        return state

    def _verify_node(self, state: AgentState) -> AgentState:
        """Verify novelty and update circuit breaker"""
        state = self.verifier.verify_and_update_state(state)
        return state

    def _reflect_node(self, state: AgentState) -> AgentState:
        """Memory reflection"""
        if self.config.DEBUG_MODE:
            print(f"\n[Reflect] Analyzing results...")

        reflection = self.memory.reflect(state)
        state = self.memory.update_state_memory(state, reflection)

        if self.config.DEBUG_MODE:
            print(f"  Verified facts: {len(state['verified_facts'])}")
            print(f"  Negative cache: {len(state['negative_cache'])}")

        # Clear retrieved_docs for next iteration
        state["retrieved_docs"] = []

        return state

    # === CONDITIONAL EDGE ===

    def _trigger_strategies(self, state: AgentState) -> List[Send]:
        """Trigger parallel strategies using Send API"""
        plan = state["search_plan"]
        if not plan or not plan.strategies:
            if self.config.DEBUG_MODE:
                print("  No strategies found, skipping search.")
            return []
            
        if self.config.DEBUG_MODE:
            print(f"\n[Map] Spawning {len(plan.strategies)} strategies...")

        return [Send("execute_strategy", s) for s in plan.strategies]

    def _should_continue(self, state: AgentState) -> str:
        """Determine if search should continue"""
        if state["should_continue"]:
            return "continue"
        else:
            return "stop"

    # === PUBLIC API ===

    def search(self, query: str) -> AgentState:
        """
        Execute the agentic search.

        Args:
            query: User's search query

        Returns:
            Final agent state with results
        """
        # Create initial state
        initial_state = create_initial_state(query)

        if self.config.DEBUG_MODE:
            print(f"\n{'='*60}")
            print(f"Starting Agentic Search")
            print(f"Query: {query}")
            print(f"Backend: {self.backend.get_stats()['backend_type']}")
            print(f"{'='*60}")

        # Run the graph
        final_state = self.app.invoke(initial_state)

        if self.config.DEBUG_MODE:
            print(f"\n{'='*60}")
            print(f"Search Complete")
            print(f"Total iterations: {final_state['step_count']}")
            print(f"Final results: {len(final_state['retrieved_docs'])}")
            print(f"{'='*60}\n")

        return final_state
