"""
LangGraph State Machine: The "Control Layer"
Implements the cyclic deterministic graph with parallel map-reduce
"""

from typing import List, Literal, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Send
# from langgraph.graph.graph import CompiledGraph

from .data_structures import AgentState, SearchResult, create_initial_state
from orchestrator import LegalOrchestrator
from retrieval import VectorSearchEngine, BM25SearchEngine, GraphSearchEngine, CascadeRecallFunnel
from .verifier import NoveltyVerifier
from .memory import MemoryEvolutionModule
from .skills.bridging import BridgeSearchSkill
from .config import get_config


class AgenticSearchEngine:
    """
    Main search engine using LangGraph for state orchestration.
    Implements the Cyclic Deterministic Graph architecture.
    """

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        bm25_engine: BM25SearchEngine,
        graph_engine: GraphSearchEngine
    ):
        """
        Initialize the agentic search engine.

        Args:
            vector_engine: Vector search engine instance
            bm25_engine: BM25 search engine instance
            graph_engine: Graph search engine instance
        """
        self.config = get_config()

        # Components
        self.orchestrator = LegalOrchestrator()
        self.vector_engine = vector_engine
        self.bm25_engine = bm25_engine
        self.graph_engine = graph_engine
        self.cascade = CascadeRecallFunnel()
        self.verifier = NoveltyVerifier()
        self.memory = MemoryEvolutionModule()
        self.bridge_skill = BridgeSearchSkill()

        # Build the graph
        self.app = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine"""
        workflow = StateGraph(AgentState)

        # === NODE DEFINITIONS ===

        # Node 1: Orchestrator (Plan)
        workflow.add_node("orchestrator", self._orchestrator_node)

        # Node 2: Parallel Search Nodes
        workflow.add_node("vector_search", self._vector_search_node)
        workflow.add_node("bm25_search", self._bm25_search_node)
        workflow.add_node("graph_search", self._graph_search_node)
        workflow.add_node("bridge_search", self._bridge_search_node)

        # Node 3: Aggregation & Filtering
        workflow.add_node("aggregate", self._aggregate_node)

        # Node 4: Verification (Circuit Breaker)
        workflow.add_node("verify", self._verify_node)

        # Node 5: Memory Reflection
        workflow.add_node("reflect", self._reflect_node)

        # === EDGE DEFINITIONS ===

        # Entry point
        workflow.set_entry_point("orchestrator")

        # Orchestrator -> Parallel Search (using Send API)
        workflow.add_conditional_edges(
            "orchestrator",
            self._trigger_parallel_search,
            ["vector_search", "bm25_search", "graph_search", "bridge_search"]
        )

        # All search nodes -> Aggregate
        workflow.add_edge("vector_search", "aggregate")
        workflow.add_edge("bm25_search", "aggregate")
        workflow.add_edge("graph_search", "aggregate")
        workflow.add_edge("bridge_search", "aggregate")

        # Aggregate -> Verify
        workflow.add_edge("aggregate", "verify")

        # Verify -> Reflect or END (conditional)
        workflow.add_conditional_edges(
            "verify",
            self._should_continue,
            {
                "continue": "reflect",
                "stop": END
            }
        )

        # Reflect -> Orchestrator (loop back)
        workflow.add_edge("reflect", "orchestrator")

        # Compile
        return workflow.compile()

    # === NODE IMPLEMENTATIONS ===

    def _orchestrator_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Generate search plan using DSPy.
        """
        if self.config.DEBUG_MODE:
            print(f"\n[Orchestrator] Step {state['step_count'] + 1}")

        # Generate plan
        search_plan = self.orchestrator.forward(state)
        state["search_plan"] = search_plan
        state["step_count"] += 1

        if self.config.DEBUG_MODE:
            print(f"  Intent: {search_plan.primary_intent}")
            print(f"  Graph Depth: {search_plan.entity_graph_depth}")
            print(f"  Filters: {search_plan.filter_constraints}")
            print(f"  Negative Constraints: {search_plan.negative_constraints}")

        return state

    def _vector_search_node(self, state: AgentState) -> AgentState:
        """Node 2a: Vector search"""
        plan = state["search_plan"]
        if plan is None:
            return {"retrieved_docs": []}

        # Use HyDE passage for vector search
        results = self.vector_engine.search(
            query=plan.hyde_passage,
            filter_constraints=plan.filter_constraints
        )

        if self.config.DEBUG_MODE:
            print(f"[Vector Search] Retrieved {len(results)} documents")

        return {"retrieved_docs": results}

    def _bm25_search_node(self, state: AgentState) -> AgentState:
        """Node 2b: BM25 search"""
        plan = state["search_plan"]
        if plan is None:
            return {"retrieved_docs": []}

        # Use lexical query for BM25
        lexical_query = plan.search_queries[0] if plan.search_queries else state["query"]

        results = self.bm25_engine.search(
            query=lexical_query,
            filter_constraints=plan.filter_constraints
        )

        if self.config.DEBUG_MODE:
            print(f"[BM25 Search] Retrieved {len(results)} documents")

        return {"retrieved_docs": results}

    def _graph_search_node(self, state: AgentState) -> AgentState:
        """Node 2c: Graph traversal search"""
        plan = state["search_plan"]
        if plan is None:
            return {"retrieved_docs": []}

        # Extract entities from relational query OR use pivot hints
        relational_query = plan.search_queries[2] if len(plan.search_queries) > 2 else state["query"]
        pivot_entities = plan.relational_pivot_hints if plan.relational_pivot_hints else []
        
        # Combine extracted entities with pivot hints
        query_entities = self._extract_entities(relational_query)
        if pivot_entities:
            query_entities = list(set(query_entities + pivot_entities))

        results = self.graph_engine.search(
            query_entities=query_entities,
            depth=plan.entity_graph_depth,
            filter_constraints=plan.filter_constraints
        )

        if self.config.DEBUG_MODE:
            print(f"[Graph Search] Retrieved {len(results)} documents")

        return {"retrieved_docs": results}

    def _bridge_search_node(self, state: AgentState) -> AgentState:
        """Node 2d: Bridge Search Skill (Deterministic Disambiguation)"""
        plan = state["search_plan"]
        
        # Check if bridge strategy is active or needed
        bridge_needed = self.bridge_skill.detect_bridge_need(state["query"], [])
        
        if not bridge_needed and not any(s.strategy_type == "bridge" for s in plan.strategies):
             return {"retrieved_docs": []}

        if self.config.DEBUG_MODE:
            print(f"[Bridge Search] Bridging strategy activated.")

        # Logic: 
        # 1. Search Anchor (CEO/Org) -> using Vector Engine as backend for now
        # 2. Extract Constraints
        # 3. Search Target
        
        # For now, we simulate the output or use the implementation from the skill if it had full backend access.
        # Since BridgeSearchSkill in this phase is a "Logic Unit", we implement the execution flow here using self.vector_engine
        
        anchors = self.bridge_skill.suggest_anchors(state["query"], "")
        anchor_docs = []
        for anchor_q in anchors[:1]: # Try primary anchor
            anchor_docs.extend(self.vector_engine.search(anchor_q, filter_constraints=None)[:3])
            
        constraints = self.bridge_skill.extract_constraints_from_docs(anchor_docs)
        
        constrained_query = self.bridge_skill.generate_constrained_query(state["query"], constraints)
        
        # Execute Final Target Search
        results = self.vector_engine.search(constrained_query, filter_constraints=plan.filter_constraints)
        
        # Tag results
        for r in results:
            r.source_index = "bridge"
            
        if self.config.DEBUG_MODE:
             print(f"[Bridge Search] Discovered constraints: {constraints}. Retrieved {len(results)} target docs.")

        return {"retrieved_docs": results}

    def _aggregate_node(self, state: AgentState) -> AgentState:
        """
        Node 3: Aggregate results and apply cascade filtering.
        """
        # Separate results by source
        vector_results = [r for r in state["retrieved_docs"] if r.source_index == "vector"]
        bm25_results = [r for r in state["retrieved_docs"] if r.source_index == "bm25"]
        vector_results = [r for r in state["retrieved_docs"] if r.source_index == "vector"]
        bm25_results = [r for r in state["retrieved_docs"] if r.source_index == "bm25"]
        graph_results = [r for r in state["retrieved_docs"] if r.source_index == "graph"]
        bridge_results = [r for r in state["retrieved_docs"] if r.source_index == "bridge"]

        if self.config.DEBUG_MODE:
            print(f"\n[Aggregate] Processing results:")
            print(f"  Vector: {len(vector_results)}, BM25: {len(bm25_results)}, Graph: {len(graph_results)}, Bridge: {len(bridge_results)}")

        # Apply cascade funnel
        plan = state["search_plan"]
        filtered = self.cascade.aggregate_and_filter(
            vector_results,
            bm25_results,
            vector_results,
            bm25_results,
            graph_results + bridge_results, # Treat bridge results as high-value graph results for now
            plan,
            state["negative_cache"]
        )

        # Rerank
        reranked = self.cascade.rerank(
            filtered,
            state["query"],
            plan.hyde_passage
        )

        # Update state with final results
        state["retrieved_docs"] = reranked

        if self.config.DEBUG_MODE:
            print(f"[Aggregate] Final: {len(reranked)} documents after cascade")

        return state

    def _verify_node(self, state: AgentState) -> AgentState:
        """
        Node 4: Verify novelty and update circuit breaker.
        """
        state = self.verifier.verify_and_update_state(state)
        return state

    def _reflect_node(self, state: AgentState) -> AgentState:
        """
        Node 5: Memory reflection and learning.
        """
        if self.config.DEBUG_MODE:
            print(f"\n[Reflect] Analyzing results...")

        # Run reflection
        reflection = self.memory.reflect(state)

        # Update state
        state = self.memory.update_state_memory(state, reflection)

        if self.config.DEBUG_MODE:
            print(f"  Verified facts: {len(state['verified_facts'])}")
            print(f"  Negative cache: {len(state['negative_cache'])}")

        # Clear retrieved_docs for next iteration
        state["retrieved_docs"] = []

        return state

    # === CONDITIONAL EDGE FUNCTIONS ===

    def _trigger_parallel_search(self, state: AgentState) -> List[Send]:
        """
        Conditional edge: Trigger parallel search across all indices.
        Uses the Send() API for true parallelism.
        """
        return [
            Send("vector_search", state),
            Send("bm25_search", state),
            Send("bm25_search", state),
            Send("graph_search", state),
            Send("bridge_search", state)
        ]

    def _should_continue(self, state: AgentState) -> Literal["continue", "stop"]:
        """
        Conditional edge: Determine if search should continue.
        """
        if state["should_continue"]:
            return "continue"
        else:
            return "stop"

    # === HELPER FUNCTIONS ===

    def _extract_entities(self, text: str) -> List[str]:
        """
        Simple entity extraction (can be enhanced with NER).
        """
        # For now, split by common separators and filter
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return words if words else [text]

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
