"""
Main Entry Point: High-SNR Agentic RAG System
Orchestrates the entire pipeline from data ingestion to search
"""

import os
from typing import List, Dict, Any
import dspy

from core.config import get_config, update_config
from core.data_structures import SearchResult
from core.backends import create_best_backend, BackendFactory
from core.graph_engine import AgenticSearchEngine
from core.mock_data import get_mock_dataset
from core.orchestrator import initialize_orchestrator
from core.memory import initialize_memory_module


class RetrievalAgent:
    """
    Agentic Retrieval System with DSPy + LangGraph
    """

    def __init__(
        self,
        es_config: Dict[str, Any] | None = None,
        kg_config: Dict[str, Any] | None = None,
        backend_type: str | None = None,
        openai_api_key: str | None = None
    ):
        """
        Initialize the Retrieval Agent.

        Args:
            es_config: Elasticsearch configuration {"host": ..., "port": ..., "index": ...}
            kg_config: Knowledge graph configuration (for future use)
            backend_type: Force specific backend ("tri-index", "elasticsearch", "hybrid")
                         If None, auto-detects best backend
            openai_api_key: OpenAI API key for DSPy (uses env var if None)

        Example:
            # Auto-detect (uses tri-index if no ES config)
            agent = RetrievalAgent()

            # Use Elasticsearch
            agent = RetrievalAgent(es_config={"host": "...", "index": "..."})

            # Force tri-index even if ES is available
            agent = RetrievalAgent(backend_type="tri-index")
        """
        self.config = get_config()

        # Initialize DSPy (Support Anthropic or OpenAI)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if anthropic_key:
            # Model Priority List (Attempting "Opus 4.5" / Best Available)
            models_to_try = [
                'anthropic/claude-3-opus-20240229',
                'anthropic/claude-3-5-sonnet-20240620',
                'anthropic/claude-3-haiku-20240307',
                'claude-3-haiku-20240307'
            ]
            
            lm = None
            print("[System] Initializing Anthropic Model...")
            for model_id in models_to_try:
                try:
                    lm = dspy.LM(model=model_id, api_key=anthropic_key, max_tokens=1000)
                    # Test connection
                    lm("Hello")
                    print(f"  ✓ Loaded: {model_id}")
                    break
                except Exception as e:
                    print(f"  ✗ Failed {model_id}: {e}")
            
            if not lm:
                raise ValueError("Could not initialize any Anthropic model.")
                
            dspy.configure(lm=lm)
            
        elif openai_api_key or "OPENAI_API_KEY" in os.environ:
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            
            lm = dspy.OpenAI(model=self.config.LLM_MODEL, max_tokens=1000)
            dspy.settings.configure(lm=lm)
        else:
             raise ValueError("No API Key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

        # Create backend (this is the key to portability!)
        self.backend = create_best_backend(es_config, kg_config, backend_type)

        # Main search engine (will be initialized after data loading)
        self.search_engine = None

        print("[System] Retrieval Agent initialized")
        print(f"  Backend: {self.backend.get_stats()['backend_type']}")

    def load_data(self, documents: List[Dict[str, Any]] | None = None) -> None:
        """
        Load and index documents.

        Args:
            documents: List of documents to index (uses mock data if None)
        """
        if documents is None:
            print("[Data] Loading mock dataset...")
            documents = get_mock_dataset()

        print(f"[Data] Indexing {len(documents)} documents...")

        # Index using backend (works with ANY backend!)
        self.backend.index_documents(documents)

        # Show stats
        stats = self.backend.get_stats()
        print(f"  Backend stats: {stats}")

        # Initialize the search engine with backend
        # Note: AgenticSearchEngine now uses the backend internally
        from backends_wrapper import BackendAgenticSearchEngine
        self.search_engine = BackendAgenticSearchEngine(self.backend)

        print("[Data] Indexing complete")

    def search(self, query: str) -> List[SearchResult]:
        """
        Execute agentic search.

        Args:
            query: User's search query

        Returns:
            List of final search results
        """
        if self.search_engine is None:
            raise RuntimeError("Must call load_data() before search()")

        # Run the search
        final_state = self.search_engine.search(query)

        return final_state["retrieved_docs"]

    def run_qian_chen_test(self) -> Dict[str, Any]:
        """
        Run the canonical "Qian Chen" test case.

        Returns:
            Test results with metrics
        """
        print("\n" + "="*80)
        print("RUNNING QIAN CHEN DISAMBIGUATION TEST")
        print("="*80)

        query = "Did Qian Chen at Meta sign a non-compete agreement?"
        print(f"\nQuery: {query}")
        print("\nExpected: Should retrieve ONLY documents about Qian Chen (Meta Researcher)")
        print("Should AVOID: Qian Chen (Shanghai Lawyer), Qian Chen (Student)")

        results = self.search(query)

        # Analyze results
        print("\n" + "-"*80)
        print("RESULTS ANALYSIS")
        print("-"*80)

        meta_count = 0
        shanghai_count = 0
        student_count = 0
        other_count = 0

        for i, result in enumerate(results):
            org = result.metadata.get("org", "Unknown")

            print(f"\n[Result {i+1}] Score: {result.score:.3f}")
            print(f"  Organization: {org}")
            print(f"  Type: {result.metadata.get('type', 'Unknown')}")
            print(f"  ID: {result.id}")
            print(f"  Content preview: {result.content[:150]}...")

            if org == "Meta":
                meta_count += 1
            elif "Shanghai" in org:
                shanghai_count += 1
            elif org == "UC Berkeley":
                student_count += 1
            else:
                other_count += 1

        # Calculate precision
        total = len(results)
        precision = meta_count / total if total > 0 else 0

        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Total results: {total}")
        print(f"Meta (target): {meta_count}")
        print(f"Shanghai (distractor): {shanghai_count}")
        print(f"Student (distractor): {student_count}")
        print(f"Other: {other_count}")
        print(f"\nPrecision: {precision:.1%}")

        if precision == 1.0:
            print("✓ TEST PASSED - Perfect disambiguation!")
        elif precision >= 0.8:
            print("~ TEST PARTIAL - Good but not perfect")
        else:
            print("✗ TEST FAILED - Poor disambiguation")

        return {
            "total_results": total,
            "meta_count": meta_count,
            "shanghai_count": shanghai_count,
            "student_count": student_count,
            "precision": precision,
            "passed": precision >= 0.8
        }

    def run_ablation_tests(self) -> Dict[str, Any]:
        """
        Run ablation tests to prove the value of each component.

        Returns:
            Ablation test results
        """
        print("\n" + "="*80)
        print("RUNNING ABLATION TESTS")
        print("="*80)

        query = "Did Qian Chen at Meta sign a non-compete agreement?"
        baseline_config = {
            "USE_DSPY_SIGNATURES": True,
            "USE_NOVELTY_CIRCUIT": True,
            "USE_NEGATIVE_MEMORY": True,
            "USE_CASCADE_RECALL": True
        }

        results_summary = {}

        # Test 1: Full system (baseline)
        print("\n[Test 1] Full System (Baseline)")
        for key, value in baseline_config.items():
            update_config(**{key: value})
        baseline_results = self.search(query)
        results_summary["baseline"] = {
            "count": len(baseline_results),
            "precision": self._calculate_precision(baseline_results)
        }

        # Test 2: Without DSPy signatures
        print("\n[Test 2] Without DSPy Signatures")
        update_config(USE_DSPY_SIGNATURES=False)
        no_dspy_results = self.search(query)
        results_summary["no_dspy"] = {
            "count": len(no_dspy_results),
            "precision": self._calculate_precision(no_dspy_results)
        }
        update_config(USE_DSPY_SIGNATURES=True)

        # Test 3: Without novelty circuit
        print("\n[Test 3] Without Novelty Circuit")
        update_config(USE_NOVELTY_CIRCUIT=False)
        no_circuit_results = self.search(query)
        results_summary["no_circuit"] = {
            "count": len(no_circuit_results),
            "precision": self._calculate_precision(no_circuit_results)
        }
        update_config(USE_NOVELTY_CIRCUIT=True)

        # Test 4: Without negative memory
        print("\n[Test 4] Without Negative Memory")
        update_config(USE_NEGATIVE_MEMORY=False)
        no_memory_results = self.search(query)
        results_summary["no_memory"] = {
            "count": len(no_memory_results),
            "precision": self._calculate_precision(no_memory_results)
        }
        update_config(USE_NEGATIVE_MEMORY=True)

        # Reset to baseline
        for key, value in baseline_config.items():
            update_config(**{key: value})

        # Print summary
        print("\n" + "="*80)
        print("ABLATION TEST SUMMARY")
        print("="*80)
        for test_name, metrics in results_summary.items():
            print(f"{test_name:20s}: Precision={metrics['precision']:.1%}, Count={metrics['count']}")

        return results_summary

    def _calculate_precision(self, results: List[SearchResult]) -> float:
        """Calculate precision (% of results that are Meta-related)"""
        if not results:
            return 0.0

        meta_count = sum(1 for r in results if r.metadata.get("org") == "Meta")
        return meta_count / len(results)


def main():
    """
    Main entry point for testing the system.
    """
    import sys

    # Check for API key
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: API KEY not found.")
        print("Set ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY")
        sys.exit(1)

    # Initialize the system
    print("Initializing Retrieval Agent...")
    agent = RetrievalAgent()

    # Load mock data
    agent.load_data()

    # Run the Qian Chen test
    test_results = agent.run_qian_chen_test()

    # Optionally run ablation tests
    if "--ablation" in sys.argv:
        print("\n\nRunning ablation tests...")
        ablation_results = agent.run_ablation_tests()

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
