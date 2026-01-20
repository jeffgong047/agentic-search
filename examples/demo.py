"""
DEMO: Retrieval Agent with Real Infrastructure

This demonstrates how to:
1. Connect to Elasticsearch and PostgreSQL
2. Load real data (or use mock data)
3. Run the agent
4. Return results in any format your mentor needs
"""

import os
import json
from typing import List, Dict, Any

from main import RetrievalAgent
from tool_adapters import ElasticsearchAdapter, PostgreSQLGraphAdapter
from mock_data import get_mock_dataset


# ====================================================================
# DEMO 1: Using Local Docker Services
# ====================================================================

def demo_local_services():
    """
    Demo with local Elasticsearch and PostgreSQL.

    Prerequisites:
        docker-compose up -d elasticsearch postgres

    This shows the agent working with real services on your laptop.
    """
    print("\n" + "="*80)
    print("DEMO 1: Local Services (Elasticsearch + PostgreSQL)")
    print("="*80)

    # Initialize agent with local services
    agent = RetrievalAgent(
        es_config={
            "host": "localhost",
            "port": 9200,
            "index": "legal_docs"
        },
        kg_config={
            "host": "localhost",
            "database": "knowledge_graph",
            "user": "retrieval",
            "password": "retrieval123"
        }
    )

    # Load mock data (replace with real data from your mentor)
    print("\nLoading mock data...")
    documents = get_mock_dataset()
    agent.load_data(documents)

    # Run search
    print("\nRunning search...")
    query = "Did Qian Chen at Meta sign a non-compete agreement?"
    results = agent.search(query)

    # Display results
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result.id}")
        print(f"    Score: {result.score:.3f}")
        print(f"    Org: {result.metadata.get('org', 'N/A')}")
        print(f"    Preview: {result.content[:100]}...")

    return results


# ====================================================================
# DEMO 2: Using Mentor's Production Services
# ====================================================================

def demo_production_services():
    """
    Demo with your mentor's actual infrastructure.

    IMPORTANT: Replace these with real values from your mentor!
    """
    print("\n" + "="*80)
    print("DEMO 2: Production Services (Mentor's Infrastructure)")
    print("="*80)

    # TODO: Get these from your mentor
    ES_CONFIG = {
        "host": "evenup-es-prod.internal",  # ← Replace with real host
        "port": 9200,
        "index": "case_documents",  # ← Replace with real index
        # Optional: Add authentication
        # "http_auth": ("username", "password"),
        # "use_ssl": True
    }

    KG_CONFIG = {
        "host": "evenup-kg-prod.internal",  # ← Replace with real host
        "database": "knowledge_graph",  # ← Replace with real database
        "user": "retrieval_user",  # ← Replace with real user
        "password": os.getenv("KG_PASSWORD", "changeme")  # ← Use env var
    }

    # Initialize agent
    agent = RetrievalAgent(
        es_config=ES_CONFIG,
        kg_config=KG_CONFIG
    )

    # Note: With production services, you don't call load_data()
    # The data is already in Elasticsearch!

    # Run search
    query = "Your test query here"
    results = agent.search(query)

    print(f"\nFound {len(results)} results from production")
    return results


# ====================================================================
# DEMO 3: Using In-Memory Indexes (No External Services)
# ====================================================================

def demo_in_memory():
    """
    Demo without any external services.

    Uses built-in FAISS, BM25, and NetworkX.
    Perfect for development and testing.
    """
    print("\n" + "="*80)
    print("DEMO 3: In-Memory Indexes (No External Services)")
    print("="*80)

    # Initialize agent WITHOUT es_config or kg_config
    agent = RetrievalAgent()

    # Load data
    print("\nLoading mock data...")
    agent.load_data()

    # Run search
    print("\nRunning search...")
    query = "Did Qian Chen at Meta sign a non-compete agreement?"
    results = agent.search(query)

    print(f"\nFound {len(results)} results (in-memory)")
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result.metadata.get('org', 'N/A')}: {result.score:.3f}")

    return results


# ====================================================================
# DEMO 4: Loading Custom Documents
# ====================================================================

def demo_custom_documents():
    """
    Demo showing how to load your own documents.

    This is what you'll do when you get real data from your mentor.
    """
    print("\n" + "="*80)
    print("DEMO 4: Loading Custom Documents")
    print("="*80)

    # Your custom documents (e.g., from mentor's dataset)
    documents = [
        {
            "id": "case_12345_depo_1",
            "content": """
            Deposition of John Doe in Case #12345.
            Plaintiff sustained spinal injuries in accident.
            Seeking damages of $500,000.
            Jurisdiction: California Superior Court.
            """,
            "metadata": {
                "case_id": "12345",
                "doc_type": "deposition",
                "plaintiff": "John Doe",
                "injury_type": "spinal",
                "jurisdiction": "California",
                "entities": [
                    {"id": "john_doe", "name": "John Doe", "type": "plaintiff"},
                    {"id": "case_12345", "name": "Case 12345", "type": "case"}
                ],
                "relations": [
                    {"source": "john_doe", "target": "case_12345", "type": "plaintiff_in"}
                ]
            }
        },
        {
            "id": "case_12345_demand_1",
            "content": """
            Demand letter for Case #12345.
            Plaintiff John Doe demands $500,000 for spinal injuries.
            Similar cases in California have settled for $400K-$600K.
            """,
            "metadata": {
                "case_id": "12345",
                "doc_type": "demand_letter",
                "plaintiff": "John Doe",
                "injury_type": "spinal",
                "settlement_demand": 500000,
                "jurisdiction": "California",
                "entities": [
                    {"id": "john_doe", "name": "John Doe", "type": "plaintiff"}
                ],
                "relations": []
            }
        }
    ]

    # Initialize agent
    agent = RetrievalAgent()

    # Load custom documents
    agent.load_data(documents)

    # Search
    query = "Find spinal injury cases with high settlement demands"
    results = agent.search(query)

    print(f"\nFound {len(results)} results")
    for result in results:
        print(f"\n{result.id}")
        print(f"  Type: {result.metadata.get('doc_type')}")
        print(f"  Injury: {result.metadata.get('injury_type')}")
        print(f"  Score: {result.score:.3f}")

    return results


# ====================================================================
# DEMO 5: Formatting Results for Mentor's Test Scaffold
# ====================================================================

def demo_result_formatting():
    """
    Demo showing different output formats.

    Adapt this to match whatever format your mentor's evaluation scripts expect.
    """
    print("\n" + "="*80)
    print("DEMO 5: Result Formatting")
    print("="*80)

    agent = RetrievalAgent()
    agent.load_data()

    query = "Qian Chen Meta non-compete"
    results = agent.search(query)

    # Format 1: List of doc IDs only
    format_1 = [r.id for r in results]
    print("\nFormat 1 (IDs only):")
    print(json.dumps(format_1[:3], indent=2))

    # Format 2: ID + Score tuples
    format_2 = [(r.id, r.score) for r in results]
    print("\nFormat 2 (ID + Score):")
    print(json.dumps(format_2[:3], indent=2))

    # Format 3: Full dict
    format_3 = [
        {
            "doc_id": r.id,
            "score": r.score,
            "content": r.content[:200],
            "metadata": r.metadata
        }
        for r in results
    ]
    print("\nFormat 3 (Full dict):")
    print(json.dumps(format_3[:1], indent=2))

    # Format 4: TREC format (common in IR evaluation)
    format_4_lines = []
    query_id = "q001"
    for rank, result in enumerate(results, 1):
        # Format: query_id Q0 doc_id rank score run_name
        format_4_lines.append(f"{query_id} Q0 {result.id} {rank} {result.score:.6f} retrieval_agent")

    print("\nFormat 4 (TREC):")
    print("\n".join(format_4_lines[:3]))

    # Format 5: JSON with query metadata
    format_5 = {
        "query_id": "q001",
        "query_text": query,
        "num_results": len(results),
        "results": [
            {"rank": i+1, "doc_id": r.id, "score": r.score}
            for i, r in enumerate(results)
        ]
    }
    print("\nFormat 5 (Eval-ready JSON):")
    print(json.dumps(format_5, indent=2))

    return format_5


# ====================================================================
# DEMO 6: Integration Wrapper for Mentor's Test Scaffold
# ====================================================================

class MentorTestInterface:
    """
    Wrapper that adapts the agent to your mentor's test framework.

    Your mentor calls this, and it handles all the translation.
    """

    def __init__(self, es_config: Dict | None = None, kg_config: Dict | None = None):
        """Initialize with mentor's service configs"""
        self.agent = RetrievalAgent(es_config=es_config, kg_config=kg_config)
        self.agent_initialized = False

    def load_documents(self, documents: List[Dict[str, Any]]):
        """Load documents if needed"""
        self.agent.load_data(documents)
        self.agent_initialized = True

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search interface matching mentor's expected format.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dicts in mentor's expected format
        """
        if not self.agent_initialized:
            raise RuntimeError("Must call load_documents() first")

        # Run agent search
        results = self.agent.search(query)

        # Return in mentor's format (adjust as needed)
        return [
            {
                "doc_id": r.id,
                "score": float(r.score),  # Ensure it's a Python float, not numpy
                "rank": i + 1
            }
            for i, r in enumerate(results[:top_k])
        ]

    def batch_search(self, queries: List[Dict[str, str]]) -> Dict[str, List[Dict]]:
        """
        Run multiple queries (for test scaffolds).

        Args:
            queries: List of {"query_id": ..., "query_text": ...}

        Returns:
            Dict mapping query_id -> results
        """
        all_results = {}

        for query_dict in queries:
            query_id = query_dict["query_id"]
            query_text = query_dict["query_text"]

            results = self.search(query_text)
            all_results[query_id] = results

        return all_results


def demo_mentor_interface():
    """Demo the mentor test interface"""
    print("\n" + "="*80)
    print("DEMO 6: Mentor Test Interface")
    print("="*80)

    # Initialize interface
    interface = MentorTestInterface()

    # Load data
    interface.load_documents(get_mock_dataset())

    # Single query
    results = interface.search("Qian Chen Meta", top_k=5)
    print(f"\nSingle query returned {len(results)} results")
    print(json.dumps(results[:2], indent=2))

    # Batch queries
    test_queries = [
        {"query_id": "q001", "query_text": "Qian Chen Meta non-compete"},
        {"query_id": "q002", "query_text": "California non-compete law"}
    ]

    batch_results = interface.batch_search(test_queries)
    print(f"\nBatch query returned results for {len(batch_results)} queries")
    for query_id, results in batch_results.items():
        print(f"  {query_id}: {len(results)} results")


# ====================================================================
# MAIN
# ====================================================================

def main():
    """
    Run demos.

    Usage:
        python demo.py                    # Run in-memory demo
        python demo.py --local            # Use local docker services
        python demo.py --production       # Use mentor's services
    """
    import sys

    # Check API key
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'\n")

    mode = sys.argv[1] if len(sys.argv) > 1 else "memory"

    if mode == "--local":
        print("Running with local Docker services...")
        print("Make sure you ran: docker-compose up -d\n")
        demo_local_services()

    elif mode == "--production":
        print("Running with production services...")
        print("Make sure you updated ES_CONFIG and KG_CONFIG in demo.py!\n")
        demo_production_services()

    elif mode == "--custom":
        demo_custom_documents()

    elif mode == "--format":
        demo_result_formatting()

    elif mode == "--interface":
        demo_mentor_interface()

    else:
        # Default: in-memory demo
        demo_in_memory()

    print("\n" + "="*80)
    print("✅ Demo complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Update ES_CONFIG and KG_CONFIG with real values from your mentor")
    print("2. Run: python demo.py --production")
    print("3. Verify results match expected format")
    print("4. Share this demo.py with your mentor\n")


if __name__ == "__main__":
    main()
