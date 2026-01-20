"""
Example Usage: Retrieval Agent
Demonstrates basic usage and advanced features
"""

import os
from main import RetrievalAgent
from config import update_config


def example_1_basic_search():
    """Example 1: Basic search with mock data"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Search")
    print("="*80)

    # Initialize agent
    agent = RetrievalAgent()

    # Load mock data
    agent.load_data()

    # Run search
    query = "Did Qian Chen at Meta sign a non-compete agreement?"
    print(f"\nQuery: {query}\n")

    results = agent.search(query)

    # Display results
    print(f"\nFound {len(results)} results:\n")
    for i, result in enumerate(results):
        print(f"[{i+1}] {result.metadata.get('type', 'Unknown Type')}")
        print(f"    Org: {result.metadata.get('org', 'Unknown')}")
        print(f"    Score: {result.score:.3f}")
        print(f"    Preview: {result.content[:150].strip()}...")
        print()


def example_2_custom_configuration():
    """Example 2: Custom configuration"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80)

    # Customize settings
    update_config(
        NOVELTY_EPSILON=0.15,  # Lower threshold (more iterations)
        FINAL_TOP_K=3,         # Fewer final results
        DEBUG_MODE=True        # Enable debug logging
    )

    agent = RetrievalAgent()
    agent.load_data()

    query = "Non-compete agreements in California"
    results = agent.search(query)

    print(f"\nReturned {len(results)} results with custom config")


def example_3_custom_data():
    """Example 3: Using custom data"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Data Ingestion")
    print("="*80)

    # Define custom documents
    custom_docs = [
        {
            "id": "custom_doc_1",
            "content": """
            Tech Company Employment Agreement

            Employee: Jane Smith
            Company: TechCorp Inc.
            Position: Senior Engineer

            This agreement includes standard IP assignment clauses and
            confidentiality provisions. California law prohibits non-compete
            agreements, so this contract focuses on trade secret protection.
            """,
            "metadata": {
                "org": "TechCorp",
                "year": 2024,
                "type": "Employment Agreement",
                "entities": [
                    {"id": "jane_smith", "name": "Jane Smith", "type": "person"},
                    {"id": "techcorp", "name": "TechCorp Inc.", "type": "organization"}
                ],
                "relations": [
                    {"source": "jane_smith", "target": "techcorp", "type": "employed_by"}
                ]
            }
        },
        {
            "id": "custom_doc_2",
            "content": """
            California Business and Professions Code Section 16600

            Except as provided in this chapter, every contract by which anyone
            is restrained from engaging in a lawful profession, trade, or business
            of any kind is to that extent void.

            This statute generally prohibits non-compete agreements in California,
            with limited exceptions for sale of business or dissolution of partnership.
            """,
            "metadata": {
                "org": "California Legislature",
                "year": 2024,
                "type": "Statute",
                "jurisdiction": "California",
                "entities": []
            }
        }
    ]

    # Load custom data
    agent = RetrievalAgent()
    agent.load_data(documents=custom_docs)

    # Search
    query = "Jane Smith employment agreement"
    results = agent.search(query)

    print(f"\nFound {len(results)} results in custom dataset")
    for result in results:
        print(f"- {result.metadata.get('type')}: {result.id}")


def example_4_knowledge_graph_exploration():
    """Example 4: Knowledge graph exploration"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Knowledge Graph Exploration")
    print("="*80)

    agent = RetrievalAgent()
    agent.load_data()

    # Explore knowledge graph
    kg = agent.knowledge_graph

    # Get graph statistics
    stats = kg.get_graph_stats()
    print("\nKnowledge Graph Statistics:")
    print(f"  Entities: {stats['num_entities']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Entity Types: {stats['entity_types']}")

    # Find entities related to "Qian Chen"
    print("\nFinding entities related to 'Qian Chen':")
    related = kg.find_related_entities("Qian Chen", max_depth=2)

    for entity_id, distance, entity_data in related[:10]:
        print(f"  - {entity_data.get('name', entity_id)} (type: {entity_data.get('type')}, distance: {distance})")

    # Find path between two entities
    print("\nFinding path between 'Qian Chen' and 'Meta Platforms':")
    paths = kg.find_path("Qian Chen", "Meta Platforms", max_length=3)

    if paths:
        for i, path in enumerate(paths[:3]):
            print(f"  Path {i+1}: {' -> '.join(path)}")
    else:
        print("  No paths found")


def example_5_ablation_comparison():
    """Example 5: Ablation testing"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Ablation Testing")
    print("="*80)

    agent = RetrievalAgent()
    agent.load_data()

    query = "Did Qian Chen at Meta sign a non-compete agreement?"

    # Test with full system
    print("\n[Test 1] Full System:")
    update_config(USE_DSPY_SIGNATURES=True, USE_NOVELTY_CIRCUIT=True, USE_NEGATIVE_MEMORY=True)
    results_full = agent.search(query)
    print(f"  Results: {len(results_full)}")
    print(f"  Precision: {agent._calculate_precision(results_full):.1%}")

    # Test without DSPy
    print("\n[Test 2] Without DSPy Signatures:")
    update_config(USE_DSPY_SIGNATURES=False)
    results_no_dspy = agent.search(query)
    print(f"  Results: {len(results_no_dspy)}")
    print(f"  Precision: {agent._calculate_precision(results_no_dspy):.1%}")

    # Reset
    update_config(USE_DSPY_SIGNATURES=True)

    print("\nAblation testing demonstrates the value of each component!")


def main():
    """Run all examples"""

    # Check API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: Set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='sk-...'")
        return

    print("\n" + "="*80)
    print("HIGH-SNR AGENTIC RAG - EXAMPLES")
    print("="*80)

    # Run examples
    example_1_basic_search()

    # Uncomment to run other examples:
    # example_2_custom_configuration()
    # example_3_custom_data()
    # example_4_knowledge_graph_exploration()
    # example_5_ablation_comparison()

    print("\n" + "="*80)
    print("Examples complete! Check the code in example.py for more details.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
