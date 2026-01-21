"""
DEMO: Retrieval Agent with Real Infrastructure

This demonstrates how to:
1. Connect to Elasticsearch and PostgreSQL
2. Load real data (or use mock data)
3. Run the agent
4. Return results in any format your mentor needs
"""

import os
import sys
import dspy
from core.config import get_config
from main import RetrievalAgent

def verify_mickey_mouse_disambiguation():
    """
    Test the agent's ability to disambiguate 'Mickey Mouse'.
    Checks if it correctly identifies the Meta Research Scientist and filters out distributors.
    """
    print("="*80)
    print("VERIFYING MICKEY MOUSE DISAMBIGUATION (Agent Freedom Test)")
    print("="*80)

    # 1. Initialize Agent
    # Note: RetrievalAgent uses LLM_MODEL from config
    agent = RetrievalAgent()
    
    # 2. Load Mock Data (contains 3 Mickey Mouses)
    agent.load_data()
    
    # 3. Test Queries
    queries = [
        "What is the non-compete status for Mickey Mouse at Meta?",
        "Tell me about Mickey Mouse's work at Anytime AI.",
        "How is Richard Wang at Anytime AI connected to Mickey Mouse?",
        "Who is Mickey Mouse at Berkeley?"
    ]
    
    for query in queries:
        print(f"\n>>> QUERY: {query}")
        results = agent.search(query)
        
        print(f"Retrieved {len(results)} docs.")
        for i, doc in enumerate(results[:3]):
            org = doc.metadata.get('org', 'Unknown')
            type = doc.metadata.get('type', 'Unknown')
            print(f" [{i+1}] Org: {org} | Type: {type} | Score: {doc.score:.3f}")

        # Validation Logic
        top_org = results[0].metadata.get('org', '').lower() if results else ''
        if "meta" in query.lower() and "meta" in top_org:
            print(" ✅ Correct: Disambiguated to Meta.")
        elif "anytime" in query.lower() and "anytime" in top_org:
            print(" ✅ Correct: Disambiguated to Anytime AI.")
        elif "richard wang" in query.lower() and ("anytime" in top_org or "bio" in top_org):
            print(" ✅ Correct: Found relational connection.")
        elif "berkeley" in query.lower() and "berkeley" in top_org:
            print(" ✅ Correct: Disambiguated to Berkeley.")
        else:
            print(" ❌ Error: Failed to disambiguate correctly.")

if __name__ == "__main__":
    # Ensure API key is set
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)
        
    verify_mickey_mouse_disambiguation()

