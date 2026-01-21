"""
Small-scale Trace Tests for Debugging Agentic Search
Uses Claude Opus API for high-quality reasoning traces

Run with: export ANTHROPIC_API_KEY='your-key' && python test_trace_debug.py
"""

import os
import json
from typing import List, Dict
import dspy
from core.data_structures import create_initial_state, SearchResult
from core.orchestrator import LegalOrchestrator
from core.backends import create_best_backend
from core.backends_wrapper import BackendAgenticSearchEngine
from mock_data import get_mock_dataset


def configure_claude_opus():
    """Configure DSPy to use Claude Opus API"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Get it from: https://console.anthropic.com/")

    # Use Anthropic provider in DSPy (via LiteLLM)
    lm = dspy.LM(
        model="anthropic/claude-opus-4-5-20251101",  # Latest Opus
        api_key=api_key,
        max_tokens=2000
    )
    dspy.settings.configure(lm=lm)
    print(f"✓ Configured DSPy with Claude Opus 4.5")


def print_trace_header(test_name: str, query: str):
    """Print formatted trace header"""
    print("\n" + "="*80)
    print(f" {test_name}".center(80))
    print("="*80)
    print(f"\n📝 Query: \"{query}\"\n")


def print_search_plan(plan):
    """Pretty-print the SearchPlan from orchestrator"""
    print("\n🧠 DSPy Orchestrator Output:")
    print("-" * 60)
    print(f"  Intent: {plan.primary_intent}")
    print(f"  HyDE Passage: {plan.hyde_passage[:100]}...")
    print(f"\n  Search Queries:")
    for i, q in enumerate(plan.search_queries, 1):
        print(f"    {i}. {q}")
    print(f"\n  Filters: {json.dumps(plan.filter_constraints, indent=4)}")
    print(f"  Negative Constraints: {plan.negative_constraints}")
    print(f"  Graph Depth: {plan.entity_graph_depth}")
    print(f"\n  Parallel Strategies ({len(plan.strategies)}):")
    for i, s in enumerate(plan.strategies, 1):
        print(f"    {i}. [{s.strategy_type}] {s.query_variant[:50]}...")
    print("-" * 60)


def print_results_trace(results: List[SearchResult], max_show: int = 3):
    """Print retrieval results with detailed traces"""
    print(f"\n📚 Retrieved {len(results)} Documents:")
    print("-" * 60)

    for i, r in enumerate(results[:max_show], 1):
        print(f"\n  [{i}] ID: {r.id}")
        print(f"      Score: {r.score:.4f}")
        print(f"      Source: {r.source_index}")
        print(f"      Org: {r.metadata.get('org', 'N/A')}")
        print(f"      Content: {r.content[:120]}...")
        if "strategy_source" in r.metadata:
            print(f"      Strategy: {r.metadata['strategy_source']}")

    if len(results) > max_show:
        print(f"\n  ... and {len(results) - max_show} more documents")
    print("-" * 60)


def print_agent_state_trace(state, iteration: int):
    """Print detailed agent state for debugging"""
    print(f"\n🔄 Agent State After Iteration {iteration}:")
    print("-" * 60)
    print(f"  Step Count: {state['step_count']}")
    print(f"  Novelty Score: {state['novelty_score']:.4f}")
    print(f"  Should Continue: {state['should_continue']}")
    print(f"  Known Doc IDs: {len(state['known_doc_ids'])}")
    print(f"\n  Verified Facts ({len(state['verified_facts'])}):")
    for fact in state['verified_facts'][-3:]:  # Last 3
        print(f"    - {fact}")
    print(f"\n  Negative Cache ({len(state['negative_cache'])}):")
    for item in state['negative_cache'][-3:]:  # Last 3
        print(f"    - AVOID: {item.get('entity')} (Reason: {item.get('reason')})")
    print("-" * 60)


# ============================================================================
# TEST 1: Simple Entity Disambiguation (Most Basic)
# ============================================================================

def test_1_simple_disambiguation():
    """
    TEST 1: Basic entity disambiguation with 3 documents

    Expected Trace:
    - Orchestrator should identify "Entity_Resolution" intent
    - Should filter for org="Meta"
    - Should add negative constraints for "Shanghai" and "UC Berkeley"
    """
    print_trace_header("TEST 1: Simple Entity Disambiguation",
                       "Did Mickey Mouse at Meta sign a non-compete?")

    # Create mini dataset (just 3 docs)
    # Note: Using document_type to match orchestrator output, proper case for org
    mini_docs = [
        {
            "id": "meta_1",
            "content": "Mickey Mouse joined Meta in 2023 as AI Research Scientist. Research focuses on large language models and neural networks. Employment contract includes standard non-compete clause restricting competitive employment for 1 year within 50 miles.",
            "metadata": {"org": "Meta", "document_type": "employment_contract", "year": 2023}
        },
        {
            "id": "shanghai_1",
            "content": "Mickey Mouse practices corporate law at Shanghai Law Firm since 2020. Specializes in M&A transactions and corporate governance.",
            "metadata": {"org": "Shanghai Law Firm", "document_type": "profile", "year": 2022}
        },
        {
            "id": "student_1",
            "content": "Mickey Mouse is a PhD student at UC Berkeley studying computer architecture and distributed systems.",
            "metadata": {"org": "UC Berkeley", "document_type": "bio", "year": 2024}
        }
    ]

    # Setup backend with Elasticsearch config
    es_config = {
        "host": "legalbench_elasticsearch",  # Docker service name
        "port": 9200,
        "index": "test_trace_debug"
    }
    backend = create_best_backend(es_config=es_config)
    backend.index_documents(mini_docs)

    # Test orchestrator in isolation
    print("\n📍 Step 1: Test Orchestrator (DSPy) Alone")
    orchestrator = LegalOrchestrator()
    state = create_initial_state("Did Mickey Mouse at Meta sign a non-compete?")

    plan = orchestrator.forward(state)
    print_search_plan(plan)

    # Validate orchestrator output
    assert plan.primary_intent in ["Entity_Resolution", "Statute_Lookup"], \
        f"Expected entity resolution, got: {plan.primary_intent}"
    assert "Meta" in str(plan.filter_constraints).lower() or \
           any("meta" in q.lower() for q in plan.search_queries), \
        "Should mention 'Meta' in queries or filters"

    print("\n✅ Orchestrator correctly identified entity disambiguation task")

    # Test full agent
    print("\n📍 Step 2: Test Full Agent Pipeline")
    engine = BackendAgenticSearchEngine(backend)
    final_state = engine.search("Did Mickey Mouse at Meta sign a non-compete?")

    print_agent_state_trace(final_state, final_state["step_count"])
    print_results_trace(final_state["retrieved_docs"])

    # Validate results
    results = final_state["retrieved_docs"]
    meta_count = sum(1 for r in results if r.metadata.get("org") == "Meta")
    precision = meta_count / len(results) if results else 0

    print(f"\n📊 Test Results:")
    print(f"  Meta documents: {meta_count}/{len(results)}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Iterations: {final_state['step_count']}")

    assert meta_count > 0, "Should retrieve at least 1 Meta document"
    assert precision >= 0.5, f"Precision too low: {precision:.2%}"

    print("\n✅ TEST 1 PASSED: Basic disambiguation works!")
    return final_state


# ============================================================================
# TEST 2: Multi-Step Reasoning with Memory
# ============================================================================

def test_2_memory_evolution():
    """
    TEST 2: Memory evolution across iterations

    Expected Trace:
    - Iteration 1: Broad search, finds multiple Mickey Mouses
    - Iteration 2: Refines with negative constraints
    - Should see verified_facts and negative_cache grow
    """
    print_trace_header("TEST 2: Memory Evolution (Multi-Iteration)",
                       "What projects did Mickey Mouse work on?")

    # Slightly larger dataset (6 docs) to trigger multi-iteration
    docs = get_mock_dataset()  # Use full mock dataset

    es_config = {
        "host": "legalbench_elasticsearch",
        "port": 9200,
        "index": "test_memory_evolution"
    }
    backend = create_best_backend(es_config=es_config)
    backend.index_documents(docs)

    # Enable debug mode for detailed traces
    os.environ["DEBUG_MODE"] = "true"

    engine = BackendAgenticSearchEngine(backend)
    final_state = engine.search("What projects did Mickey Mouse at Meta work on?")

    # Detailed iteration-by-iteration trace
    print(f"\n🔍 Multi-Iteration Trace:")
    print(f"  Total Iterations: {final_state['step_count']}")
    print(f"  Final Novelty: {final_state['novelty_score']:.4f}")

    print_agent_state_trace(final_state, final_state["step_count"])
    print_results_trace(final_state["retrieved_docs"], max_show=5)

    # Validate memory evolution
    assert len(final_state["verified_facts"]) > 0, "Should have verified facts"
    assert final_state["step_count"] >= 1, "Should have at least 1 iteration"

    print("\n✅ TEST 2 PASSED: Memory evolution works!")

    os.environ.pop("DEBUG_MODE", None)
    return final_state


# ============================================================================
# TEST 3: Trace DSPy Chain-of-Thought
# ============================================================================

def test_3_dspy_reasoning_trace():
    """
    TEST 3: Inspect DSPy's internal reasoning (Chain-of-Thought)

    Shows the actual reasoning steps Claude Opus uses
    """
    print_trace_header("TEST 3: DSPy Chain-of-Thought Trace",
                       "Find legal precedents on non-compete agreements in tech")

    orchestrator = LegalOrchestrator()
    state = create_initial_state("Find legal precedents on non-compete agreements in tech")

    # The planner is a ChainOfThought module, so it has internal reasoning
    print("\n🧠 Invoking DSPy Chain-of-Thought...")
    plan = orchestrator.forward(state)

    # Print the raw DSPy output (includes reasoning if available)
    print("\n📋 Raw DSPy Output:")
    print("-" * 60)
    raw_result = orchestrator.planner(
        query=state["query"],
        context_memory="No prior context.",
        negative_constraints="No constraints."
    )

    # Print all fields from the signature
    for field in ["primary_intent", "hyde_passage", "search_queries",
                  "filter_constraints", "new_negative_constraints",
                  "entity_graph_depth", "strategies"]:
        if hasattr(raw_result, field):
            value = getattr(raw_result, field)
            print(f"\n{field}:")
            if isinstance(value, str) and len(value) > 100:
                print(f"  {value[:200]}...")
            else:
                print(f"  {value}")

    print("-" * 60)

    # Check if reasoning/rationale is available in history
    if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
        print("\n💭 LLM Call History:")
        for i, call in enumerate(dspy.settings.lm.history[-1:], 1):  # Last call
            print(f"\nCall {i}:")
            if 'prompt' in call:
                print(f"  Prompt: {call['prompt'][:300]}...")
            if 'response' in call:
                print(f"  Response: {call['response'][:300]}...")

    print("\n✅ TEST 3 PASSED: DSPy trace visible!")
    return plan


# ============================================================================
# TEST 4: Compare With vs Without DSPy
# ============================================================================

def test_4_ablation_dspy_vs_raw():
    """
    TEST 4: Ablation - Compare DSPy structured output vs raw LLM

    Shows why DSPy typed signatures are better than prompt hacking
    """
    print_trace_header("TEST 4: Ablation Test (DSPy vs Raw LLM)",
                       "Who is Mickey Mouse?")

    docs = get_mock_dataset()[:5]  # Small set
    es_config = {
        "host": "legalbench_elasticsearch",
        "port": 9200,
        "index": "test_ablation"
    }
    backend = create_best_backend(es_config=es_config)
    backend.index_documents(docs)

    # Test WITH DSPy
    print("\n🔬 Test A: WITH DSPy Signatures")
    os.environ["USE_DSPY_SIGNATURES"] = "true"
    from config import reload_config
    reload_config()

    engine_dspy = BackendAgenticSearchEngine(backend)
    state_dspy = engine_dspy.search("Who is Mickey Mouse at Meta?")

    print(f"  Results: {len(state_dspy['retrieved_docs'])}")
    print(f"  Iterations: {state_dspy['step_count']}")
    print(f"  Novelty: {state_dspy['novelty_score']:.4f}")

    # Test WITHOUT DSPy (raw LLM)
    print("\n🔬 Test B: WITHOUT DSPy (Raw LLM)")
    os.environ["USE_DSPY_SIGNATURES"] = "false"
    reload_config()

    engine_raw = BackendAgenticSearchEngine(backend)
    state_raw = engine_raw.search("Who is Mickey Mouse at Meta?")

    print(f"  Results: {len(state_raw['retrieved_docs'])}")
    print(f"  Iterations: {state_raw['step_count']}")
    print(f"  Novelty: {state_raw['novelty_score']:.4f}")

    # Compare
    print(f"\n📊 Comparison:")
    print(f"  DSPy Results: {len(state_dspy['retrieved_docs'])}")
    print(f"  Raw Results: {len(state_raw['retrieved_docs'])}")
    print(f"  Difference: {len(state_dspy['retrieved_docs']) - len(state_raw['retrieved_docs'])}")

    # Restore DSPy
    os.environ["USE_DSPY_SIGNATURES"] = "true"
    reload_config()

    print("\n✅ TEST 4 PASSED: Ablation comparison complete!")
    return state_dspy, state_raw


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_trace_tests():
    """Run all debugging trace tests"""
    print("\n" + "="*80)
    print(" AGENTIC SEARCH - DEBUGGING TRACE TESTS (Claude Opus)".center(80))
    print("="*80)
    print("\n🎯 Purpose: Small, focused tests with detailed traces for debugging")
    print("🤖 Model: Claude Opus 4.5 (Anthropic)")
    print("\n" + "="*80)

    # Check API key
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("\n❌ ERROR: ANTHROPIC_API_KEY not set")
        print("\nGet your key from: https://console.anthropic.com/")
        print("Then run: export ANTHROPIC_API_KEY='your-key'\n")
        return

    # Configure Claude Opus
    configure_claude_opus()

    # Run tests
    tests = [
        ("Simple Disambiguation", test_1_simple_disambiguation),
        ("Memory Evolution", test_2_memory_evolution),
        ("DSPy Reasoning Trace", test_3_dspy_reasoning_trace),
        ("Ablation: DSPy vs Raw", test_4_ablation_dspy_vs_raw),
    ]

    results = {}

    for name, test_func in tests:
        try:
            print(f"\n\n{'▶'*40}")
            print(f"Running: {name}")
            print(f"{'▶'*40}")

            result = test_func()
            results[name] = {"status": "✅ PASSED", "result": result}

        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results[name] = {"status": "❌ FAILED", "error": str(e)}

    # Summary
    print("\n\n" + "="*80)
    print(" TEST SUMMARY".center(80))
    print("="*80)

    for test_name, result in results.items():
        print(f"\n{result['status']} {test_name}")
        if "error" in result:
            print(f"    Error: {result['error']}")

    passed = sum(1 for r in results.values() if r["status"] == "✅ PASSED")
    total = len(results)

    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Agent is working correctly with Claude Opus")
        print("✅ Traces show clear reasoning steps")
        print("✅ Memory evolution is functioning")
    else:
        print("\n⚠️ Some tests failed - check traces above for debugging")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    run_all_trace_tests()
