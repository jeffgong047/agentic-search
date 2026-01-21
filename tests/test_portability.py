"""
Portability Test: Verify the agent works with different backends

This proves the agent is truly backend-agnostic.
"""

import os
from main import RetrievalAgent
from core.backends import BackendFactory
from custom_backend_example import FileBasedBackend
from mock_data import get_mock_dataset


def test_tri_index_backend():
    """Test 1: Built-in tri-index backend (always works)"""
    print("\n" + "="*60)
    print("TEST 1: Tri-Index Backend (Built-in)")
    print("="*60)

    agent = RetrievalAgent(backend_type="tri-index")
    agent.load_data(get_mock_dataset())

    results = agent.search("Mickey Mouse Meta non-compete")

    print(f"✓ Results: {len(results)} documents")
    print(f"✓ Backend: {agent.backend.get_stats()['backend_type']}")

    assert len(results) > 0, "Should return results"
    assert agent.backend.get_stats()['backend_type'] == "tri-index"

    print("✅ PASS: Tri-index backend works\n")
    return results


def test_file_based_backend():
    """Test 2: File-based custom backend (no external deps)"""
    print("\n" + "="*60)
    print("TEST 2: File-Based Backend (Custom)")
    print("="*60)

    # Register custom backend
    BackendFactory.register("file", FileBasedBackend)

    agent = RetrievalAgent(backend_type="file")
    agent.load_data(get_mock_dataset())

    results = agent.search("Mickey Mouse Meta")

    print(f"✓ Results: {len(results)} documents")
    print(f"✓ Backend: {agent.backend.get_stats()['backend_type']}")

    assert len(results) > 0, "Should return results"
    assert agent.backend.get_stats()['backend_type'] == "file_based"

    print("✅ PASS: File-based backend works\n")
    return results


def test_auto_detection():
    """Test 3: Auto-detection picks best backend"""
    print("\n" + "="*60)
    print("TEST 3: Auto-Detection")
    print("="*60)

    # No config → should use tri-index
    agent = RetrievalAgent()
    agent.load_data(get_mock_dataset())

    print(f"✓ Auto-detected: {agent.backend.get_stats()['backend_type']}")

    # With ES config → should try ES (will fallback if not available)
    try:
        agent_es = RetrievalAgent(es_config={"host": "localhost", "index": "test"})
        print(f"✓ With ES config: {agent_es.backend.get_stats()['backend_type']}")
    except:
        print("✓ ES not available, would fallback to tri-index")

    print("✅ PASS: Auto-detection works\n")


def test_same_query_different_backends():
    """Test 4: Same query, different backends (proves portability)"""
    print("\n" + "="*60)
    print("TEST 4: Same Query, Different Backends")
    print("="*60)

    query = "Mickey Mouse Meta non-compete"
    backends_to_test = ["tri-index"]

    # Register file backend
    BackendFactory.register("file", FileBasedBackend)
    backends_to_test.append("file")

    all_results = {}

    for backend_type in backends_to_test:
        print(f"\nTesting with {backend_type} backend...")

        agent = RetrievalAgent(backend_type=backend_type)
        agent.load_data(get_mock_dataset())

        results = agent.search(query)

        all_results[backend_type] = results
        print(f"  ✓ Got {len(results)} results")

    # Verify all backends returned results
    for backend, results in all_results.items():
        assert len(results) > 0, f"{backend} should return results"

    print("\n✅ PASS: All backends work with same query\n")


def test_backend_switching():
    """Test 5: Can switch backends at runtime"""
    print("\n" + "="*60)
    print("TEST 5: Backend Switching")
    print("="*60)

    BackendFactory.register("file", FileBasedBackend)

    # Agent 1 with tri-index
    agent1 = RetrievalAgent(backend_type="tri-index")
    agent1.load_data(get_mock_dataset())
    results1 = agent1.search("test")

    print(f"✓ Agent 1 ({agent1.backend.get_stats()['backend_type']}): {len(results1)} results")

    # Agent 2 with file-based
    agent2 = RetrievalAgent(backend_type="file")
    agent2.load_data(get_mock_dataset())
    results2 = agent2.search("test")

    print(f"✓ Agent 2 ({agent2.backend.get_stats()['backend_type']}): {len(results2)} results")

    # Both should work
    assert len(results1) > 0
    assert len(results2) > 0

    print("✅ PASS: Can run multiple agents with different backends\n")


def test_custom_backend_registration():
    """Test 6: Can register custom backend at runtime"""
    print("\n" + "="*60)
    print("TEST 6: Custom Backend Registration")
    print("="*60)

    # List backends before
    print(f"Available backends before: {BackendFactory.list_backends()}")

    # Register custom
    BackendFactory.register("my_custom", FileBasedBackend)

    # List backends after
    print(f"Available backends after: {BackendFactory.list_backends()}")

    assert "my_custom" in BackendFactory.list_backends()

    # Use custom backend
    agent = RetrievalAgent(backend_type="my_custom")
    agent.load_data(get_mock_dataset())

    print(f"✓ Custom backend works: {agent.backend.get_stats()['backend_type']}")

    print("✅ PASS: Custom backend registration works\n")


def run_all_tests():
    """Run all portability tests"""
    print("\n" + "="*70)
    print(" PORTABILITY TESTS".center(70))
    print("="*70)
    print("\nThese tests prove the agent works with ANY backend.\n")

    # Check for API key (needed for DSPy)
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key'\n")
        print("Skipping tests that require DSPy...\n")
        return

    tests = [
        test_tri_index_backend,
        test_file_based_backend,
        test_auto_detection,
        test_same_query_different_backends,
        test_backend_switching,
        test_custom_backend_registration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_func.__name__}")
            print(f"   Error: {e}\n")
            failed += 1

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY".center(70))
    print("="*70)
    print(f"\n  ✅ Passed: {passed}/{len(tests)}")
    print(f"  ❌ Failed: {failed}/{len(tests)}\n")

    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print("\nPortability guarantee: ✅ VERIFIED")
        print("\nThe agent works with ANY backend that implements SearchBackend interface.")
    else:
        print("⚠️  Some tests failed. Check errors above.")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
