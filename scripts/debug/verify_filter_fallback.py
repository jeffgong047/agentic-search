
import sys
import os
from unittest.mock import MagicMock, patch

# Patch sys.path to ensure we can import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    print(f"Adding {project_root} to sys.path")
    sys.path.append(project_root)


# Mock dependencies BEFORE importing backend
from unittest.mock import MagicMock
sys.modules["elasticsearch"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

from core.elasticsearch_hybrid_backend import ElasticsearchHybridBackend
from core.production_interfaces import SearchQuery

def test_fallback_logic():
    print("\n=== Testing Filter Fallback Logic ===")
    
    # Mock Elasticsearch client
    mock_es = MagicMock()
    
    # Mock search responses
    def search_side_effect(index=None, body=None):
        # Check if query has a filter
        bool_query = body.get("query", {}).get("bool", {})
        has_filter = "filter" in bool_query and len(bool_query["filter"]) > 0
        
        if has_filter:
            print(f"[MockES] Received query WITH filters ({bool_query['filter']}). Returning 0 hits.")
            return {"hits": {"hits": []}}
        else:
            print("[MockES] Received query WITHOUT filters (Fallback). Returning 1 hit.")
            return {
                "hits": {
                    "hits": [{
                        "_id": "doc1",
                        "_score": 0.9,
                        "_source": {
                            "content": "Fallback content",
                            "metadata": {"doc_type": "decision"}
                        }
                    }]
                }
            }
            
    mock_es.search.side_effect = search_side_effect
    
    # Mock Embedder to avoid downloading models
    mock_embedder = MagicMock()
    # Mock return value to have tolist() method
    mock_vector = MagicMock()
    mock_vector.tolist.return_value = [0.1] * 384
    mock_embedder.encode.return_value = [mock_vector]

    # Initialize backend with mocks
    with patch('core.elasticsearch_hybrid_backend.Elasticsearch', return_value=mock_es), \
         patch('core.elasticsearch_hybrid_backend.SentenceTransformer', return_value=mock_embedder):
        
        backend = ElasticsearchHybridBackend()
        # Manually attach mocks if init doesn't use the return value directly (it does)
        backend.es = mock_es 
        backend.embedder = mock_embedder
        
        # Test Case: Hallucinated Filter
        print("\n--- Sending Query with Hallucinated Filter 'statutes' ---")
        query = SearchQuery(
            text="test",
            vector_query="test",
            keyword_query="test",
            filters={"doc_type": "statutes"}, # Assume this returns 0 results
            negative_constraints=[],
            top_k=1
        )
        
        # Inject known schema so validation doesn't kill it immediately (we want to test the FALLBACK, not just validation)
        # Or we can just let validation happen. 
        # But to test FALLBACK specifically, we need the query to PASS validation but FAIL search.
        # So let's pretend "statutes" IS in the schema, but just happens to have 0 results for this specific query.
        backend.valid_doc_types = {"statutes", "decision"} 
        
        results = backend.hybrid_search(query)
        
        print(f"\n[Result] Retrieved {len(results)} docs.")
        if len(results) == 1 and results[0].content == "Fallback content":
            print("✅ SUGGESTION: Fallback logic worked! Retrieved document from fallback query.")
        else:
            print("❌ FAILURE: Fallback did not return expected document.")

def test_validation_logic():
    print("\n=== Testing Filter Validation Logic ===")
    
    mock_es = MagicMock()
    mock_embedder = MagicMock()
    
    with patch('core.elasticsearch_hybrid_backend.Elasticsearch', return_value=mock_es), \
         patch('core.elasticsearch_hybrid_backend.SentenceTransformer', return_value=mock_embedder):
        
        backend = ElasticsearchHybridBackend()
        backend.es = mock_es
        
        # Pre-load schema
        backend.valid_doc_types = {"decision", "contract"}
        
        # Query with INVALID type
        filters = {"doc_type": "hallucinated_type", "org": "Meta"}
        print(f"Input filters: {filters}")
        
        safe_filters = backend.validate_filters(filters)
        print(f"Validated filters: {safe_filters}")
        
        if "doc_type" not in safe_filters and safe_filters["org"] == "Meta":
            print("✅ SUGGESTION: Validation correctly removed invalid doc_type.")
        else:
            print("❌ FAILURE: Validation failed.")

if __name__ == "__main__":
    test_validation_logic()
    test_fallback_logic()
