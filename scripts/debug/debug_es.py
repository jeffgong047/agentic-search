
import sys
from elasticsearch_hybrid_backend import ElasticsearchHybridBackend
from core.production_interfaces import SearchQuery

def main():
    print("Initializing Backend...")
    backend = ElasticsearchHybridBackend(
        es_url="http://legalbench_elasticsearch:9200",
        index_name="legalbench_documents",
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    
    query_text = "employment contract non-compete clause"
    # Create valid search query object
    query = SearchQuery(
        text=query_text,
        vector_query=query_text,  # Use same text for vector
        keyword_query=query_text, # Use same text for keyword
        filters={},
        negative_constraints=[],
        top_k=20
    )
    
    print(f"Running hybrid search for: '{query.text}'")
    
    try:
        results = backend.hybrid_search(query)
        print("Success!")
        print(results)
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()
        # Debug info is already printed by the backend instrumentation

if __name__ == "__main__":
    main()
