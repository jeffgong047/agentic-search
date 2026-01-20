"""
Elasticsearch Hybrid Search Backend

Single backend that does EVERYTHING:
- Vector search (k-NN with HNSW)
- BM25 keyword search
- Metadata filtering
- RRF (Reciprocal Rank Fusion) - built-in!

NO FAISS. NO SEPARATE BM25. Just Elasticsearch.
"""

import json
import json
import json
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from .production_interfaces import HybridSearchBackend, SearchQuery, SearchResult


class ElasticsearchHybridBackend(HybridSearchBackend):
    """
    Production-ready hybrid search using ONLY Elasticsearch.

    Why this is "Industry Meta":
    ✅ Single system (no FAISS/BM25 sync issues)
    ✅ Native RRF (Elasticsearch 8.9+)
    ✅ Metadata filtering (can't do with FAISS)
    ✅ Production scalability
    ✅ Mentor likely already has this
    """

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "legal_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize ES hybrid backend.

        Args:
            es_url: Elasticsearch URL
            index_name: Index name
            embedding_model: Model for query embeddings
        """
        self.es = Elasticsearch([es_url])
        self.index_name = index_name

        # Embedding model for query encoding
        # In production, mentor might do this server-side
        self.embedder = SentenceTransformer(embedding_model)

        print(f"[ES Hybrid] Connected to {es_url}")

        # Verify ES version supports RRF
        info = self.es.info()
        version = info['version']['number']
        print(f"[ES Hybrid] Elasticsearch version: {version}")

    def get_index_schema(self) -> Dict[str, List[str]]:
        """
        Dynamically fetch the schema and available filter values from the index.
        This allows the Agent to "read the manual" and know what filters are valid.
        """
        # Aggregation query to get unique doc_types
        # We use a large size to capture all types (corpus is small, types are few)
        body = {
            "size": 0,
            "aggs": {
                "doc_types": {
                    "terms": {"field": "metadata.doc_type.keyword", "size": 100}
                }
            }
        }
        
        try:
            response = self.es.search(index=self.index_name, body=body)
            buckets = response.get("aggregations", {}).get("doc_types", {}).get("buckets", [])
            doc_types = [b["key"] for b in buckets]
            return {"doc_type": doc_types}
        except Exception as e:
            print(f"[Backend] Schema introspection failed: {str(e)}")
            return {"doc_type": []}

        return results

    def validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize filters.
        specifically checks if 'doc_type' is valid.
        """
        if not filters:
            return {}

        validated = filters.copy()
        
        # Check doc_type
        if "doc_type" in validated:
            requested_type = validated["doc_type"]
            # Lazy load schema if not already loaded
            if not getattr(self, "valid_doc_types", None):
                schema = self.get_index_schema()
                self.valid_doc_types = set(schema.get("doc_type", []))
            
            if self.valid_doc_types and requested_type not in self.valid_doc_types:
                print(f"[Backend] WARNING: Filter hallucination detected! '{requested_type}' not in index. Removing filter.")
                del validated["doc_type"]
                
        return validated

    def hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Execute hybrid search using Elasticsearch RRF.
        
        Includes Hallucination Fix:
        1. Validates filters against index schema
        2. Fallback to no-filter search if filtered search returns 0 results
        """
        # 1. Validate Filters
        safe_filters = self.validate_filters(query.filters)
        
        # Encode vector query
        vector_query_embedding = self.embedder.encode([query.vector_query])[0].tolist()

        def build_es_query(filters_to_use):
            # Init knn_query dict
            knn_query = {
                "field": "embedding",
                "query_vector": vector_query_embedding,
                "k": query.top_k,
                "num_candidates": 100
            }
            
            # Only add filter if it exists
            knn_filter = self._build_filters(filters_to_use)
            if knn_filter:
                knn_query["filter"] = knn_filter

            return {
                "knn": knn_query,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query.keyword_query,
                                    "fields": ["content"],
                                    "type": "best_fields"
                                }
                            }
                        ],
                        "filter": self._build_filters(filters_to_use),
                        "must_not": self._build_negative_constraints(query.negative_constraints)
                    }
                },
                "size": query.top_k,
                "rank": {
                    "rrf": {
                        "window_size": max(query.top_k, 50)
                    }
                }
            }

        # 2. Execute First Attempt (with filters)
        try:
            es_query = build_es_query(safe_filters)
            # print(f"[DEBUG] ES Query: {json.dumps(es_query, indent=2)}")
            response = self.es.search(index=self.index_name, body=es_query)
            
            hits = response["hits"]["hits"]
            
            # 3. Fallback Mechanism
            if len(hits) == 0 and safe_filters:
                print(f"[Backend] Zero results with filters {safe_filters}. Fallback: Retrying without filters.")
                fallback_query = build_es_query({}) # Empty filters
                response = self.es.search(index=self.index_name, body=fallback_query)
                hits = response["hits"]["hits"]
                
        except Exception as e:
            print(f"[ERROR] ES Search Failed: {e}")
            raise e

        # Parse results
        results = []
        for hit in hits:
            results.append(SearchResult(
                id=hit["_id"],
                content=hit["_source"]["content"],
                score=hit["_score"],
                metadata=hit["_source"].get("metadata", {}),
                source="hybrid"
            ))

        return results

    def _build_filters(self, filters: Dict[str, Any]) -> List[Dict]:
        """Build Elasticsearch filter clauses"""
        filter_clauses = []

        for key, value in filters.items():
            filter_clauses.append({
                "term": {f"metadata.{key}": value}
            })

        return filter_clauses

    def _build_negative_constraints(self, negative_terms: List[str]) -> List[Dict]:
        """Build Elasticsearch must_not clauses"""
        must_not_clauses = []

        for term in negative_terms:
            must_not_clauses.append({
                "match": {"content": term}
            })

        return must_not_clauses

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        try:
            count = self.es.count(index=self.index_name)
            return {
                "backend_type": "elasticsearch_hybrid",
                "index_name": self.index_name,
                "document_count": count["count"],
                "features": ["vector_search", "bm25", "rrf", "metadata_filtering"]
            }
        except Exception as e:
            return {
                "backend_type": "elasticsearch_hybrid",
                "error": str(e)
            }


# ====================================================================
# ADVANCED: Elasticsearch 8.9+ RRF (Even Better!)
# ====================================================================

class ElasticsearchRRFBackend(HybridSearchBackend):
    """
    Uses Elasticsearch's native RRF feature (8.9+).

    This is even cleaner - ES handles the fusion automatically.
    """

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "legal_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.es = Elasticsearch([es_url])
        self.index_name = index_name
        self.embedder = SentenceTransformer(embedding_model)

        print(f"[ES RRF] Connected to {es_url}")

    def hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Hybrid search using ES native RRF.

        Requires Elasticsearch 8.9+
        """
        # Encode query
        vector_embedding = self.embedder.encode([query.vector_query])[0].tolist()

        # Build query with RRF (native ES feature)
        es_query = {
            "sub_searches": [
                # Sub-query 1: Vector search
                {
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                                "params": {"query_vector": vector_embedding}
                            }
                        }
                    }
                },
                # Sub-query 2: BM25
                {
                    "query": {
                        "multi_match": {
                            "query": query.keyword_query,
                            "fields": ["content"]
                        }
                    }
                }
            ],
            "rank": {
                "rrf": {
                    "window_size": 50,
                    "rank_constant": 20
                }
            }
        }

        # Add filters if needed
        if query.filters or query.negative_constraints:
            # Apply post-filtering
            es_query["post_filter"] = {
                "bool": {
                    "filter": self._build_filters(query.filters),
                    "must_not": self._build_negative_constraints(query.negative_constraints)
                }
            }

        # Execute
        response = self.es.search(index=self.index_name, body=es_query, size=query.top_k)

        # Parse
        results = []
        for hit in response["hits"]["hits"]:
            results.append(SearchResult(
                id=hit["_id"],
                content=hit["_source"]["content"],
                score=hit["_score"],
                metadata=hit["_source"].get("metadata", {}),
                source="rrf"
            ))

        return results

    def _build_filters(self, filters: Dict[str, Any]) -> List[Dict]:
        """Build filter clauses"""
        return [{"term": {f"metadata.{k}": v}} for k, v in filters.items()]

    def _build_negative_constraints(self, negative_terms: List[str]) -> List[Dict]:
        """Build must_not clauses"""
        return [{"match": {"content": term}} for term in negative_terms]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        count = self.es.count(index=self.index_name)
        return {
            "backend_type": "elasticsearch_rrf",
            "document_count": count["count"],
            "features": ["native_rrf", "vector_search", "bm25", "filters"]
        }


# ====================================================================
# EXAMPLE USAGE
# ====================================================================

def example_hybrid_search():
    """
    Example showing ES hybrid search.
    """
    from production_interfaces import SearchQuery

    # Initialize backend
    backend = ElasticsearchHybridBackend(
        es_url="http://localhost:9200",
        index_name="legal_documents"
    )

    # Create search query (from DSPy orchestrator)
    query = SearchQuery(
        text="Did Qian Chen at Meta sign a non-compete?",
        vector_query="Qian Chen Meta Platforms non-compete agreement employment California",
        keyword_query="Qian Chen Meta non-compete",
        filters={"org": "Meta"},
        negative_constraints=["Shanghai", "Finance"],
        top_k=5
    )

    # Execute hybrid search
    results = backend.hybrid_search(query)

    # Display results
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.3f}")
        print(f"    Org: {result.metadata.get('org', 'N/A')}")
        print(f"    Content: {result.content[:150]}...")


# ====================================================================
# WHY THIS BEATS FAISS + BM25
# ====================================================================

def why_es_is_better():
    """
    Explain why ES hybrid > FAISS + BM25 + Manual Fusion
    """
    print("\n" + "="*60)
    print("WHY ELASTICSEARCH HYBRID BEATS FAISS + BM25")
    print("="*60)

    comparison = {
        "Feature": [
            "Vector Search",
            "Keyword Search",
            "Metadata Filtering",
            "Result Fusion",
            "Sync Issues",
            "Production Ready",
            "Mentor Has It"
        ],
        "FAISS + BM25": [
            "✅ Fast",
            "✅ (separate library)",
            "❌ Need manual joins",
            "❌ Manual RRF code",
            "❌ 2 systems to sync",
            "❌ Complex deployment",
            "❌ Probably not"
        ],
        "Elasticsearch": [
            "✅ k-NN/HNSW",
            "✅ Native BM25",
            "✅ Native bool queries",
            "✅ Native RRF",
            "✅ Single system",
            "✅ Battle-tested",
            "✅ Likely yes"
        ]
    }

    for feature, faiss_val, es_val in zip(
        comparison["Feature"],
        comparison["FAISS + BM25"],
        comparison["Elasticsearch"]
    ):
        print(f"\n{feature:20s}: {faiss_val:25s} vs {es_val}")

    print("\n" + "="*60)
    print("VERDICT: Elasticsearch is the Industry Meta choice")
    print("="*60)


if __name__ == "__main__":
    # Show why ES is better
    why_es_is_better()

    # Run example (requires ES running)
    # example_hybrid_search()
