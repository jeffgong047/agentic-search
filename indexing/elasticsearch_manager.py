"""
Elasticsearch Integration Layer
Provides interface for indexing and searching legal documents
"""

from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch, helpers
from config import get_config
import logging


class ElasticsearchManager:
    """
    Manages Elasticsearch connection and operations.
    Handles both indexing and search with metadata filtering.
    """

    def __init__(self, host: str | None = None, port: int | None = None, index_name: str | None = None):
        """
        Initialize Elasticsearch connection.

        Args:
            host: Elasticsearch host (uses config default if None)
            port: Elasticsearch port (uses config default if None)
            index_name: Index name (uses config default if None)
        """
        config = get_config()

        self.host = host or config.ES_HOST
        self.port = port or config.ES_PORT
        self.index_name = index_name or config.ES_INDEX_NAME

        # Initialize client (will connect when needed)
        self.client: Elasticsearch | None = None
        self.is_connected = False

    def connect(self) -> bool:
        """
        Connect to Elasticsearch.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = Elasticsearch([f"http://{self.host}:{self.port}"])
            self.is_connected = self.client.ping()

            if self.is_connected:
                print(f"[ES] Connected to Elasticsearch at {self.host}:{self.port}")
            else:
                print(f"[ES] Failed to ping Elasticsearch at {self.host}:{self.port}")

            return self.is_connected
        except Exception as e:
            print(f"[ES] Connection error: {e}")
            self.is_connected = False
            return False

    def create_index(self, mapping: Dict[str, Any] | None = None) -> bool:
        """
        Create the Elasticsearch index with mapping.

        Args:
            mapping: Index mapping configuration (uses default legal doc mapping if None)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or self.client is None:
            print("[ES] Not connected. Call connect() first.")
            return False

        if mapping is None:
            mapping = self._get_default_mapping()

        try:
            if self.client.indices.exists(index=self.index_name):
                print(f"[ES] Index '{self.index_name}' already exists")
                return True

            self.client.indices.create(index=self.index_name, body=mapping)
            print(f"[ES] Created index '{self.index_name}'")
            return True
        except Exception as e:
            print(f"[ES] Error creating index: {e}")
            return False

    def index_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Bulk index documents into Elasticsearch.

        Args:
            documents: List of documents with keys: id, content, metadata

        Returns:
            Number of successfully indexed documents
        """
        if not self.is_connected or self.client is None:
            print("[ES] Not connected. Call connect() first.")
            return 0

        # Prepare bulk actions
        actions = []
        for doc in documents:
            action = {
                "_index": self.index_name,
                "_id": doc["id"],
                "_source": {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
            }
            actions.append(action)

        try:
            success, failed = helpers.bulk(self.client, actions, stats_only=True)
            print(f"[ES] Indexed {success} documents ({failed} failed)")
            return success
        except Exception as e:
            print(f"[ES] Bulk indexing error: {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_constraints: Dict[str, Any] | None = None,
        negative_constraints: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Search Elasticsearch with filters and negative constraints.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_constraints: Metadata filters to apply (MUST match)
            negative_constraints: Terms that MUST NOT appear

        Returns:
            List of search results with id, content, metadata, score
        """
        if not self.is_connected or self.client is None:
            print("[ES] Not connected. Call connect() first.")
            return []

        # Build Elasticsearch query
        must_clauses = [
            {"match": {"content": query}}
        ]

        # Add filter constraints as MUST clauses
        if filter_constraints:
            for key, value in filter_constraints.items():
                # Determine field path - some metadata fields are already keywords
                # Check mapping to see if field is text (needs .keyword) or already keyword
                field_path = f"metadata.{key}"

                # Common text fields that need .keyword suffix
                text_fields = ["document_type", "name", "title", "description"]
                if key in text_fields:
                    field_path = f"metadata.{key}.keyword"

                # Handle list values with "terms" (plural) query
                if isinstance(value, list):
                    must_clauses.append({
                        "terms": {field_path: value}
                    })
                else:
                    must_clauses.append({
                        "term": {field_path: value}
                    })

        # Add negative constraints as MUST NOT clauses
        must_not_clauses = []
        if negative_constraints:
            for term in negative_constraints:
                must_not_clauses.append({
                    "match": {"content": term}
                })

        # Construct full query
        es_query = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "must_not": must_not_clauses
                }
            },
            "size": top_k
        }

        try:
            response = self.client.search(index=self.index_name, body=es_query)
            hits = response["hits"]["hits"]

            results = []
            for hit in hits:
                results.append({
                    "id": hit["_id"],
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"]
                })

            return results
        except Exception as e:
            print(f"[ES] Search error: {e}")
            return []

    def delete_index(self) -> bool:
        """Delete the index"""
        if not self.is_connected or self.client is None:
            return False

        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                print(f"[ES] Deleted index '{self.index_name}'")
            return True
        except Exception as e:
            print(f"[ES] Error deleting index: {e}")
            return False

    def _get_default_mapping(self) -> Dict[str, Any]:
        """Get default mapping for legal documents"""
        return {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "org": {"type": "keyword"},
                            "year": {"type": "integer"},
                            "type": {"type": "keyword"},
                            "entities": {
                                "type": "nested",
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "name": {"type": "text"},
                                    "type": {"type": "keyword"}
                                }
                            }
                        }
                    }
                }
            }
        }

    def get_document_count(self) -> int:
        """Get total number of documents in the index"""
        if not self.is_connected or self.client is None:
            return 0

        try:
            count = self.client.count(index=self.index_name)
            return count["count"]
        except Exception as e:
            print(f"[ES] Error counting documents: {e}")
            return 0
