"""
Tool Adapters: How to wrap external tools for the agent to use

This shows the pattern for connecting the agent to:
1. External Elasticsearch clusters
2. PostgreSQL knowledge graphs
3. Neo4j graph databases
4. Custom APIs
"""

from typing import List, Dict, Any, Set
import psycopg2
from psycopg2.extras import RealDictCursor
from elasticsearch import Elasticsearch


# ====================================================================
# PATTERN 1: Elasticsearch Adapter
# ====================================================================

class ElasticsearchAdapter:
    """
    Adapter for connecting to external Elasticsearch clusters.

    Usage:
        es_adapter = ElasticsearchAdapter(
            host="your-es-host.com",
            port=9200,
            index="your_index"
        )

        # Agent uses this internally
        results = es_adapter.search("query", filters={"field": "value"})
    """

    def __init__(self, host: str, port: int = 9200, index: str = "documents", **kwargs):
        """
        Initialize Elasticsearch adapter.

        Args:
            host: ES host (e.g., "localhost" or "es.company.com")
            port: ES port (default: 9200)
            index: Index name
            **kwargs: Additional ES client options (e.g., http_auth, use_ssl)
        """
        self.host = host
        self.port = port
        self.index = index

        # Initialize ES client
        self.client = Elasticsearch(
            [f"{host}:{port}"],
            **kwargs
        )

        # Test connection
        if self.client.ping():
            print(f"[ES Adapter] Connected to {host}:{port}")
        else:
            raise ConnectionError(f"Cannot connect to Elasticsearch at {host}:{port}")

    def search(
        self,
        query: str,
        filters: Dict[str, Any] | None = None,
        negative_constraints: List[str] | None = None,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search documents with filters.

        Args:
            query: Search query
            filters: Metadata filters (must match)
            negative_constraints: Terms that must NOT appear
            top_k: Number of results

        Returns:
            List of documents with id, content, metadata, score
        """
        # Build ES query
        must = [{"match": {"content": query}}]

        # Add filters
        if filters:
            for key, value in filters.items():
                must.append({"term": {f"metadata.{key}": value}})

        # Add negative constraints
        must_not = []
        if negative_constraints:
            for term in negative_constraints:
                must_not.append({"match": {"content": term}})

        # Construct query
        es_query = {
            "query": {
                "bool": {
                    "must": must,
                    "must_not": must_not
                }
            },
            "size": top_k
        }

        # Execute search
        response = self.client.search(index=self.index, body=es_query)

        # Parse results
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "score": hit["_score"]
            })

        return results

    def index_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Bulk index documents"""
        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": self.index,
                "_id": doc["id"],
                "_source": {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
            }
            for doc in documents
        ]

        success, _ = bulk(self.client, actions)
        return success


# ====================================================================
# PATTERN 2: PostgreSQL Knowledge Graph Adapter
# ====================================================================

class PostgreSQLGraphAdapter:
    """
    Adapter for PostgreSQL-based knowledge graph.

    Usage:
        pg_adapter = PostgreSQLGraphAdapter(
            host="localhost",
            database="knowledge_graph",
            user="retrieval",
            password="retrieval123"
        )

        # Query related entities
        related = pg_adapter.find_related_entities("mickey_mouse_meta", depth=2)
    """

    def __init__(self, host: str, database: str, user: str, password: str, port: int = 5432):
        """Initialize PostgreSQL connection"""
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        print(f"[PG Adapter] Connected to {host}:{port}/{database}")

    def find_related_entities(
        self,
        entity_id: str,
        max_depth: int = 2,
        entity_types: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity.

        Args:
            entity_id: Starting entity ID
            max_depth: Maximum traversal depth
            entity_types: Filter by entity types

        Returns:
            List of related entities with distance
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Recursive CTE to traverse graph
            query = """
            WITH RECURSIVE entity_path AS (
                -- Base case: starting entity
                SELECT id, name, type, attributes, 0 as depth
                FROM entities
                WHERE id = %s

                UNION

                -- Recursive case: follow edges
                SELECT e.id, e.name, e.type, e.attributes, ep.depth + 1
                FROM entities e
                JOIN relations r ON (r.source_id = e.id OR r.target_id = e.id)
                JOIN entity_path ep ON (
                    (r.source_id = ep.id AND e.id = r.target_id) OR
                    (r.target_id = ep.id AND e.id = r.source_id)
                )
                WHERE ep.depth < %s
            )
            SELECT DISTINCT id, name, type, attributes, depth
            FROM entity_path
            """

            params = [entity_id, max_depth]

            # Add type filter if specified
            if entity_types:
                query += " WHERE type = ANY(%s)"
                params.append(entity_types)

            query += " ORDER BY depth, name"

            cur.execute(query, params)
            results = cur.fetchall()

            return [dict(row) for row in results]

    def get_entity_documents(self, entity_id: str) -> List[str]:
        """Get all documents mentioning an entity"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT document_id FROM document_entities WHERE entity_id = %s",
                [entity_id]
            )
            return [row[0] for row in cur.fetchall()]

    def add_entity(self, entity_id: str, name: str, entity_type: str, attributes: Dict | None = None):
        """Add an entity to the knowledge graph"""
        import json

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entities (id, name, type, attributes)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    attributes = EXCLUDED.attributes
                """,
                [entity_id, name, entity_type, json.dumps(attributes or {})]
            )
            self.conn.commit()

    def add_relation(self, source_id: str, target_id: str, relation_type: str):
        """Add a relation between entities"""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO relations (source_id, target_id, relation_type)
                VALUES (%s, %s, %s)
                """,
                [source_id, target_id, relation_type]
            )
            self.conn.commit()

    def close(self):
        """Close connection"""
        self.conn.close()


# ====================================================================
# PATTERN 3: Generic Tool Wrapper
# ====================================================================

class ToolWrapper:
    """
    Generic pattern for wrapping any external tool/API.

    The agent calls methods on this class, and it translates
    to whatever API the external tool expects.
    """

    def __init__(self, endpoint: str, api_key: str | None = None):
        """
        Initialize tool wrapper.

        Args:
            endpoint: API endpoint URL
            api_key: API key for authentication (if needed)
        """
        self.endpoint = endpoint
        self.api_key = api_key
        print(f"[Tool Wrapper] Connected to {endpoint}")

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Generic search method.

        This is where you translate the agent's request to your API's format.
        """
        import requests

        # Build request
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "query": query,
            **kwargs
        }

        # Call external API
        response = requests.post(
            f"{self.endpoint}/search",
            json=payload,
            headers=headers
        )

        # Parse response
        if response.status_code == 200:
            data = response.json()
            # Translate to agent's expected format
            return [
                {
                    "id": item["doc_id"],
                    "content": item["text"],
                    "metadata": item.get("metadata", {}),
                    "score": item.get("relevance", 0.0)
                }
                for item in data.get("results", [])
            ]
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")


# ====================================================================
# EXAMPLE: How Agent Uses These Adapters
# ====================================================================

def example_usage():
    """
    Example showing how the agent uses tool adapters
    """

    # Option 1: Use Elasticsearch adapter
    es_tool = ElasticsearchAdapter(
        host="localhost",
        port=9200,
        index="legal_docs"
    )

    # Agent calls this internally
    results = es_tool.search(
        query="non-compete agreement",
        filters={"org": "Meta"},
        negative_constraints=["Shanghai"]
    )

    print(f"Found {len(results)} documents from ES")

    # Option 2: Use PostgreSQL graph adapter
    pg_tool = PostgreSQLGraphAdapter(
        host="localhost",
        database="knowledge_graph",
        user="retrieval",
        password="retrieval123"
    )

    # Agent calls this internally
    related = pg_tool.find_related_entities("mickey_mouse_meta", max_depth=2)
    print(f"Found {len(related)} related entities")

    # Get documents for entity
    docs = pg_tool.get_entity_documents("mickey_mouse_meta")
    print(f"Entity appears in {len(docs)} documents")

    pg_tool.close()


if __name__ == "__main__":
    example_usage()
