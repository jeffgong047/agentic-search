"""
Custom Backend Example

This shows how to plug in your own retrieval system, database, or API.

The agent will work with ANYTHING as long as it implements the SearchBackend interface.
"""

from typing import List, Dict, Any
from interfaces import SearchBackend, BackendFactory
from data_structures import SearchResult, SearchPlan
from main import RetrievalAgent


# ====================================================================
# EXAMPLE 1: Custom API Backend
# ====================================================================

class CustomAPIBackend(SearchBackend):
    """
    Example: Wrap your mentor's custom retrieval API.

    This shows how to integrate with ANY external system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with API configuration.

        Args:
            config: {"endpoint": "http://api.company.com", "api_key": "..."}
        """
        self.endpoint = config.get("endpoint")
        self.api_key = config.get("api_key")

        print(f"[CustomAPI] Connected to {self.endpoint}")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Send documents to external API for indexing.
        """
        import requests

        # Call your API's indexing endpoint
        response = requests.post(
            f"{self.endpoint}/index",
            json={"documents": documents},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

        if response.status_code == 200:
            print(f"[CustomAPI] Indexed {len(documents)} documents")
        else:
            raise Exception(f"Indexing failed: {response.text}")

    def search(
        self,
        query: str,
        search_plan: SearchPlan,
        negative_cache: List[Dict[str, str]] | None = None
    ) -> List[SearchResult]:
        """
        Call your API's search endpoint.
        """
        import requests

        # Build request payload
        payload = {
            "query": query,
            "filters": search_plan.filter_constraints,
            "negative_terms": search_plan.negative_constraints,
            "top_k": 10
        }

        # Call API
        response = requests.post(
            f"{self.endpoint}/search",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Search failed: {response.text}")

        data = response.json()

        # Convert API response to SearchResult format
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                id=item["doc_id"],
                content=item["text"],
                score=item["score"],
                metadata=item.get("metadata", {}),
                source_index="custom_api"
            ))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "backend_type": "custom_api",
            "endpoint": self.endpoint
        }


# ====================================================================
# EXAMPLE 2: Custom Database Backend
# ====================================================================

class CustomDatabaseBackend(SearchBackend):
    """
    Example: Use your own database (not ES).

    This could be MongoDB, Cassandra, or any database.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database connection.

        Args:
            config: {"db_url": "mongodb://...", "collection": "..."}
        """
        # Example with MongoDB
        from pymongo import MongoClient

        self.client = MongoClient(config.get("db_url"))
        self.db = self.client[config.get("database", "retrieval")]
        self.collection = self.db[config.get("collection", "documents")]

        print(f"[CustomDB] Connected to MongoDB")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Insert documents into MongoDB"""
        self.collection.insert_many(documents)
        print(f"[CustomDB] Indexed {len(documents)} documents")

    def search(
        self,
        query: str,
        search_plan: SearchPlan,
        negative_cache: List[Dict[str, str]] | None = None
    ) -> List[SearchResult]:
        """
        Search MongoDB using text index.
        """
        # Build MongoDB query
        mongo_query = {"$text": {"$search": query}}

        # Add filters
        if search_plan.filter_constraints:
            for key, value in search_plan.filter_constraints.items():
                mongo_query[f"metadata.{key}"] = value

        # Add negative constraints
        if search_plan.negative_constraints:
            mongo_query["content"] = {
                "$not": {
                    "$regex": "|".join(search_plan.negative_constraints),
                    "$options": "i"
                }
            }

        # Execute query
        cursor = self.collection.find(
            mongo_query,
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(10)

        # Convert to SearchResult
        results = []
        for doc in cursor:
            results.append(SearchResult(
                id=doc["id"],
                content=doc["content"],
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
                source_index="mongodb"
            ))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "backend_type": "mongodb",
            "document_count": self.collection.count_documents({})
        }


# ====================================================================
# EXAMPLE 3: File-Based Backend (Ultra Simple)
# ====================================================================

class FileBasedBackend(SearchBackend):
    """
    Example: Use local JSON files.

    Useful for debugging without any external dependencies.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize file-based backend.

        Args:
            config: {"data_dir": "/path/to/data"}
        """
        import os
        self.data_dir = config.get("data_dir", "./data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.documents = []

        print(f"[FileBackend] Using {self.data_dir}")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Save documents to JSON file"""
        import json

        self.documents = documents

        with open(f"{self.data_dir}/documents.json", "w") as f:
            json.dump(documents, f, indent=2)

        print(f"[FileBackend] Saved {len(documents)} documents")

    def search(
        self,
        query: str,
        search_plan: SearchPlan,
        negative_cache: List[Dict[str, str]] | None = None
    ) -> List[SearchResult]:
        """Simple keyword search on documents"""
        query_lower = query.lower()
        results = []

        for doc in self.documents:
            # Simple scoring: count query words in content
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)

            # Apply filters
            if search_plan.filter_constraints:
                match = all(
                    doc.get("metadata", {}).get(k) == v
                    for k, v in search_plan.filter_constraints.items()
                )
                if not match:
                    continue

            # Apply negative constraints
            if search_plan.negative_constraints:
                has_negative = any(
                    neg.lower() in content_lower
                    for neg in search_plan.negative_constraints
                )
                if has_negative:
                    continue

            if score > 0:
                results.append(SearchResult(
                    id=doc["id"],
                    content=doc["content"],
                    score=float(score),
                    metadata=doc.get("metadata", {}),
                    source_index="file"
                ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:10]

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "backend_type": "file_based",
            "document_count": len(self.documents),
            "data_dir": self.data_dir
        }


# ====================================================================
# HOW TO USE CUSTOM BACKENDS
# ====================================================================

def demo_custom_api():
    """Demo using custom API backend"""
    print("\n" + "="*60)
    print("DEMO: Custom API Backend")
    print("="*60)

    # Register your custom backend
    BackendFactory.register("custom_api", CustomAPIBackend)

    # Create agent with custom backend
    agent = RetrievalAgent(backend_type="custom_api")

    # Note: You'd need to configure the API endpoint
    # agent.backend.endpoint = "http://your-api.com"
    # agent.backend.api_key = "your-key"

    print("Agent is ready to use your custom API!")


def demo_custom_database():
    """Demo using custom database backend"""
    print("\n" + "="*60)
    print("DEMO: Custom Database Backend")
    print("="*60)

    # Register custom backend
    BackendFactory.register("mongodb", CustomDatabaseBackend)

    # Use it (if MongoDB is running)
    try:
        from main import RetrievalAgent

        # Note: This requires MongoDB running
        # agent = RetrievalAgent(backend_type="mongodb")
        print("To use: agent = RetrievalAgent(backend_type='mongodb')")

    except Exception as e:
        print(f"MongoDB not available: {e}")


def demo_file_based():
    """Demo using file-based backend (works everywhere!)"""
    print("\n" + "="*60)
    print("DEMO: File-Based Backend (No External Dependencies)")
    print("="*60)

    # Register file backend
    BackendFactory.register("file", FileBasedBackend)

    # Create agent
    agent = RetrievalAgent(backend_type="file")

    # Load mock data
    from mock_data import get_mock_dataset
    agent.load_data(get_mock_dataset())

    # Search
    results = agent.search("Mickey Mouse Meta")

    print(f"\nFound {len(results)} results using file-based backend")
    for r in results[:3]:
        print(f"  {r.id}: {r.score}")


# ====================================================================
# THE KEY INSIGHT
# ====================================================================

"""
THE MAGIC: The agent doesn't care what backend you use!

As long as your backend implements:
  - index_documents(documents)
  - search(query, search_plan, negative_cache)
  - get_stats()

The agent will work perfectly.

This means:
✅ Your mentor can use their existing retrieval system
✅ You can debug locally with simple file-based backend
✅ Production can use Elasticsearch
✅ Future can switch to Pinecone/Weaviate/whatever

NO CODE CHANGES NEEDED in the agent itself!
"""


def main():
    """Run demos"""
    import sys

    if "--api" in sys.argv:
        demo_custom_api()
    elif "--db" in sys.argv:
        demo_custom_database()
    else:
        # File-based works everywhere
        demo_file_based()


if __name__ == "__main__":
    main()
