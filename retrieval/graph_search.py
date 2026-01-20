"""
Graph Search Engine: Knowledge graph traversal for entity relationships
"""

from typing import List, Dict, Any, Set
import networkx as nx
from data_structures import SearchResult
from config import get_config


class GraphSearchEngine:
    """
    Relational retrieval using a knowledge graph.
    Handles entity disambiguation through relationship traversal.
    """

    def __init__(self):
        """Initialize the knowledge graph"""
        self.graph = nx.MultiDiGraph()
        self.entity_to_docs: Dict[str, Set[str]] = {}  # entity_id -> set of doc_ids
        self.doc_store: Dict[str, Dict[str, Any]] = {}  # doc_id -> document data

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the knowledge graph.

        Args:
            documents: List of dicts with keys: id, content, metadata
                      metadata should include 'entities' and 'relations'
        """
        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})

            # Store document
            self.doc_store[doc_id] = {
                "content": doc["content"],
                "metadata": metadata
            }

            # Extract entities from metadata
            entities = metadata.get("entities", [])
            for entity in entities:
                entity_id = entity.get("id") or entity.get("name")
                entity_type = entity.get("type", "unknown")

                # Add entity node
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id, type=entity_type, label=entity.get("name", entity_id))

                # Link entity to document
                if entity_id not in self.entity_to_docs:
                    self.entity_to_docs[entity_id] = set()
                self.entity_to_docs[entity_id].add(doc_id)

            # Add relationships
            relations = metadata.get("relations", [])
            for relation in relations:
                source = relation.get("source")
                target = relation.get("target")
                rel_type = relation.get("type", "related_to")

                if source and target:
                    self.graph.add_edge(source, target, type=rel_type)

    def search(
        self,
        query_entities: List[str],
        depth: int = 1,
        top_k: int | None = None,
        filter_constraints: Dict[str, Any] | None = None
    ) -> List[SearchResult]:
        """
        Perform graph traversal search starting from query entities.

        Args:
            query_entities: List of entity names/IDs to start search from
            depth: Maximum depth of graph traversal (0-2)
            top_k: Number of results to return
            filter_constraints: Metadata filters to apply

        Returns:
            List of SearchResult objects
        """
        config = get_config()
        top_k = top_k or config.VECTOR_TOP_K

        if not query_entities:
            return []

        # Find matching entities in the graph
        seed_entities = self._find_entities(query_entities)

        if not seed_entities:
            return []

        # Traverse graph to collect related entities
        related_entities = self._traverse_graph(seed_entities, depth)

        # Collect documents linked to these entities
        doc_scores: Dict[str, float] = {}
        for entity in related_entities:
            entity_dist = related_entities[entity]
            docs = self.entity_to_docs.get(entity, set())

            for doc_id in docs:
                # Score based on graph distance (closer = higher score)
                score = 1.0 / (1.0 + entity_dist)

                if doc_id in doc_scores:
                    doc_scores[doc_id] = max(doc_scores[doc_id], score)
                else:
                    doc_scores[doc_id] = score

        # Sort by score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Build SearchResult objects
        results = []
        for doc_id, score in ranked_docs:
            doc_data = self.doc_store.get(doc_id)
            if doc_data is None:
                continue

            # Apply metadata filters
            if filter_constraints and not self._matches_filters(doc_data["metadata"], filter_constraints):
                continue

            results.append(SearchResult(
                id=doc_id,
                content=doc_data["content"],
                score=score,
                metadata=doc_data["metadata"],
                source_index="graph"
            ))

            if len(results) >= top_k:
                break

        return results

    def _find_entities(self, query_terms: List[str]) -> Set[str]:
        """Find entity IDs matching query terms"""
        matched = set()

        for term in query_terms:
            term_lower = term.lower()

            # Exact match
            if term in self.graph.nodes:
                matched.add(term)
                continue

            # Fuzzy match on node labels
            for node in self.graph.nodes:
                node_data = self.graph.nodes[node]
                label = node_data.get("label", node).lower()

                if term_lower in label or label in term_lower:
                    matched.add(node)

        return matched

    def _traverse_graph(self, seed_entities: Set[str], max_depth: int) -> Dict[str, int]:
        """
        Traverse graph from seed entities up to max_depth.

        Returns:
            Dict mapping entity_id -> distance from seed
        """
        visited: Dict[str, int] = {}

        for seed in seed_entities:
            if seed not in self.graph.nodes:
                continue

            # BFS from this seed
            queue = [(seed, 0)]
            while queue:
                entity, dist = queue.pop(0)

                if entity in visited:
                    continue

                visited[entity] = dist

                if dist < max_depth:
                    # Add neighbors
                    for neighbor in self.graph.neighbors(entity):
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))

        return visited

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches filter constraints"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def get_graph_stats(self) -> Dict[str, int]:
        """Return statistics about the knowledge graph"""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "documents": len(self.doc_store)
        }
