"""
Knowledge Graph Builder
Constructs and manages case-level knowledge graphs for reasoning
"""

from typing import List, Dict, Any, Set, Tuple
import networkx as nx


class KnowledgeGraphBuilder:
    """
    Builds and manages a knowledge graph from legal documents.
    Supports entity extraction, relationship linking, and graph reasoning.
    """

    def __init__(self):
        """Initialize the knowledge graph builder"""
        self.graph = nx.MultiDiGraph()
        self.entity_index: Dict[str, Dict[str, Any]] = {}  # entity_id -> entity data
        self.doc_to_entities: Dict[str, Set[str]] = {}  # doc_id -> entity_ids
        self.entity_to_docs: Dict[str, Set[str]] = {}  # entity_id -> doc_ids

    def add_document(
        self,
        doc_id: str,
        content: str,
        entities: List[Dict[str, str]],
        relations: List[Dict[str, str]] | None = None
    ) -> None:
        """
        Add a document and its entities/relations to the knowledge graph.

        Args:
            doc_id: Unique document identifier
            content: Document text
            entities: List of entities with keys: id, name, type
            relations: List of relations with keys: source, target, type
        """
        self.doc_to_entities[doc_id] = set()

        # Add entities
        for entity in entities:
            entity_id = entity.get("id") or entity.get("name", "")
            if not entity_id:
                continue

            entity_name = entity.get("name", entity_id)
            entity_type = entity.get("type", "unknown")

            # Add to entity index
            if entity_id not in self.entity_index:
                self.entity_index[entity_id] = {
                    "name": entity_name,
                    "type": entity_type,
                    "attributes": entity.get("attributes", {})
                }

            # Add entity node to graph
            if not self.graph.has_node(entity_id):
                self.graph.add_node(
                    entity_id,
                    name=entity_name,
                    type=entity_type,
                    attributes=entity.get("attributes", {})
                )

            # Link entity to document
            self.doc_to_entities[doc_id].add(entity_id)

            if entity_id not in self.entity_to_docs:
                self.entity_to_docs[entity_id] = set()
            self.entity_to_docs[entity_id].add(doc_id)

        # Add relations
        if relations:
            for relation in relations:
                source = relation.get("source")
                target = relation.get("target")
                rel_type = relation.get("type", "related_to")

                if source and target and source in self.entity_index and target in self.entity_index:
                    self.graph.add_edge(
                        source,
                        target,
                        type=rel_type,
                        metadata=relation.get("metadata", {})
                    )

    def find_related_entities(
        self,
        entity_query: str,
        max_depth: int = 2,
        entity_types: List[str] | None = None
    ) -> List[Tuple[str, int, Dict[str, Any]]]:
        """
        Find entities related to the query entity.

        Args:
            entity_query: Entity name or ID to search from
            max_depth: Maximum graph traversal depth
            entity_types: Filter by entity types (e.g., ['person', 'organization'])

        Returns:
            List of tuples: (entity_id, distance, entity_data)
        """
        # Find matching entities
        seed_entities = self._find_matching_entities(entity_query)

        if not seed_entities:
            return []

        # BFS traversal
        related: Dict[str, int] = {}
        for seed in seed_entities:
            visited = self._bfs_traversal(seed, max_depth)
            for entity_id, dist in visited.items():
                if entity_id not in related or dist < related[entity_id]:
                    related[entity_id] = dist

        # Filter by entity type if specified
        results = []
        for entity_id, dist in related.items():
            entity_data = self.entity_index.get(entity_id, {})

            if entity_types:
                if entity_data.get("type") not in entity_types:
                    continue

            results.append((entity_id, dist, entity_data))

        # Sort by distance (closer first)
        results.sort(key=lambda x: x[1])

        return results

    def find_path(self, source_entity: str, target_entity: str, max_length: int = 5) -> List[List[str]] | None:
        """
        Find paths between two entities.

        Args:
            source_entity: Source entity ID or name
            target_entity: Target entity ID or name
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of entity IDs)
        """
        source_ids = self._find_matching_entities(source_entity)
        target_ids = self._find_matching_entities(target_entity)

        if not source_ids or not target_ids:
            return None

        paths = []
        for src in source_ids:
            for tgt in target_ids:
                try:
                    # Find all simple paths up to max_length
                    for path in nx.all_simple_paths(self.graph, src, tgt, cutoff=max_length):
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue

        return paths if paths else None

    def get_entity_attributes(self, entity_id: str) -> Dict[str, Any] | None:
        """Get attributes of an entity"""
        return self.entity_index.get(entity_id)

    def get_entity_documents(self, entity_id: str) -> Set[str]:
        """Get all documents mentioning an entity"""
        return self.entity_to_docs.get(entity_id, set())

    def _find_matching_entities(self, query: str) -> Set[str]:
        """Find entity IDs matching the query string"""
        matched = set()
        query_lower = query.lower()

        # Check entity index
        for entity_id, entity_data in self.entity_index.items():
            # Exact match on ID
            if entity_id.lower() == query_lower:
                matched.add(entity_id)
                continue

            # Fuzzy match on name
            entity_name = entity_data.get("name", "").lower()
            if query_lower in entity_name or entity_name in query_lower:
                matched.add(entity_id)

        return matched

    def _bfs_traversal(self, start: str, max_depth: int) -> Dict[str, int]:
        """
        BFS traversal from start node up to max_depth.

        Returns:
            Dict mapping entity_id -> distance from start
        """
        if start not in self.graph.nodes:
            return {}

        visited: Dict[str, int] = {start: 0}
        queue = [(start, 0)]

        while queue:
            node, dist = queue.pop(0)

            if dist >= max_depth:
                continue

            # Add neighbors (both outgoing and incoming edges)
            for neighbor in self.graph.successors(node):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

            for neighbor in self.graph.predecessors(node):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return visited

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        return {
            "num_entities": len(self.entity_index),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_documents": len(self.doc_to_entities),
            "avg_entities_per_doc": len(self.doc_to_entities) / max(1, len(self.doc_to_entities)),
            "entity_types": self._get_entity_type_counts()
        }

    def _get_entity_type_counts(self) -> Dict[str, int]:
        """Count entities by type"""
        type_counts: Dict[str, int] = {}
        for entity_data in self.entity_index.values():
            entity_type = entity_data.get("type", "unknown")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts

    def export_graph(self, filepath: str, format: str = "gexf") -> bool:
        """
        Export the graph to a file.

        Args:
            filepath: Output file path
            format: Graph format (gexf, graphml, gml, etc.)

        Returns:
            True if successful
        """
        try:
            if format == "gexf":
                nx.write_gexf(self.graph, filepath)
            elif format == "graphml":
                nx.write_graphml(self.graph, filepath)
            elif format == "gml":
                nx.write_gml(self.graph, filepath)
            else:
                print(f"Unsupported format: {format}")
                return False

            print(f"[KG] Exported graph to {filepath}")
            return True
        except Exception as e:
            print(f"[KG] Export error: {e}")
            return False
