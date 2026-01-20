"""
Load Legal Document Corpus into ES + PostgreSQL

This script shows how to properly index legal documents organized by cases.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from elasticsearch import Elasticsearch, helpers
from typing import List, Dict, Any
import json


class LegalCorpusLoader:
    """
    Loads legal documents into ES + PostgreSQL.

    ES: Full-text search on documents
    PostgreSQL: Case metadata + knowledge graph
    """

    def __init__(
        self,
        es_host: str = "localhost",
        es_port: int = 9200,
        pg_host: str = "localhost",
        pg_database: str = "legal_corpus",
        pg_user: str = "legal",
        pg_password: str = "legal123"
    ):
        """Initialize connections"""
        # Elasticsearch
        self.es = Elasticsearch([f"http://{es_host}:{es_port}"])
        self.es_index = "legal_documents"

        # PostgreSQL
        self.pg_conn = psycopg2.connect(
            host=pg_host,
            database=pg_database,
            user=pg_user,
            password=pg_password
        )

        print("[Loader] Connected to ES and PostgreSQL")

        # Create ES index with legal document mapping
        self._create_es_index()

    def _create_es_index(self):
        """Create Elasticsearch index optimized for legal documents"""
        mapping = {
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "case_id": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {
                        "type": "text",
                        "analyzer": "english"
                    },
                    "metadata": {
                        "properties": {
                            "case_number": {"type": "keyword"},
                            "plaintiff": {"type": "text"},
                            "defendant": {"type": "text"},
                            "injury_type": {"type": "keyword"},
                            "jurisdiction": {"type": "keyword"},
                            "settlement_amount": {"type": "float"},
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

        if not self.es.indices.exists(index=self.es_index):
            self.es.indices.create(index=self.es_index, body=mapping)
            print(f"[ES] Created index: {self.es_index}")
        else:
            print(f"[ES] Index exists: {self.es_index}")

    def load_case_with_documents(self, case_data: Dict[str, Any], documents: List[Dict[str, Any]]):
        """
        Load a complete case with all its documents.

        Args:
            case_data: Case metadata (plaintiff, injury_type, etc.)
            documents: List of documents for this case
        """
        case_id = case_data["case_id"]

        print(f"\n[Loader] Loading case {case_id}...")

        # 1. Insert case into PostgreSQL
        self._insert_case(case_data)

        # 2. Extract entities and relations from documents
        entities, relations = self._extract_entities_and_relations(documents, case_id)

        # 3. Insert entities and relations into PostgreSQL
        self._insert_entities(entities)
        self._insert_relations(relations)

        # 4. Insert documents into PostgreSQL
        self._insert_documents(documents, case_id)

        # 5. Index documents into Elasticsearch (with enriched metadata)
        self._index_to_elasticsearch(documents, case_data)

        print(f"[Loader] ✓ Loaded {len(documents)} documents for case {case_id}")

    def _insert_case(self, case_data: Dict[str, Any]):
        """Insert case into PostgreSQL"""
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cases (case_id, case_number, plaintiff, defendant, injury_type, jurisdiction, settlement_amount, status)
                VALUES (%(case_id)s, %(case_number)s, %(plaintiff)s, %(defendant)s, %(injury_type)s, %(jurisdiction)s, %(settlement_amount)s, %(status)s)
                ON CONFLICT (case_id) DO UPDATE SET
                    plaintiff = EXCLUDED.plaintiff,
                    injury_type = EXCLUDED.injury_type,
                    settlement_amount = EXCLUDED.settlement_amount
            """, case_data)
            self.pg_conn.commit()

    def _extract_entities_and_relations(self, documents: List[Dict[str, Any]], case_id: str):
        """
        Extract entities and relations from documents.

        In production, you'd use NER here. For now, we use provided metadata.
        """
        entities = []
        relations = []

        for doc in documents:
            # Get entities from document metadata
            doc_entities = doc.get("metadata", {}).get("entities", [])
            entities.extend(doc_entities)

            # Get relations from document metadata
            doc_relations = doc.get("metadata", {}).get("relations", [])
            for rel in doc_relations:
                rel["case_id"] = case_id
            relations.extend(doc_relations)

        # Deduplicate entities by ID
        unique_entities = {e["id"]: e for e in entities}.values()

        return list(unique_entities), relations

    def _insert_entities(self, entities: List[Dict[str, Any]]):
        """Insert entities into PostgreSQL"""
        with self.pg_conn.cursor() as cur:
            for entity in entities:
                cur.execute("""
                    INSERT INTO entities (entity_id, entity_type, name, attributes)
                    VALUES (%(id)s, %(type)s, %(name)s, %(attributes)s)
                    ON CONFLICT (entity_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        attributes = EXCLUDED.attributes
                """, {
                    "id": entity["id"],
                    "type": entity["type"],
                    "name": entity["name"],
                    "attributes": json.dumps(entity.get("attributes", {}))
                })
            self.pg_conn.commit()

    def _insert_relations(self, relations: List[Dict[str, Any]]):
        """Insert relations into PostgreSQL"""
        with self.pg_conn.cursor() as cur:
            for rel in relations:
                cur.execute("""
                    INSERT INTO relations (source_id, target_id, relation_type, case_id)
                    VALUES (%(source)s, %(target)s, %(type)s, %(case_id)s)
                """, rel)
            self.pg_conn.commit()

    def _insert_documents(self, documents: List[Dict[str, Any]], case_id: str):
        """Insert documents into PostgreSQL"""
        with self.pg_conn.cursor() as cur:
            for doc in documents:
                cur.execute("""
                    INSERT INTO documents (document_id, case_id, doc_type, title, content)
                    VALUES (%(id)s, %(case_id)s, %(doc_type)s, %(title)s, %(content)s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        content = EXCLUDED.content
                """, {
                    "id": doc["id"],
                    "case_id": case_id,
                    "doc_type": doc.get("metadata", {}).get("doc_type", "unknown"),
                    "title": doc.get("metadata", {}).get("title", ""),
                    "content": doc["content"]
                })

                # Link document to entities
                entities = doc.get("metadata", {}).get("entities", [])
                for entity in entities:
                    cur.execute("""
                        INSERT INTO document_entities (document_id, entity_id)
                        VALUES (%(doc_id)s, %(entity_id)s)
                        ON CONFLICT DO NOTHING
                    """, {
                        "doc_id": doc["id"],
                        "entity_id": entity["id"]
                    })

            self.pg_conn.commit()

    def _index_to_elasticsearch(self, documents: List[Dict[str, Any]], case_data: Dict[str, Any]):
        """Index documents into Elasticsearch with enriched case metadata"""
        actions = []

        for doc in documents:
            # Enrich document with case metadata for better filtering
            action = {
                "_index": self.es_index,
                "_id": doc["id"],
                "_source": {
                    "document_id": doc["id"],
                    "case_id": case_data["case_id"],
                    "doc_type": doc.get("metadata", {}).get("doc_type", "unknown"),
                    "title": doc.get("metadata", {}).get("title", ""),
                    "content": doc["content"],
                    "metadata": {
                        "case_number": case_data.get("case_number"),
                        "plaintiff": case_data.get("plaintiff"),
                        "defendant": case_data.get("defendant"),
                        "injury_type": case_data.get("injury_type"),
                        "jurisdiction": case_data.get("jurisdiction"),
                        "settlement_amount": case_data.get("settlement_amount"),
                        "entities": doc.get("metadata", {}).get("entities", [])
                    }
                }
            }
            actions.append(action)

        # Bulk index
        success, failed = helpers.bulk(self.es, actions, stats_only=True)
        print(f"[ES] Indexed {success} documents ({failed} failed)")

    def search_documents(
        self,
        query: str,
        injury_type: str | None = None,
        jurisdiction: str | None = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents with filters.

        Example:
            results = loader.search_documents(
                "spinal injury settlement",
                injury_type="spinal_injury",
                jurisdiction="California"
            )
        """
        # Build ES query
        must = [{"match": {"content": query}}]

        if injury_type:
            must.append({"term": {"metadata.injury_type": injury_type}})

        if jurisdiction:
            must.append({"term": {"metadata.jurisdiction": jurisdiction}})

        es_query = {
            "query": {"bool": {"must": must}},
            "size": top_k
        }

        response = self.es.search(index=self.es_index, body=es_query)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "case_id": hit["_source"]["case_id"],
                "doc_type": hit["_source"]["doc_type"],
                "content": hit["_source"]["content"][:200] + "..."
            })

        return results

    def get_case_documents(self, case_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a case from PostgreSQL"""
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT d.*, array_agg(e.name) as entity_names
                FROM documents d
                LEFT JOIN document_entities de ON d.document_id = de.document_id
                LEFT JOIN entities e ON de.entity_id = e.entity_id
                WHERE d.case_id = %s
                GROUP BY d.document_id
            """, [case_id])

            return [dict(row) for row in cur.fetchall()]

    def close(self):
        """Close connections"""
        self.pg_conn.close()


# ====================================================================
# EXAMPLE USAGE
# ====================================================================

def example_load_corpus():
    """Example: Load a legal corpus"""
    loader = LegalCorpusLoader()

    # Define a case with documents
    case_1 = {
        "case_id": "case_99999",
        "case_number": "CV-2024-999",
        "plaintiff": "Alice Johnson",
        "defendant": "MegaCorp Inc.",
        "injury_type": "spinal_injury",
        "jurisdiction": "California",
        "settlement_amount": 600000.00,
        "status": "active"
    }

    documents_1 = [
        {
            "id": "case_99999_depo_1",
            "content": """
            Deposition of Alice Johnson in case CV-2024-999.

            Q: Can you describe the accident?
            A: I was working at the MegaCorp warehouse when a forklift operator
               struck me, causing severe spinal injuries. I sustained an L3-L4
               herniated disc that required surgery.

            Q: What was your occupation at the time?
            A: I was a warehouse supervisor at MegaCorp for 5 years.
            """,
            "metadata": {
                "doc_type": "deposition",
                "title": "Plaintiff Deposition",
                "entities": [
                    {"id": "plaintiff_alice", "name": "Alice Johnson", "type": "plaintiff"},
                    {"id": "injury_spinal_999", "name": "L3-L4 Herniated Disc", "type": "injury"}
                ],
                "relations": [
                    {"source": "plaintiff_alice", "target": "injury_spinal_999", "type": "sustained"}
                ]
            }
        },
        {
            "id": "case_99999_demand_1",
            "content": """
            Demand Letter - Case CV-2024-999

            We demand settlement in the amount of $600,000 for injuries sustained
            by our client, Alice Johnson. Similar spinal injury cases in California
            have settled in the range of $500,000-$750,000.

            The defendant's negligence in warehouse operations directly caused
            our client's L3-L4 herniated disc, requiring surgical intervention
            and resulting in permanent disability.
            """,
            "metadata": {
                "doc_type": "demand_letter",
                "title": "Demand Letter",
                "entities": [
                    {"id": "plaintiff_alice", "name": "Alice Johnson", "type": "plaintiff"}
                ],
                "relations": []
            }
        }
    ]

    # Load the case
    loader.load_case_with_documents(case_1, documents_1)

    # Test search
    print("\n[Test] Searching for 'spinal injury California'...")
    results = loader.search_documents(
        "spinal injury",
        injury_type="spinal_injury",
        jurisdiction="California"
    )

    for r in results:
        print(f"\n  Case: {r['case_id']}, Doc: {r['doc_type']}, Score: {r['score']:.2f}")
        print(f"  Preview: {r['content']}")

    loader.close()


if __name__ == "__main__":
    example_load_corpus()
