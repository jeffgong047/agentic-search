"""
Raw Elasticsearch Ingestion Pipeline

This is the "production" implementation.
Direct ES calls, no LlamaIndex dependency.

Use this when:
- Mentor has existing ES infrastructure
- Want maximum control over chunking
- Don't want LlamaIndex dependency
"""

from typing import List, Dict, Any
from elasticsearch import Elasticsearch, helpers
import re
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.production_interfaces import DocumentIngestionPipeline


class RawElasticsearchPipeline(DocumentIngestionPipeline):
    """
    Raw Elasticsearch implementation.

    Pros:
    - No extra dependencies
    - Full control over logic
    - Matches mentor's infrastructure
    - Production-ready

    Cons:
    - More code to write
    - Need to handle chunking manually

    SWAPPABLE: Can replace LlamaIndexPipeline with this
    """

    def __init__(self):
        """Initialize raw ES pipeline"""
        print("[RawES] Pipeline initialized")

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents using simple sentence-based splitting.

        Args:
            documents: List of dicts with 'id', 'content', 'metadata'

        Returns:
            List of chunks
        """
        chunks = []

        for doc in documents:
            content = doc["content"]
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})

            # Split into sentences
            sentences = self._split_sentences(content)

            # Group into chunks
            doc_chunks = self._group_sentences(
                sentences,
                chunk_size,
                chunk_overlap
            )

            # Create chunk objects
            for i, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    "id": f"{doc_id}_chunk_{i}",
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "source_doc_id": doc_id,
                        "chunk_index": i
                    }
                })

        print(f"[RawES] Chunked {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses simple regex. In production, use spaCy or nltk for better accuracy.
        """
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_sentences(
        self,
        sentences: List[str],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Group sentences into chunks with overlap.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_words = int(chunk_overlap)
                current_chunk = current_chunk[-overlap_words:] if len(current_chunk) > overlap_words else current_chunk
                current_length = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from document.

        This is where you'd plug in your entity extraction logic.
        """
        metadata = document.get("metadata", {}).copy()

        # Example: Simple entity extraction (replace with real NER)
        content = document.get("content", "")

        # Extract case IDs (pattern: case_XXXXX)
        case_ids = re.findall(r'case_\d+', content.lower())
        if case_ids:
            metadata["extracted_case_ids"] = list(set(case_ids))

        # Extract dollar amounts (pattern: $XXX,XXX)
        amounts = re.findall(r'\$[\d,]+', content)
        if amounts:
            metadata["mentioned_amounts"] = amounts

        # Extract proper nouns (simple heuristic)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content)
        if proper_nouns:
            metadata["mentioned_names"] = list(set(proper_nouns))[:10]  # Limit to 10

        return metadata

    def index_to_backend(
        self,
        chunks: List[Dict[str, Any]],
        backend_config: Dict[str, Any]
    ) -> int:
        """
        Index chunks to Elasticsearch using direct API calls.

        This is production-ready and gives full control.
        """
        # ES connection
        es_url = backend_config.get("es_url", "http://localhost:9200")
        index_name = backend_config.get("index_name", "legal_documents")

        es = Elasticsearch([es_url])

        # Create index if not exists (with proper mapping)
        if not es.indices.exists(index=index_name):
            mapping = self._get_index_mapping()
            es.indices.create(index=index_name, body=mapping)
            print(f"[RawES] Created index: {index_name}")

        # Prepare bulk actions
        actions = []
        for chunk in chunks:
            action = {
                "_index": index_name,
                "_id": chunk["id"],
                "_source": {
                    "content": chunk["content"],
                    "metadata": chunk["metadata"]
                }
            }
            actions.append(action)

        # Bulk index
        success, failed = helpers.bulk(es, actions, stats_only=True)
        print(f"[RawES] Indexed {success} chunks ({failed} failed)")

        return success

    def _get_index_mapping(self) -> Dict[str, Any]:
        """
        Get Elasticsearch index mapping.

        This defines the schema for hybrid search (vector + BM25).
        """
        return {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "english"
                    },
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": 384,  # For sentence-transformers/all-MiniLM-L6-v2
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "properties": {
                            "org": {"type": "keyword"},
                            "year": {"type": "integer"},
                            "doc_type": {"type": "keyword"},
                            "case_id": {"type": "keyword"},
                            "source_doc_id": {"type": "keyword"},
                            "chunk_index": {"type": "integer"}
                        }
                    }
                }
            }
        }


# ====================================================================
# EXAMPLE USAGE
# ====================================================================

def example_raw_es_ingestion():
    """
    Example showing raw ES for production.
    """
    # Sample legal documents
    documents = [
        {
            "id": "case_001_doc_1",
            "content": """
            Employment Agreement - Meta Platforms Inc.

            Employee: Mickey Mouse
            Position: Senior Research Scientist

            Non-Compete Clause: Employee agrees not to engage in competitive
            activities for 12 months following termination. This agreement is
            subject to California law, which generally prohibits non-compete
            clauses except in limited circumstances.
            """,
            "metadata": {
                "org": "Meta",
                "year": 2024,
                "doc_type": "employment_agreement",
                "case_id": "case_001"
            }
        }
    ]

    # Initialize pipeline
    pipeline = RawElasticsearchPipeline()

    # Step 1: Chunk
    chunks = pipeline.chunk_documents(documents, chunk_size=100)
    print(f"\nChunked into {len(chunks)} pieces")

    # Step 2: Extract metadata
    for chunk in chunks:
        enhanced_metadata = pipeline.extract_metadata(chunk)
        chunk["metadata"].update(enhanced_metadata)

    print(f"\nSample chunk metadata:")
    print(chunks[0]["metadata"])

    # Step 3: Index to ES
    backend_config = {
        "es_url": "http://localhost:9200",
        "index_name": "legal_documents_raw"
    }

    indexed_count = pipeline.index_to_backend(chunks, backend_config)
    print(f"\nIndexed {indexed_count} chunks")


# ====================================================================
# COMPARISON WITH LLAMAINDEX
# ====================================================================

def compare_implementations():
    """
    Show the difference between LlamaIndex and Raw ES.
    """
    print("\n" + "="*60)
    print("COMPARISON: LlamaIndex vs Raw ES")
    print("="*60)

    doc = {
        "id": "test",
        "content": "This is a test document. " * 50,
        "metadata": {"org": "Test"}
    }

    # LlamaIndex approach
    print("\n[1] LlamaIndex Approach:")
    from llama_index_ingestion import LlamaIndexPipeline
    llama_pipeline = LlamaIndexPipeline()
    llama_chunks = llama_pipeline.chunk_documents([doc])
    print(f"  Chunks created: {len(llama_chunks)}")
    print(f"  Dependencies: llama-index, sentence-transformers")
    print(f"  Use case: Rapid prototyping")

    # Raw ES approach
    print("\n[2] Raw ES Approach:")
    raw_pipeline = RawElasticsearchPipeline()
    raw_chunks = raw_pipeline.chunk_documents([doc])
    print(f"  Chunks created: {len(raw_chunks)}")
    print(f"  Dependencies: elasticsearch only")
    print(f"  Use case: Production deployment")

    print("\n✓ Both satisfy DocumentIngestionPipeline interface")
    print("✓ Agent code works with either implementation")


if __name__ == "__main__":
    # Run example
    example_raw_es_ingestion()

    # Show comparison
    compare_implementations()
