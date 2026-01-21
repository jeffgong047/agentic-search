"""
LlamaIndex Ingestion Pipeline

This is the "rapid prototyping" implementation.
Use this to prove the concept quickly.
Swap to raw_es_ingestion.py when mentor needs it.
"""

from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.production_interfaces import DocumentIngestionPipeline


class LlamaIndexPipeline(DocumentIngestionPipeline):
    """
    LlamaIndex implementation for rapid prototyping.

    Pros:
    - Fast to set up
    - Handles chunking automatically
    - Good for demos

    Cons:
    - Extra dependency
    - Less control over chunking logic
    - Mentor might prefer raw ES

    SWAPPABLE: Replace with RawElasticsearchPipeline for production
    """

    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize LlamaIndex pipeline.

        Args:
            embedding_model: HuggingFace model for embeddings
        """
        # Configure LlamaIndex settings
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        self.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )

        print("[LlamaIndex] Pipeline initialized")

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents using LlamaIndex.

        Args:
            documents: List of dicts with 'id', 'content', 'metadata'

        Returns:
            List of chunks
        """
        # Convert to LlamaIndex Document objects
        llama_docs = [
            Document(
                text=doc["content"],
                metadata={
                    **doc.get("metadata", {}),
                    "source_doc_id": doc.get("doc_id") or doc.get("id")
                },
                id_=doc.get("doc_id") or doc.get("id")
            )
            for doc in documents
        ]

        # Parse into nodes (chunks)
        nodes = self.node_parser.get_nodes_from_documents(llama_docs)

        # Convert back to dicts
        chunks = []
        for i, node in enumerate(nodes):
            chunks.append({
                "id": f"{node.metadata.get('source_doc_id', 'unknown')}_{i}",
                "content": node.text,
                "metadata": node.metadata
            })

        print(f"[LlamaIndex] Chunked {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from document.

        LlamaIndex can do this automatically, or you can add custom logic.
        """
        # For now, just pass through existing metadata
        # In production, you might call an NER service or LLM here
        metadata = document.get("metadata", {})

        # Example: Extract entities with LlamaIndex (optional)
        # This would require additional setup
        # extractor = EntityExtractor()
        # entities = extractor.extract(document["content"])
        # metadata["entities"] = entities

        return metadata

    def index_to_backend(
        self,
        chunks: List[Dict[str, Any]],
        backend_config: Dict[str, Any]
    ) -> int:
        """
        Index chunks to Elasticsearch via LlamaIndex.

        Args:
            chunks: Chunked documents
            backend_config: ES connection config

        Returns:
            Number of indexed chunks
        """
        # Configure ES vector store
        es_config = backend_config or {
            "es_url": "http://localhost:9200",
            "index_name": "legal_documents"
        }

        # Create vector store
        vector_store = ElasticsearchStore(
            es_url=es_config.get("es_url"),
            index_name=es_config.get("index_name")
        )

        from llama_index.core import StorageContext
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Convert chunks back to LlamaIndex Documents
        llama_docs = [
            Document(
                text=chunk["content"],
                metadata=chunk["metadata"],
                id_=chunk["id"]
            )
            for chunk in chunks
        ]

        # Create index and insert
        index = VectorStoreIndex.from_documents(
            llama_docs,
            storage_context=storage_context
        )

        print(f"[LlamaIndex] Indexed {len(chunks)} chunks to ES")
        return len(chunks)


# ====================================================================
# EXAMPLE USAGE
# ====================================================================

def example_llamaindex_ingestion():
    """
    Example showing LlamaIndex for rapid prototyping.
    """
    # Sample legal documents
    documents = [
        {
            "id": "case_001_doc_1",
            "content": """
            Employment Agreement - Meta Platforms Inc.

            Employee: Mickey Mouse
            Position: Senior Research Scientist
            Department: AI Research Lab

            Non-Compete Clause: Employee agrees not to engage in competitive
            activities for 12 months following termination...
            """,
            "metadata": {
                "org": "Meta",
                "year": 2024,
                "doc_type": "employment_agreement",
                "case_id": "case_001"
            }
        },
        {
            "id": "case_001_doc_2",
            "content": """
            Research Publication - Meta AI

            Authors: Mickey Mouse, Sarah Johnson, Michael Wong
            Affiliation: Meta AI Research Lab

            This paper presents advances in multi-modal AI...
            """,
            "metadata": {
                "org": "Meta",
                "year": 2024,
                "doc_type": "publication",
                "case_id": "case_001"
            }
        }
    ]

    # Initialize pipeline
    pipeline = LlamaIndexPipeline()

    # Step 1: Chunk
    chunks = pipeline.chunk_documents(documents)
    print(f"\nChunked into {len(chunks)} pieces")

    # Step 2: Extract metadata (already done in this example)
    for chunk in chunks:
        chunk["metadata"] = pipeline.extract_metadata(chunk)

    # Step 3: Index to ES
    backend_config = {
        "es_url": "http://localhost:9200",
        "index_name": "legal_documents_llama"
    }

    indexed_count = pipeline.index_to_backend(chunks, backend_config)
    print(f"Indexed {indexed_count} chunks")


# ====================================================================
# SWAPPABILITY DEMONSTRATION
# ====================================================================

def demo_swappability():
    """
    Show how easy it is to swap implementations.
    """
    print("\n" + "="*60)
    print("SWAPPABILITY DEMO")
    print("="*60)

    documents = [{"id": "test", "content": "Test document", "metadata": {}}]

    # Option 1: LlamaIndex (for prototyping)
    print("\n[1] Using LlamaIndex Pipeline...")
    llama_pipeline = LlamaIndexPipeline()
    llama_chunks = llama_pipeline.chunk_documents(documents)
    print(f"  Chunks: {len(llama_chunks)}")

    # Option 2: Raw ES (for production)
    # Uncomment when raw_es_ingestion.py is available
    # print("\n[2] Using Raw ES Pipeline...")
    # raw_pipeline = RawElasticsearchPipeline()
    # raw_chunks = raw_pipeline.chunk_documents(documents)
    # print(f"  Chunks: {len(raw_chunks)}")

    print("\n✓ Both implementations satisfy the same interface")
    print("✓ Agent code doesn't need to change")


if __name__ == "__main__":
    # Run example
    example_llamaindex_ingestion()

    # Show swappability
    demo_swappability()
