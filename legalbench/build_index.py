"""
Semantic Index Builder

Builds Elasticsearch index using LlamaIndex with free embedding models.
Reuses existing llama_index_ingestion.py pipeline.
"""

import json
import sys
import os
from typing import List, Dict
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index_ingestion import LlamaIndexPipeline
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


class LegalBenchIndexBuilder:
    """Build semantic index for LegalBench evaluation"""

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        embedding_model: str = "BAAI/bge-small-en-v1.5"
    ):
        """
        Initialize index builder

        Args:
            es_url: Elasticsearch URL
            embedding_model: Free HuggingFace embedding model
        """
        self.es_url = es_url
        self.embedding_model_name = embedding_model
        self.es = Elasticsearch(es_url)
        self.pipeline = None  # Lazy load

    def build_index(
        self,
        corpus_path: str,
        index_name: str = "legalbench_documents",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        force_rebuild: bool = False
    ) -> int:
        """
        Build complete index from corpus

        Args:
            corpus_path: Path to corpus.jsonl
            index_name: Elasticsearch index name
            chunk_size: Chunk size in characters
            chunk_overlap: Overlap between chunks
            force_rebuild: If True, delete existing index

        Returns:
            Number of chunks indexed
        """
        print(f"[Index] Building index '{index_name}' from {corpus_path}")
        print(f"[Index] Embedding model: {self.embedding_model_name}")
        print(f"[Index] Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

        # Check if index exists
        if self.es.indices.exists(index=index_name):
            if force_rebuild:
                print(f"[Index] Deleting existing index '{index_name}'...")
                self.es.indices.delete(index=index_name)
            else:
                print(f"[Index] Index '{index_name}' already exists. Use force_rebuild=True to recreate.")
                return 0

        # Load corpus
        print("[Index] Loading corpus...")
        corpus = self._load_corpus(corpus_path)
        print(f"[Index] Loaded {len(corpus)} documents")

        # Initialize LlamaIndex pipeline
        print("[Index] Initializing LlamaIndex pipeline...")
        self.pipeline = LlamaIndexPipeline(embedding_model=self.embedding_model_name)

        # Chunk documents
        print("[Index] Chunking documents...")
        chunks = self.pipeline.chunk_documents(
            corpus,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"[Index] Created {len(chunks)} chunks from {len(corpus)} documents")

        # Index to Elasticsearch
        print(f"[Index] Indexing to Elasticsearch at {self.es_url}...")
        indexed_count = self.pipeline.index_to_backend(
            chunks,
            backend_config={
                "es_url": self.es_url,
                "index_name": index_name
            }
        )

        print(f"✓ Indexed {indexed_count} chunks to '{index_name}'")
        return indexed_count

    def verify_index(self, index_name: str = "legalbench_documents") -> Dict:
        """
        Verify index was created correctly

        Returns:
            Index statistics
        """
        print(f"[Verify] Checking index '{index_name}'...")

        if not self.es.indices.exists(index=index_name):
            print(f"✗ Index '{index_name}' does not exist")
            print(f"DEBUG: Existing indices: {self.es.indices.get_alias(index='*').keys()}")
            return {}

        self.es.indices.refresh(index=index_name)

        # Get index stats
        stats = self.es.indices.stats(index=index_name)
        doc_count = stats['indices'][index_name]['total']['docs']['count']
        size_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']

        print(f"✓ Index '{index_name}' contains:")
        print(f"  Documents: {doc_count}")
        print(f"  Size: {size_bytes / 1024 / 1024:.2f} MB")

        # Test search
        test_query = "employment contract non-compete"
        print(f"\n[Verify] Testing search: '{test_query}'")

        try:
            response = self.es.search(
                index=index_name,
                body={
                    "query": {"match": {"content": test_query}},
                    "size": 3
                }
            )

            hits = response['hits']['hits']
            print(f"✓ Test search returned {len(hits)} results")

            if hits:
                print("\nTop result:")
                print(f"  Doc ID: {hits[0]['_id']}")
                print(f"  Score: {hits[0]['_score']}")
                print(f"  Content: {hits[0]['_source']['content'][:200]}...")

        except Exception as e:
            print(f"✗ Test search failed: {e}")

        return {
            "index_name": index_name,
            "doc_count": doc_count,
            "size_bytes": size_bytes,
            "exists": True
        }

    def get_embedding_model_info(self) -> Dict:
        """Get information about the embedding model"""
        try:
            model = SentenceTransformer(self.embedding_model_name)
            dim = model.get_sentence_embedding_dimension()

            return {
                "model_name": self.embedding_model_name,
                "dimensions": dim,
                "max_seq_length": model.max_seq_length
            }
        except Exception as e:
            return {
                "model_name": self.embedding_model_name,
                "error": str(e)
            }

    def _load_corpus(self, corpus_path: str) -> List[Dict]:
        """Load corpus from JSONL file"""
        corpus = []
        with open(corpus_path, 'r') as f:
            for line in f:
                corpus.append(json.loads(line))
        return corpus


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LegalBench semantic index")
    parser.add_argument("--corpus", default="legalbench/data/corpus.jsonl", help="Path to corpus.jsonl")
    parser.add_argument("--index", default="legalbench_documents", help="Index name")
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild index")

    args = parser.parse_args()

    # Build index
    builder = LegalBenchIndexBuilder(
        es_url=args.es_url,
        embedding_model=args.embedding_model
    )

    # Get embedding model info
    model_info = builder.get_embedding_model_info()
    print("\n" + "="*60)
    print("EMBEDDING MODEL INFO")
    print("="*60)
    print(f"Model: {model_info.get('model_name')}")
    print(f"Dimensions: {model_info.get('dimensions', 'unknown')}")
    print(f"Max sequence length: {model_info.get('max_seq_length', 'unknown')}")
    print("="*60 + "\n")

    # Build index
    indexed_count = builder.build_index(
        corpus_path=args.corpus,
        index_name=args.index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_rebuild=args.force_rebuild
    )

    # Verify
    if indexed_count > 0:
        builder.verify_index(args.index)
