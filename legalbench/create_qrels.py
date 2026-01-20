"""
Qrels (Relevance Judgments) Generator

Creates query-document relevance judgments for LegalBench evaluation.

Methods:
1. Pseudo-relevance: Use BM25 + similarity to create automatic judgments (fast)
2. LLM-as-judge: Use GPT to judge relevance (better quality)
3. Manual: Load from manual annotations (gold standard)
"""

import json
import os
from typing import List, Dict, Set, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class QrelsGenerator:
    """Generate query-document relevance judgments"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize qrels generator

        Args:
            embedding_model: Model for semantic similarity
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None  # Lazy load

    def generate_qrels(
        self,
        tasks_path: str,
        corpus_path: str,
        output_path: str,
        method: str = "pseudo",
        top_k: int = 50,
        relevance_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Generate relevance judgments

        Args:
            tasks_path: Path to tasks.jsonl
            corpus_path: Path to corpus.jsonl
            output_path: Path to save qrels.tsv
            method: "pseudo" | "llm" | "manual"
            top_k: Number of docs to judge per query
            relevance_threshold: Threshold for binary relevance (pseudo method)

        Returns:
            List of qrel dictionaries
        """
        print(f"[Qrels] Method: {method}, Top-K: {top_k}")

        # Load tasks and corpus
        tasks = self._load_jsonl(tasks_path)
        corpus = self._load_jsonl(corpus_path)

        print(f"[Qrels] Loaded {len(tasks)} tasks, {len(corpus)} documents")

        if method == "pseudo":
            qrels = self._generate_pseudo_qrels(tasks, corpus, top_k, relevance_threshold)
        elif method == "llm":
            qrels = self._generate_llm_qrels(tasks, corpus, top_k)
        elif method == "manual":
            qrels = self._load_manual_qrels(output_path)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Save to TSV
        self.save_qrels_tsv(qrels, output_path)

        print(f"✓ Saved {len(qrels)} qrels to {output_path}")
        return qrels

    def _generate_pseudo_qrels(
        self,
        tasks: List[Dict],
        corpus: List[Dict],
        top_k: int,
        relevance_threshold: float
    ) -> List[Dict]:
        """
        Generate pseudo-relevance judgments using BM25 + embedding similarity

        Strategy:
        1. For each query, retrieve top-K docs using BM25
        2. Re-rank using semantic similarity
        3. Label docs above threshold as relevant (grade 2-3)
        4. Label docs below threshold as marginally relevant (grade 1) or not relevant (grade 0)
        """
        print("[Pseudo] Generating qrels using BM25 + embedding similarity...")

        # Prepare corpus for BM25
        tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        # Load embedding model
        if self.embedding_model is None:
            print(f"[Pseudo] Loading embedding model: {self.embedding_model_name}...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Encode all documents
        print("[Pseudo] Encoding documents...")
        doc_embeddings = self.embedding_model.encode(
            [doc["content"] for doc in corpus],
            show_progress_bar=True,
            convert_to_numpy=True
        )

        qrels = []

        for task in tqdm(tasks, desc="Generating qrels"):
            task_id = task["task_id"]
            query = task["query"]

            # Step 1: BM25 retrieval
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)

            # Get top-K docs by BM25
            top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]

            # Step 2: Semantic re-ranking
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

            # Calculate cosine similarity for top-K docs
            similarities = []
            for idx in top_k_indices:
                doc_emb = doc_embeddings[idx]
                sim = np.dot(query_embedding, doc_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                )
                similarities.append((idx, sim))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Step 3: Assign relevance grades
            for rank, (doc_idx, sim) in enumerate(similarities):
                doc_id = corpus[doc_idx]["doc_id"]

                # Determine relevance grade based on similarity score
                if sim >= relevance_threshold + 0.2:
                    # Highly relevant
                    grade = 3
                elif sim >= relevance_threshold:
                    # Relevant
                    grade = 2
                elif sim >= relevance_threshold - 0.1:
                    # Marginally relevant
                    grade = 1
                else:
                    # Not relevant (skip to save space)
                    continue

                qrels.append({
                    "task_id": task_id,
                    "doc_id": doc_id,
                    "relevance_grade": grade,
                    "judgment_source": "pseudo",
                    "similarity_score": float(sim),
                    "rank": rank + 1
                })

        return qrels

    def _generate_llm_qrels(
        self,
        tasks: List[Dict],
        corpus: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Generate qrels using LLM-as-judge (GPT-4)

        For each query-document pair:
        - Prompt GPT to judge relevance on 0-3 scale
        - 0 = Not relevant
        - 1 = Marginally relevant
        - 2 = Relevant
        - 3 = Highly relevant
        """
        print("[LLM] Generating qrels using LLM-as-judge...")

        try:
            import openai
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")

            client = openai.OpenAI(api_key=api_key)

        except Exception as e:
            print(f"✗ LLM-as-judge failed: {e}")
            print("[Fallback] Using pseudo-relevance instead...")
            return self._generate_pseudo_qrels(tasks, corpus, top_k, 0.5)

        qrels = []

        # First, use BM25 to get candidate docs
        tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        for task in tqdm(tasks[:10], desc="LLM judging (limited to 10 tasks for cost)"):  # Limit for cost
            task_id = task["task_id"]
            query = task["query"]

            # Get top-K candidates with BM25
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]

            for idx in top_k_indices[:5]:  # Judge only top 5 per query
                doc_id = corpus[idx]["doc_id"]
                doc_content = corpus[idx]["content"][:1000]  # First 1000 chars

                # Prompt LLM
                prompt = f"""You are a legal expert evaluating document relevance.

Query: {query}

Document: {doc_content}

Rate the relevance of this document to the query on a scale of 0-3:
- 0: Not relevant (document does not answer the query)
- 1: Marginally relevant (document touches on the topic but doesn't answer)
- 2: Relevant (document provides useful information to answer the query)
- 3: Highly relevant (document directly answers the query)

Respond with ONLY the number (0, 1, 2, or 3)."""

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # Use mini for cost
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=10
                    )

                    grade_str = response.choices[0].message.content.strip()
                    grade = int(grade_str)

                    if grade > 0:  # Only store relevant docs
                        qrels.append({
                            "task_id": task_id,
                            "doc_id": doc_id,
                            "relevance_grade": grade,
                            "judgment_source": "llm_gpt4o-mini"
                        })

                except Exception as e:
                    print(f"✗ LLM judgment failed for {task_id}-{doc_id}: {e}")
                    continue

        return qrels

    def _load_manual_qrels(self, filepath: str) -> List[Dict]:
        """Load manually annotated qrels from TSV file"""
        qrels = []
        with open(filepath, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                task_id, doc_id, grade = line.strip().split('\t')
                qrels.append({
                    "task_id": task_id,
                    "doc_id": doc_id,
                    "relevance_grade": int(grade),
                    "judgment_source": "manual"
                })
        return qrels

    def save_qrels_tsv(self, qrels: List[Dict], filepath: str):
        """Save qrels in TSV format (TREC format)"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            # Write header
            f.write("task_id\tdoc_id\trelevance_grade\tjudgment_source\n")

            # Write qrels
            for qrel in qrels:
                f.write(f"{qrel['task_id']}\t{qrel['doc_id']}\t{qrel['relevance_grade']}\t{qrel.get('judgment_source', 'unknown')}\n")

    def load_qrels_tsv(self, filepath: str) -> List[Dict]:
        """Load qrels from TSV file"""
        qrels = []
        with open(filepath, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    qrels.append({
                        "task_id": parts[0],
                        "doc_id": parts[1],
                        "relevance_grade": int(parts[2]),
                        "judgment_source": parts[3] if len(parts) > 3 else "unknown"
                    })
        return qrels

    def get_qrels_statistics(self, qrels: List[Dict]) -> Dict:
        """Get statistics about qrels"""
        stats = {
            "total_judgments": len(qrels),
            "total_tasks": len(set(qrel["task_id"] for qrel in qrels)),
            "total_docs": len(set(qrel["doc_id"] for qrel in qrels)),
            "by_grade": {0: 0, 1: 0, 2: 0, 3: 0},
            "by_source": {},
            "avg_relevant_per_task": 0
        }

        for qrel in qrels:
            grade = qrel["relevance_grade"]
            source = qrel.get("judgment_source", "unknown")

            stats["by_grade"][grade] = stats["by_grade"].get(grade, 0) + 1
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

        # Calculate average relevant docs per task
        task_relevant_counts = defaultdict(int)
        for qrel in qrels:
            if qrel["relevance_grade"] >= 2:  # Grade 2 or 3 = relevant
                task_relevant_counts[qrel["task_id"]] += 1

        if task_relevant_counts:
            stats["avg_relevant_per_task"] = sum(task_relevant_counts.values()) / len(task_relevant_counts)

        return stats

    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file"""
        items = []
        with open(filepath, 'r') as f:
            for line in f:
                items.append(json.loads(line))
        return items


if __name__ == "__main__":
    # Test script
    generator = QrelsGenerator()

    # Generate pseudo qrels
    qrels = generator.generate_qrels(
        tasks_path="legalbench/data/tasks.jsonl",
        corpus_path="legalbench/data/corpus.jsonl",
        output_path="legalbench/data/qrels.tsv",
        method="pseudo",
        top_k=50,
        relevance_threshold=0.5
    )

    # Print statistics
    stats = generator.get_qrels_statistics(qrels)
    print("\n" + "="*60)
    print("QRELS STATISTICS")
    print("="*60)
    print(f"Total judgments: {stats['total_judgments']}")
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Total docs judged: {stats['total_docs']}")
    print(f"Avg relevant docs per task: {stats['avg_relevant_per_task']:.1f}")
    print(f"\nBy Relevance Grade:")
    for grade, count in sorted(stats["by_grade"].items()):
        print(f"  Grade {grade}: {count}")
    print(f"\nBy Judgment Source:")
    for source, count in stats["by_source"].items():
        print(f"  {source}: {count}")
    print("="*60)
