"""
LegalBench Evaluation Runner

Main evaluation loop that:
1. Loads tasks, qrels, and connects to index
2. For each task: DSPy → Search → Track metrics
3. Calculates all metrics (Recall@K, Precision@K, NDCG, latency, circuit breaker)
4. Generates evaluation report
"""

import json
import sys
import os
import time
from typing import List, Dict, Set
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import dspy

# Add parent directory to import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import get_orchestrator
from elasticsearch_hybrid_backend import ElasticsearchHybridBackend
from retrieval.reranker import CrossEncoderReranker
from production_interfaces import SearchQuery
from core.data_structures import create_initial_state
from legalbench.metrics import (
    MetricsTracker,
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_ndcg_at_k
)


class LegalBenchEvaluator:
    """Main evaluation orchestrator"""

    def __init__(self, config: Dict):
        """
        Initialize evaluator

        Args:
            config: Configuration dict with:
                - es_url: Elasticsearch URL
                - es_index: Index name
                - tasks_path: Path to tasks.jsonl
                - qrels_path: Path to qrels.tsv
                - embedding_model: Embedding model name
        """
        self.config = config

        # Initialize DSPy orchestrator (the "Brain")
        # Initialize backend first to get schema
        print(f"[Eval] Connecting to Elasticsearch at {config.get('es_url')}...")
        self.backend = ElasticsearchHybridBackend(
            es_url=config.get("es_url", "http://localhost:9200"),
            index_name=config.get("es_index", "legal_documents"),
            embedding_model=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )

        # Fetch schema for Agentic Skill (Tool Adaptation)
        print("[Eval] Fetching index schema for Agentic Skill...")
        schema_context = self.backend.get_index_schema()
        print(f"[Eval] Agent adaptation: Found doc_types {schema_context.get('doc_type')}")

        # Initialize DSPy orchestrator with schema context
        print("[Eval] Initializing DSPy orchestrator with Tool Context...")
        self.orchestrator = get_orchestrator(schema_context=schema_context)

        # Initialize Reranker (Tier 2)
        # Using a small but effective model for speed
        self.reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
      
        # Load tasks and qrels
        print(f"[Eval] Loading tasks from {config['tasks_path']}...")
        self.tasks = self._load_jsonl(config["tasks_path"])
        print(f"[Eval] Loaded {len(self.tasks)} tasks")

        print(f"[Eval] Loading qrels from {config['qrels_path']}...")
        self.qrels = self._load_qrels(config["qrels_path"])
        print(f"[Eval] Loaded {len(self.qrels)} relevance judgments")

        # Build task_id -> relevant_docs mapping
        self.task_relevant_docs = self._build_relevant_docs_map()

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()

    def run_evaluation(
        self,
        max_queries: int = None,
        top_k: int = 20,
        filter_category: str = None
    ) -> Dict:
        """
        Run full evaluation loop

        Args:
            max_queries: Limit number of queries (for testing)
            top_k: Number of documents to retrieve
            filter_category: Optional filter by task category

        Returns:
            Evaluation results with all metrics
        """
        print("\n" + "="*70)
        print("STARTING LEGALBENCH EVALUATION")
        print("="*70)
        print(f"Total tasks: {len(self.tasks)}")
        print(f"Top-K: {top_k}")
        if filter_category:
            print(f"Category filter: {filter_category}")
        if max_queries:
            print(f"Max queries: {max_queries}")
        print("="*70 + "\n")

        # Filter tasks
        tasks_to_evaluate = self.tasks
        if filter_category:
            tasks_to_evaluate = [t for t in self.tasks if t.get("category") == filter_category]
            print(f"[Eval] Filtered to {len(tasks_to_evaluate)} tasks in category '{filter_category}'")

        # Filter tasks to only those with relevance judgments (qrels)
        # This is CRITICAL to avoid 0.000 relevance metrics for tasks without ground truth
        tasks_with_qrels = [t for t in tasks_to_evaluate if t["task_id"] in self.task_relevant_docs]
        print(f"[Eval] Filtering to {len(tasks_with_qrels)} tasks that have qrels (out of {len(tasks_to_evaluate)} candidates)")
        tasks_to_evaluate = tasks_with_qrels

        if max_queries:
            tasks_to_evaluate = tasks_to_evaluate[:max_queries]

        # Start evaluation timer
        self.metrics_tracker.start_evaluation()

        # Evaluate each task
        for task in tqdm(tasks_to_evaluate, desc="Evaluating tasks"):
            result = self._evaluate_single_task(task, top_k)
            self.metrics_tracker.track_query_result(result)

        # End evaluation timer
        self.metrics_tracker.end_evaluation()

        # Get summary metrics
        summary = self.metrics_tracker.get_summary(tasks=self.tasks)

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        self._print_summary(summary)

        return {
            "config": self.config,
            "summary": summary,
            "per_query_results": self.metrics_tracker.query_results
        }

    def _evaluate_single_task(self, task: Dict, top_k: int) -> Dict:
        """
        Evaluate a single task

        Returns:
            Result dict with metrics for this task
        """
        task_id = task["task_id"]
        query = task["query"]

        # Start timer
        start_time = time.time()

        if self.config.get("baseline"):
            # === BASELINE MODE (No Agent) ===
            search_query = SearchQuery(
                text=query,
                vector_query=query,
                keyword_query=query,
                filters={},
                negative_constraints=[],
                top_k=top_k
            )
            # Execute baseline search once
            search_query.top_k = 50
            retrieved_docs = self.backend.hybrid_search(search_query)
            reranked_docs = self.reranker.rank(query, retrieved_docs, top_k=top_k)
            retrieved_docs = reranked_docs

        else:
            # === AGENT MODE (Reflective Scholar) ===
            state = create_initial_state(query)
            max_retries = 2
            attempt = 0
            retry_history = []
            
            # Loop for Reflexion
            while attempt <= max_retries:
                # Step 1: Plan
                # Update history so Planner sees previous failures
                # We hack 'history' (usually Q/A) to include reflection context
                if retry_history:
                     state["history"] = [{"query": "Previous Attempt Reflection", "answer": h} for h in retry_history]

                search_plan = self.orchestrator.forward(state)

                # Step 2: Search (Broad)
                print(f"\n[Trace {task_id}] Attempt {attempt+1} Query: {state['query']}")
                search_query = SearchQuery(
                    text=state["query"],
                    vector_query=search_plan.hyde_passage,
                    keyword_query=search_plan.search_queries[0] if search_plan.search_queries else state["query"],
                    filters=search_plan.filter_constraints,
                    negative_constraints=search_plan.negative_constraints,
                    top_k=50 # Broad retrieval
                )
                retrieved_docs_broad = self.backend.hybrid_search(search_query)
                
                # Step 3: Rerank
                reranked_docs = self.reranker.rank(state["query"], retrieved_docs_broad, top_k=top_k)
                
                # Step 4: Critique (Self-RAG)
                # Convert docs to string content
                top_docs_content = [d.content for d in reranked_docs[:5]]
                critic_res = self.orchestrator.evaluate_retrieval(query, top_docs_content) # Use ORIGINAL query for critique
                
                score = float(getattr(critic_res, 'relevance_score', 0.0))
                print(f"[Trace {task_id}] Critic Score: {score}")

                # Check Success
                if score >= 0.7:
                    print(f"[Trace {task_id}] Success! Stopping early.")
                    retrieved_docs = reranked_docs
                    break
                
                # Step 5: Reflect (Reflexion)
                if attempt < max_retries:
                     print(f"[Trace {task_id}] retrieving is insufficient. Reflecting...")
                     reflection_res = self.orchestrator.reflect_on_failure(
                         query=query,
                         docs=top_docs_content,
                         critique=getattr(critic_res, 'critique', 'No critique'),
                         history=retry_history
                     )
                     
                     new_query = getattr(reflection_res, 'improved_query', state["query"])
                     reflection_text = getattr(reflection_res, 'reflection', 'No reflection')
                     
                     print(f"[Trace {task_id}] New Query: {new_query}")
                     
                     # Update State for next loop
                     state["query"] = new_query
                     retry_history.append(f"Attempt {attempt}: Critique={critic_res.critique}. Reflection={reflection_text}")
                     attempt += 1
                else:
                    print(f"[Trace {task_id}] Max retries reached.")
                    retrieved_docs = reranked_docs
                    break
            
            # Final result is from the last loop iteration
            retrieved_docs = reranked_docs

        # End timer
        latency_ms = (time.time() - start_time) * 1000

        # Step 4: Extract retrieved doc IDs
        # KEY FIX: Map chunk UUIDs back to source Document IDs (e.g. doc_0468) for Qrels matching
        retrieved_ids = []
        for doc in retrieved_docs:
            mapped_id = doc.metadata.get("source_doc_id") or doc.metadata.get("doc_id") or doc.id
            if isinstance(mapped_id, list):
                mapped_id = mapped_id[0]
            retrieved_ids.append(str(mapped_id))

        # Step 5: Get relevant docs for this task
        relevant_ids = self.task_relevant_docs.get(task_id, set())

        # Step 6: Calculate metrics
        metrics = self._calculate_metrics(retrieved_ids, relevant_ids, top_k)

        # Step 7: Build result
        result = {
            "task_id": task_id,
            "query": query,
            "category": task.get("category", "unknown"),
            "domain": task.get("domain", "unknown"),
            "latency_ms": latency_ms,
            "num_retrieved": len(retrieved_docs),
            "num_relevant": len(relevant_ids),
            **metrics,
            "retrieved_ids": retrieved_ids[:10],  # Store top 10 for analysis
        }

        return result

    def _calculate_metrics(self, retrieved_ids: List[str], relevant_ids: Set[str], top_k: int) -> Dict:
        """Calculate Recall@K, Precision@K, NDCG@K"""
        metrics = {}

        # Get graded relevance for NDCG
        graded_relevance = {}
        for qrel in self.qrels:
            if qrel["doc_id"] in relevant_ids:
                graded_relevance[qrel["doc_id"]] = qrel["relevance_grade"]

        # Calculate metrics at K=[5, 10, 20]
        for k in [5, 10, 20]:
            if k > top_k:
                continue

            metrics[f"recall@{k}"] = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"precision@{k}"] = calculate_precision_at_k(retrieved_ids, relevant_ids, k)

        # NDCG@10
        metrics["ndcg@10"] = calculate_ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k=min(10, top_k),
            graded_relevance=graded_relevance
        )

        return metrics

    def _build_relevant_docs_map(self) -> Dict[str, Set[str]]:
        """
        Build mapping from task_id to set of relevant doc_ids

        Uses qrels with relevance_grade >= 2 (relevant or highly relevant)
        """
        relevant_docs = defaultdict(set)

        for qrel in self.qrels:
            if qrel["relevance_grade"] >= 2:  # 2=relevant, 3=highly relevant
                relevant_docs[qrel["task_id"]].add(qrel["doc_id"])

        return dict(relevant_docs)

    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file"""
        items = []
        with open(filepath, 'r') as f:
            for line in f:
                items.append(json.loads(line))
        return items

    def _load_qrels(self, filepath: str) -> List[Dict]:
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

    def _print_summary(self, summary: Dict):
        """Print evaluation summary to console"""
        print(f"\nTotal Queries: {summary['total_queries']}")

        # Retrieval metrics
        print("\n--- RETRIEVAL METRICS ---")
        retrieval = summary.get("retrieval_metrics", {})
        for k in [5, 10, 20]:
            recall = retrieval.get(f"mean_recall@{k}", 0)
            precision = retrieval.get(f"mean_precision@{k}", 0)
            print(f"Recall@{k}: {recall:.3f}  |  Precision@{k}: {precision:.3f}")

        ndcg = retrieval.get("mean_ndcg@10", 0)
        print(f"NDCG@10: {ndcg:.3f}")

        # Latency metrics
        print("\n--- LATENCY METRICS ---")
        latency = summary.get("latency_metrics", {})
        print(f"Mean: {latency.get('mean_ms', 0):.1f} ms")
        print(f"Median: {latency.get('median_ms', 0):.1f} ms")
        print(f"P95: {latency.get('p95_ms', 0):.1f} ms")
        print(f"P99: {latency.get('p99_ms', 0):.1f} ms")

        # Throughput
        print(f"\nThroughput: {summary.get('throughput_qps', 0):.2f} QPS")

        # Circuit breaker
        cb = summary.get("circuit_breaker_metrics", {})
        if cb.get("early_stop_rate", 0) > 0:
            print("\n--- CIRCUIT BREAKER ---")
            print(f"Early stop rate: {cb['early_stop_rate']:.1%}")
            print(f"Mean novelty: {cb.get('mean_novelty', 0):.3f}")

        # By category
        if "by_category" in summary:
            print("\n--- BY CATEGORY ---")
            for cat, metrics in summary["by_category"].items():
                print(f"{cat:30s} | Queries: {metrics['num_queries']:3d} | Recall@10: {metrics['mean_recall@10']:.3f}")

        # By domain
        if "by_domain" in summary:
            print("\n--- BY DOMAIN ---")
            for domain, metrics in summary["by_domain"].items():
                print(f"{domain:30s} | Queries: {metrics['num_queries']:3d} | Recall@10: {metrics['mean_recall@10']:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LegalBench evaluation")
    parser.add_argument("--tasks", default="legalbench/data/tasks.jsonl", help="Path to tasks.jsonl")
    parser.add_argument("--qrels", default="legalbench/data/qrels.tsv", help="Path to qrels.tsv")
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--es-index", default="legalbench_documents", help="Index name")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model")
    parser.add_argument("--max-queries", type=int, help="Limit number of queries")
    parser.add_argument("--top-k", type=int, default=20, help="Number of docs to retrieve")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--output", default="legalbench/results", help="Output directory")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no agent) evaluation")

    args = parser.parse_args()

    # Create config
    config = {
        "es_url": args.es_url,
        "es_index": args.es_index,
        "tasks_path": args.tasks,
        "qrels_path": args.qrels,
        "embedding_model": args.embedding_model,
        "baseline": args.baseline
    }

    # Load API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY not found. DSPy might fail.")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
    else:
        # User specified Opus 4.5 - trying standard naming pattern
        model_name = "claude-opus-4-20250514"
        print(f"[Eval] Configuring DSPy with model: {model_name}")
        lm = dspy.LM(model=f"anthropic/{model_name}", api_key=api_key)
        dspy.configure(lm=lm)

    # Run evaluation
    evaluator = LegalBenchEvaluator(config)
    results = evaluator.run_evaluation(
        max_queries=args.max_queries,
        top_k=args.top_k,
        filter_category=args.category
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
