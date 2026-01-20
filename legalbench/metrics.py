"""
Metrics Tracker

Tracks retrieval, latency, and circuit breaker metrics for evaluation.
"""

import numpy as np
from typing import List, Dict, Set
from collections import defaultdict
import time


class MetricsTracker:
    """Track retrieval, latency, and circuit breaker metrics"""

    def __init__(self):
        """Initialize metrics tracker"""
        self.latencies = []
        self.novelty_scores = []
        self.early_stops = 0
        self.total_queries = 0

        # Per-query results
        self.query_results = []

        # Start time for throughput calculation
        self.start_time = None
        self.end_time = None

    def start_evaluation(self):
        """Mark evaluation start time"""
        self.start_time = time.time()

    def end_evaluation(self):
        """Mark evaluation end time"""
        self.end_time = time.time()

    def track_query_result(self, result: Dict):
        """
        Track a single query result

        Args:
            result: Dict with keys:
                - task_id
                - latency_ms
                - recall@5, recall@10, recall@20
                - precision@5, precision@10, precision@20
                - ndcg@10
                - num_relevant
                - num_retrieved
                - circuit_breaker_stopped (optional)
                - novelty_score (optional)
        """
        self.query_results.append(result)
        self.latencies.append(result.get("latency_ms", 0))
        self.total_queries += 1

        if "novelty_score" in result:
            self.novelty_scores.append(result["novelty_score"])

        if result.get("circuit_breaker_stopped", False):
            self.early_stops += 1

    def get_latency_stats(self) -> Dict:
        """Calculate latency statistics"""
        if not self.latencies:
            return {}

        return {
            "mean_ms": float(np.mean(self.latencies)),
            "median_ms": float(np.median(self.latencies)),
            "std_ms": float(np.std(self.latencies)),
            "min_ms": float(np.min(self.latencies)),
            "max_ms": float(np.max(self.latencies)),
            "p50_ms": float(np.percentile(self.latencies, 50)),
            "p95_ms": float(np.percentile(self.latencies, 95)),
            "p99_ms": float(np.percentile(self.latencies, 99))
        }

    def get_throughput(self) -> float:
        """
        Calculate queries per second

        Returns:
            QPS (queries per second)
        """
        if self.start_time is None or self.end_time is None:
            return 0.0

        total_time_sec = self.end_time - self.start_time
        if total_time_sec <= 0:
            return 0.0

        return self.total_queries / total_time_sec

    def get_circuit_breaker_stats(self) -> Dict:
        """Calculate circuit breaker efficiency metrics"""
        if not self.novelty_scores:
            return {
                "early_stop_rate": 0.0,
                "mean_novelty": 0.0,
                "median_novelty": 0.0
            }

        return {
            "early_stop_rate": self.early_stops / self.total_queries if self.total_queries > 0 else 0.0,
            "mean_novelty": float(np.mean(self.novelty_scores)),
            "median_novelty": float(np.median(self.novelty_scores)),
            "std_novelty": float(np.std(self.novelty_scores)),
            "total_early_stops": self.early_stops,
            "total_full_runs": self.total_queries - self.early_stops
        }

    def get_retrieval_metrics(self) -> Dict:
        """
        Calculate aggregate retrieval metrics

        Returns:
            Dict with mean/median for Recall@K, Precision@K, NDCG@K
        """
        if not self.query_results:
            return {}

        metrics = {
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "precision@5": [],
            "precision@10": [],
            "precision@20": [],
            "ndcg@10": []
        }

        for result in self.query_results:
            for metric_name in metrics.keys():
                if metric_name in result:
                    metrics[metric_name].append(result[metric_name])

        # Calculate mean and median for each metric
        aggregate = {}
        for metric_name, values in metrics.items():
            if values:
                aggregate[f"mean_{metric_name}"] = float(np.mean(values))
                aggregate[f"median_{metric_name}"] = float(np.median(values))
                aggregate[f"std_{metric_name}"] = float(np.std(values))
                aggregate[f"min_{metric_name}"] = float(np.min(values))
                aggregate[f"max_{metric_name}"] = float(np.max(values))

        return aggregate

    def get_metrics_by_category(self, tasks: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate metrics broken down by task category

        Args:
            tasks: List of task dicts with task_id and category

        Returns:
            Dict mapping category to metrics
        """
        # Build task_id -> category mapping
        task_categories = {task["task_id"]: task["category"] for task in tasks}

        # Group results by category
        category_results = defaultdict(list)
        for result in self.query_results:
            task_id = result["task_id"]
            category = task_categories.get(task_id, "unknown")
            category_results[category].append(result)

        # Calculate metrics for each category
        metrics_by_cat = {}
        for category, results in category_results.items():
            if not results:
                continue

            recall_10 = [r.get("recall@10", 0) for r in results]
            precision_10 = [r.get("precision@10", 0) for r in results]
            ndcg_10 = [r.get("ndcg@10", 0) for r in results]
            latencies = [r.get("latency_ms", 0) for r in results]

            metrics_by_cat[category] = {
                "num_queries": len(results),
                "mean_recall@10": float(np.mean(recall_10)),
                "mean_precision@10": float(np.mean(precision_10)),
                "mean_ndcg@10": float(np.mean(ndcg_10)),
                "mean_latency_ms": float(np.mean(latencies))
            }

        return metrics_by_cat

    def get_metrics_by_domain(self, tasks: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate metrics broken down by legal domain

        Args:
            tasks: List of task dicts with task_id and domain

        Returns:
            Dict mapping domain to metrics
        """
        # Build task_id -> domain mapping
        task_domains = {task["task_id"]: task.get("domain", "unknown") for task in tasks}

        # Group results by domain
        domain_results = defaultdict(list)
        for result in self.query_results:
            task_id = result["task_id"]
            domain = task_domains.get(task_id, "unknown")
            domain_results[domain].append(result)

        # Calculate metrics for each domain
        metrics_by_dom = {}
        for domain, results in domain_results.items():
            if not results:
                continue

            recall_10 = [r.get("recall@10", 0) for r in results]
            precision_10 = [r.get("precision@10", 0) for r in results]
            ndcg_10 = [r.get("ndcg@10", 0) for r in results]

            metrics_by_dom[domain] = {
                "num_queries": len(results),
                "mean_recall@10": float(np.mean(recall_10)),
                "mean_precision@10": float(np.mean(precision_10)),
                "mean_ndcg@10": float(np.mean(ndcg_10))
            }

        return metrics_by_dom

    def get_summary(self, tasks: List[Dict] = None) -> Dict:
        """
        Get complete summary of all metrics

        Args:
            tasks: Optional list of tasks for category/domain breakdowns

        Returns:
            Complete metrics summary
        """
        summary = {
            "total_queries": self.total_queries,
            "retrieval_metrics": self.get_retrieval_metrics(),
            "latency_metrics": self.get_latency_stats(),
            "throughput_qps": self.get_throughput(),
            "circuit_breaker_metrics": self.get_circuit_breaker_stats()
        }

        if tasks:
            summary["by_category"] = self.get_metrics_by_category(tasks)
            summary["by_domain"] = self.get_metrics_by_domain(tasks)

        return summary


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@K

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Cutoff rank

    Returns:
        Recall@K (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k & relevant_ids

    return len(relevant_retrieved) / len(relevant_ids)


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Precision@K

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Cutoff rank

    Returns:
        Precision@K (0.0 to 1.0)
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k & relevant_ids

    return len(relevant_retrieved) / k


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int, graded_relevance: Dict[str, int] = None) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain)

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Cutoff rank
        graded_relevance: Optional dict mapping doc_id to relevance grade (0-3)
                         If None, uses binary relevance (1 for relevant, 0 for not)

    Returns:
        NDCG@K (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if graded_relevance:
            relevance = graded_relevance.get(doc_id, 0)
        else:
            relevance = 1.0 if doc_id in relevant_ids else 0.0

        dcg += relevance / np.log2(i + 1)

    # Calculate Ideal DCG
    if graded_relevance:
        # Sort by relevance grade (descending)
        ideal_relevances = sorted(
            [graded_relevance.get(doc_id, 0) for doc_id in relevant_ids],
            reverse=True
        )
    else:
        # Binary relevance: all relevant docs have grade 1
        ideal_relevances = [1.0] * len(relevant_ids)

    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances[:k], start=1):
        idcg += relevance / np.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_map(retrieved_ids_list: List[List[str]], relevant_ids_list: List[Set[str]]) -> float:
    """
    Calculate MAP (Mean Average Precision)

    Args:
        retrieved_ids_list: List of retrieved doc ID lists (one per query)
        relevant_ids_list: List of relevant doc ID sets (one per query)

    Returns:
        MAP (0.0 to 1.0)
    """
    avg_precisions = []

    for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
        if not relevant_ids:
            continue

        num_relevant_found = 0
        precision_sum = 0.0

        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / i
                precision_sum += precision_at_i

        if num_relevant_found > 0:
            avg_precision = precision_sum / len(relevant_ids)
            avg_precisions.append(avg_precision)

    if not avg_precisions:
        return 0.0

    return float(np.mean(avg_precisions))
