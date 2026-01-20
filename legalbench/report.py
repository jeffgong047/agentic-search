"""
Evaluation Report Generator

Generates comprehensive evaluation reports with:
- summary.md: Markdown report with aggregate metrics
- results.json: Raw per-query results
- plots/: Visualizations (latency distribution, recall by category, etc.)
"""

import json
import numpy as np
from typing import Dict, List
from pathlib import Path
from datetime import datetime


class EvaluationReporter:
    """Generate evaluation reports and visualizations"""

    def __init__(self):
        """Initialize reporter"""
        pass

    def generate_report(self, results: Dict, output_dir: str):
        """
        Generate comprehensive evaluation report

        Args:
            results: Evaluation results dict from run_evaluation()
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[Report] Generating report in {output_dir}...")

        # 1. Save raw results JSON
        self._save_results_json(results, output_path / "results.json")

        # 2. Generate markdown summary
        summary_md = self._create_summary_markdown(results)
        with open(output_path / "summary.md", 'w') as f:
            f.write(summary_md)
        print(f"[Report] ✓ Created summary.md")

        # 3. Create plots directory
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Try to generate visualizations (optional, requires matplotlib)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns

            self._plot_latency_distribution(results, plots_dir / "latency_distribution.png")
            self._plot_recall_by_category(results, plots_dir / "recall_by_category.png")
            self._plot_precision_recall(results, plots_dir / "precision_recall.png")

            print(f"[Report] ✓ Created visualizations in plots/")

        except ImportError:
            print("[Report] ⚠ matplotlib not available, skipping plots")
            print("[Report]   Install with: pip install matplotlib seaborn")

        print(f"[Report] ✓ Report complete!")

    def _save_results_json(self, results: Dict, filepath: Path):
        """Save raw results as JSON"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[Report] ✓ Saved results.json")

    def _create_summary_markdown(self, results: Dict) -> str:
        """Create markdown summary with aggregate metrics"""
        summary = results.get("summary", {})
        config = results.get("config", {})

        md = f"""# LegalBench Evaluation Summary

**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Queries**: {summary.get('total_queries', 0)}
**Elasticsearch Index**: {config.get('es_index', 'unknown')}
**Embedding Model**: {config.get('embedding_model', 'unknown')}
**Top-K Retrieved**: 20

---

## Retrieval Metrics

| Metric       | Mean  | Median | Std Dev | Min   | Max   |
|--------------|-------|--------|---------|-------|-------|
"""

        # Add retrieval metrics
        retrieval = summary.get("retrieval_metrics", {})
        for k in [5, 10, 20]:
            recall_key = f"recall@{k}"
            if f"mean_{recall_key}" in retrieval:
                md += f"| Recall@{k}    | {retrieval[f'mean_{recall_key}']:.3f} | {retrieval.get(f'median_{recall_key}', 0):.3f} | {retrieval.get(f'std_{recall_key}', 0):.3f} | {retrieval.get(f'min_{recall_key}', 0):.3f} | {retrieval.get(f'max_{recall_key}', 0):.3f} |\n"

        for k in [5, 10, 20]:
            prec_key = f"precision@{k}"
            if f"mean_{prec_key}" in retrieval:
                md += f"| Precision@{k} | {retrieval[f'mean_{prec_key}']:.3f} | {retrieval.get(f'median_{prec_key}', 0):.3f} | {retrieval.get(f'std_{prec_key}', 0):.3f} | {retrieval.get(f'min_{prec_key}', 0):.3f} | {retrieval.get(f'max_{prec_key}', 0):.3f} |\n"

        if "mean_ndcg@10" in retrieval:
            md += f"| NDCG@10      | {retrieval['mean_ndcg@10']:.3f} | {retrieval.get('median_ndcg@10', 0):.3f} | {retrieval.get('std_ndcg@10', 0):.3f} | {retrieval.get('min_ndcg@10', 0):.3f} | {retrieval.get('max_ndcg@10', 0):.3f} |\n"

        md += "\n---\n\n## Latency Metrics\n\n"
        md += "| Metric    | Value (ms) |\n"
        md += "|-----------|------------|\n"

        latency = summary.get("latency_metrics", {})
        for metric in ["mean_ms", "median_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"]:
            if metric in latency:
                label = metric.replace("_ms", "").replace("_", " ").title()
                md += f"| {label:9s} | {latency[metric]:10.1f} |\n"

        throughput = summary.get("throughput_qps", 0)
        md += f"\n**Throughput**: {throughput:.2f} queries/second\n"

        # Circuit breaker stats
        cb = summary.get("circuit_breaker_metrics", {})
        if cb and cb.get("early_stop_rate", 0) > 0:
            md += "\n---\n\n## Circuit Breaker Efficiency\n\n"
            md += "| Metric           | Value |\n"
            md += "|------------------|-------|\n"
            md += f"| Early Stop Rate  | {cb['early_stop_rate']:.1%} |\n"
            md += f"| Mean Novelty     | {cb.get('mean_novelty', 0):.3f} |\n"
            md += f"| Median Novelty   | {cb.get('median_novelty', 0):.3f} |\n"

        # Breakdown by category
        if "by_category" in summary:
            md += "\n---\n\n## Breakdown by Task Category\n\n"
            md += "| Category                  | Queries | Recall@10 | Precision@10 | NDCG@10 | Avg Latency (ms) |\n"
            md += "|---------------------------|---------|-----------|--------------|---------|------------------|\n"

            for cat, metrics in sorted(summary["by_category"].items()):
                md += f"| {cat:25s} | {metrics['num_queries']:7d} | {metrics.get('mean_recall@10', 0):9.3f} | {metrics.get('mean_precision@10', 0):12.3f} | {metrics.get('mean_ndcg@10', 0):7.3f} | {metrics.get('mean_latency_ms', 0):16.1f} |\n"

        # Breakdown by domain
        if "by_domain" in summary:
            md += "\n---\n\n## Breakdown by Legal Domain\n\n"
            md += "| Domain                    | Queries | Recall@10 | Precision@10 | NDCG@10 |\n"
            md += "|---------------------------|---------|-----------|--------------|------| ---|\n"

            for domain, metrics in sorted(summary["by_domain"].items()):
                md += f"| {domain:25s} | {metrics['num_queries']:7d} | {metrics.get('mean_recall@10', 0):9.3f} | {metrics.get('mean_precision@10', 0):12.3f} | {metrics.get('mean_ndcg@10', 0):7.3f} |\n"

        md += "\n---\n\n## Configuration\n\n"
        md += f"- **Elasticsearch URL**: {config.get('es_url', 'unknown')}\n"
        md += f"- **Index Name**: {config.get('es_index', 'unknown')}\n"
        md += f"- **Embedding Model**: {config.get('embedding_model', 'unknown')}\n"
        md += f"- **Tasks Path**: {config.get('tasks_path', 'unknown')}\n"
        md += f"- **Qrels Path**: {config.get('qrels_path', 'unknown')}\n"

        md += "\n---\n\n"
        md += f"*Generated by LegalBench Evaluation Pipeline on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        return md

    def _plot_latency_distribution(self, results: Dict, output_path: Path):
        """Plot latency histogram and CDF"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        per_query = results.get("per_query_results", [])
        latencies = [r["latency_ms"] for r in per_query]

        if not latencies:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        ax1.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.1f} ms')
        ax1.axvline(np.median(latencies), color='blue', linestyle='--', label=f'Median: {np.median(latencies):.1f} ms')
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Latency Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # CDF
        sorted_latencies = np.sort(latencies)
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        ax2.plot(sorted_latencies, cdf, linewidth=2)
        ax2.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='P95')
        ax2.axhline(0.99, color='orange', linestyle='--', alpha=0.5, label='P99')
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Latency CDF')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_recall_by_category(self, results: Dict, output_path: Path):
        """Plot Recall@10 grouped by task category"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        summary = results.get("summary", {})
        by_category = summary.get("by_category", {})

        if not by_category:
            return

        categories = list(by_category.keys())
        recall_10 = [by_category[cat]["mean_recall@10"] for cat in categories]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(categories, recall_10, color='steelblue', edgecolor='black')

        # Add value labels on bars
        for i, (cat, val) in enumerate(zip(categories, recall_10)):
            plt.text(val + 0.01, i, f'{val:.3f}', va='center')

        plt.xlabel('Mean Recall@10')
        plt.title('Recall@10 by Task Category')
        plt.xlim(0, 1.0)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_precision_recall(self, results: Dict, output_path: Path):
        """Plot precision-recall relationship"""
        import matplotlib.pyplot as plt

        per_query = results.get("per_query_results", [])

        if not per_query:
            return

        # Extract recall and precision at different K values
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, k in enumerate([5, 10, 20]):
            recall_key = f"recall@{k}"
            prec_key = f"precision@{k}"

            recalls = [r.get(recall_key, 0) for r in per_query if recall_key in r]
            precisions = [r.get(prec_key, 0) for r in per_query if prec_key in r]

            if recalls and precisions:
                axes[idx].scatter(recalls, precisions, alpha=0.5, s=20)
                axes[idx].set_xlabel(f'Recall@{k}')
                axes[idx].set_ylabel(f'Precision@{k}')
                axes[idx].set_title(f'Precision vs Recall @{k}')
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)
                axes[idx].grid(alpha=0.3)

                # Add diagonal line (perfect correlation)
                axes[idx].plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Perfect correlation')
                axes[idx].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Test with sample results
    sample_results = {
        "config": {
            "es_url": "http://localhost:9200",
            "es_index": "legalbench_documents",
            "embedding_model": "BAAI/bge-small-en-v1.5"
        },
        "summary": {
            "total_queries": 162,
            "retrieval_metrics": {
                "mean_recall@10": 0.687,
                "mean_precision@10": 0.521,
                "mean_ndcg@10": 0.642
            },
            "latency_metrics": {
                "mean_ms": 142.3,
                "median_ms": 128.5,
                "p95_ms": 256.8,
                "p99_ms": 312.1
            },
            "throughput_qps": 7.02
        },
        "per_query_results": []
    }

    reporter = EvaluationReporter()
    reporter.generate_report(sample_results, "legalbench/results_test")
