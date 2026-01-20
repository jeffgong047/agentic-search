"""
LegalBench Analysis Pipeline with Claude Opus 4.5
Data Science Analysis of Agentic Search Performance

Runs 50-100 queries and analyzes:
- Agent behavior patterns
- Memory evolution effectiveness
- Query refinement quality
- Circuit breaker efficiency
- Bottlenecks and failure modes
"""

import os
import json
import time
from typing import List, Dict, Any
from collections import defaultdict
import statistics
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import dspy
from core.data_structures import create_initial_state
from core.backends import create_best_backend
from core.backends_wrapper import BackendAgenticSearchEngine
from legalbench.load_corpus import LegalCorpusManager
from legalbench.download_tasks import LegalBenchDownloader


class LegalBenchAnalyzer:
    """Data scientist analyzer for agentic search evaluation"""

    def __init__(self, output_dir: str = "legalbench/analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Metrics storage
        self.results = []
        self.traces = []
        self.errors = []

        # Aggregate stats
        self.stats = {
            "iterations_per_query": [],
            "novelty_scores": [],
            "retrieval_counts": [],
            "latencies_ms": [],
            "memory_growth": [],
            "circuit_breaker_triggers": 0,
            "failed_queries": 0
        }

        # Pattern analysis
        self.patterns = {
            "high_iteration_queries": [],
            "zero_result_queries": [],
            "memory_heavy_queries": [],
            "fast_converge_queries": []
        }

    def setup_claude_opus(self):
        """Configure DSPy with Claude Opus 4.5"""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        print("🤖 Configuring Claude Opus 4.5...")
        lm = dspy.LM(
            model="anthropic/claude-opus-4-5-20251101",
            api_key=api_key,
            max_tokens=2000
        )
        dspy.settings.configure(lm=lm)
        print("✓ Claude Opus 4.5 configured\n")

    def generate_test_queries(self, num_queries: int = 50) -> List[Dict]:
        """Generate LegalBench queries"""
        print(f"📋 Generating {num_queries} LegalBench queries...")

        task_downloader = LegalBenchDownloader()
        tasks = task_downloader.download_tasks(output_dir="legalbench/data", source="synthetic")

        # Sample to get exact number
        import random
        tasks = random.sample(tasks, min(num_queries, len(tasks)))

        print(f"✓ Generated {len(tasks)} queries across {len(set(t['category'] for t in tasks))} categories\n")
        return tasks

    def generate_corpus(self, num_docs: int = 100) -> List[Dict]:
        """Generate document corpus"""
        print(f"📚 Generating corpus with {num_docs} documents...")

        corpus_manager = LegalCorpusManager()
        docs = corpus_manager.load_or_generate_corpus(source="synthetic", num_docs=num_docs)

        print(f"✓ Generated {len(docs)} documents\n")
        return docs

    def run_single_query(self, engine: BackendAgenticSearchEngine, task: Dict, query_id: int) -> Dict:
        """Run single query and capture detailed trace"""
        query = task["query"]
        start_time = time.time()

        try:
            # Execute search
            final_state = engine.search(query)

            latency_ms = (time.time() - start_time) * 1000

            # Extract metrics
            result = {
                "query_id": query_id,
                "query": query,
                "category": task.get("category"),
                "domain": task.get("domain"),
                "difficulty": task.get("difficulty"),
                "success": True,
                "iterations": final_state["step_count"],
                "novelty_score": final_state["novelty_score"],
                "num_results": len(final_state["retrieved_docs"]),
                "latency_ms": latency_ms,
                "memory_facts": len(final_state["verified_facts"]),
                "negative_cache_size": len(final_state["negative_cache"]),
                "circuit_breaker_fired": not final_state["should_continue"],
                "search_plan": {
                    "intent": final_state["search_plan"].primary_intent if final_state["search_plan"] else None,
                    "num_strategies": len(final_state["search_plan"].strategies) if final_state["search_plan"] else 0
                }
            }

            # Capture detailed trace for first 10 queries
            if query_id < 10:
                result["full_trace"] = {
                    "verified_facts": final_state["verified_facts"],
                    "negative_cache": final_state["negative_cache"],
                    "result_ids": [r.id for r in final_state["retrieved_docs"][:5]]
                }

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"❌ Query {query_id} failed: {str(e)}")

            return {
                "query_id": query_id,
                "query": query,
                "category": task.get("category"),
                "success": False,
                "error": str(e),
                "latency_ms": latency_ms
            }

    def run_evaluation(self, num_queries: int = 50, num_docs: int = 100):
        """Run full evaluation pipeline"""
        print("="*80)
        print(" LEGALBENCH EVALUATION WITH CLAUDE OPUS 4.5".center(80))
        print("="*80)
        print()

        # Setup
        self.setup_claude_opus()
        tasks = self.generate_test_queries(num_queries)
        docs = self.generate_corpus(num_docs)

        # Initialize backend and engine
        print("🔧 Initializing search backend...")
        es_config = {
            "host": "legalbench_elasticsearch",
            "port": 9200,
            "index": "legalbench_evaluation"
        }
        backend = create_best_backend(es_config=es_config)
        backend.index_documents(docs)
        engine = BackendAgenticSearchEngine(backend)
        print(f"✓ Indexed {len(docs)} documents\n")

        # Run queries
        print(f"🚀 Running {len(tasks)} queries...\n")
        print("-"*80)

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Query: {task['query'][:60]}...")

            result = self.run_single_query(engine, task, i)
            self.results.append(result)

            if result["success"]:
                print(f"  ✓ Iterations: {result['iterations']} | "
                      f"Results: {result['num_results']} | "
                      f"Latency: {result['latency_ms']:.0f}ms | "
                      f"Novelty: {result['novelty_score']:.3f}")

                # Update stats
                self.stats["iterations_per_query"].append(result["iterations"])
                self.stats["novelty_scores"].append(result["novelty_score"])
                self.stats["retrieval_counts"].append(result["num_results"])
                self.stats["latencies_ms"].append(result["latency_ms"])
                self.stats["memory_growth"].append(result["memory_facts"])

                if result["circuit_breaker_fired"]:
                    self.stats["circuit_breaker_triggers"] += 1

                # Pattern detection
                if result["iterations"] >= 3:
                    self.patterns["high_iteration_queries"].append(i)
                if result["num_results"] == 0:
                    self.patterns["zero_result_queries"].append(i)
                if result["memory_facts"] >= 5:
                    self.patterns["memory_heavy_queries"].append(i)
                if result["iterations"] == 1 and result["novelty_score"] < 0.3:
                    self.patterns["fast_converge_queries"].append(i)
            else:
                print(f"  ❌ Error: {result['error']}")
                self.stats["failed_queries"] += 1

            # Progress update every 10 queries
            if i % 10 == 0:
                print(f"\n{'─'*80}")
                print(f"Progress: {i}/{len(tasks)} queries completed")
                if self.stats["iterations_per_query"]:
                    avg_iter = statistics.mean(self.stats["iterations_per_query"])
                    avg_latency = statistics.mean(self.stats["latencies_ms"])
                    print(f"Running avg: {avg_iter:.2f} iterations, {avg_latency:.0f}ms latency")
                print(f"{'─'*80}")

        print("\n" + "="*80)
        print(" EVALUATION COMPLETE".center(80))
        print("="*80 + "\n")

    def analyze_results(self):
        """Data scientist analysis of results"""
        print("\n" + "="*80)
        print(" DATA SCIENCE ANALYSIS".center(80))
        print("="*80 + "\n")

        successful = [r for r in self.results if r["success"]]
        total = len(self.results)

        # === 1. OVERALL PERFORMANCE ===
        print("📊 1. OVERALL PERFORMANCE")
        print("-"*80)
        print(f"Total Queries: {total}")
        print(f"Successful: {len(successful)} ({len(successful)/total*100:.1f}%)")
        print(f"Failed: {self.stats['failed_queries']} ({self.stats['failed_queries']/total*100:.1f}%)")
        print()

        if not successful:
            print("❌ No successful queries to analyze.\n")
            return

        # === 2. ITERATION ANALYSIS ===
        print("📊 2. ITERATION ANALYSIS (Agentic Loop Behavior)")
        print("-"*80)
        iters = self.stats["iterations_per_query"]
        print(f"Average Iterations: {statistics.mean(iters):.2f}")
        print(f"Median Iterations: {statistics.median(iters):.1f}")
        print(f"Min/Max: {min(iters)} / {max(iters)}")
        print(f"\nIteration Distribution:")
        iter_dist = defaultdict(int)
        for i in iters:
            iter_dist[i] += 1
        for i in sorted(iter_dist.keys()):
            pct = iter_dist[i] / len(iters) * 100
            bar = "█" * int(pct / 2)
            print(f"  {i} iterations: {iter_dist[i]:3d} queries ({pct:5.1f}%) {bar}")

        print(f"\n💡 INSIGHT: ", end="")
        if statistics.mean(iters) > 2.5:
            print("⚠️  HIGH - Agent is doing extensive refinement (may be overworking)")
        elif statistics.mean(iters) < 1.5:
            print("⚠️  LOW - Agent converges too quickly (may miss information)")
        else:
            print("✅ OPTIMAL - Balanced iteration count")
        print()

        # === 3. NOVELTY & CIRCUIT BREAKER ===
        print("📊 3. NOVELTY & CIRCUIT BREAKER ANALYSIS")
        print("-"*80)
        novelty = self.stats["novelty_scores"]
        print(f"Average Final Novelty: {statistics.mean(novelty):.3f}")
        print(f"Median Final Novelty: {statistics.median(novelty):.3f}")
        print(f"Circuit Breaker Fired: {self.stats['circuit_breaker_triggers']}/{len(successful)} "
              f"({self.stats['circuit_breaker_triggers']/len(successful)*100:.1f}%)")

        print(f"\n💡 INSIGHT: ", end="")
        avg_novelty = statistics.mean(novelty)
        if avg_novelty > 0.4:
            print("⚠️  Agent often stops with high novelty (may stop too early)")
        elif avg_novelty < 0.1:
            print("✅ Good convergence (low novelty at termination)")
        else:
            print("✅ Reasonable novelty threshold")
        print()

        # === 4. RETRIEVAL QUALITY ===
        print("📊 4. RETRIEVAL QUALITY")
        print("-"*80)
        results_counts = self.stats["retrieval_counts"]
        print(f"Average Results: {statistics.mean(results_counts):.1f}")
        print(f"Median Results: {statistics.median(results_counts):.0f}")
        print(f"Zero-result queries: {len(self.patterns['zero_result_queries'])} "
              f"({len(self.patterns['zero_result_queries'])/total*100:.1f}%)")

        print(f"\n💡 INSIGHT: ", end="")
        if len(self.patterns['zero_result_queries']) > total * 0.1:
            print("⚠️  HIGH zero-result rate (corpus too small or queries too specific)")
        elif statistics.mean(results_counts) > 15:
            print("⚠️  Too many results per query (may need better filtering)")
        else:
            print("✅ Reasonable retrieval counts")
        print()

        # === 5. LATENCY ANALYSIS ===
        print("📊 5. LATENCY ANALYSIS")
        print("-"*80)
        latencies = self.stats["latencies_ms"]
        print(f"Average Latency: {statistics.mean(latencies):.0f}ms")
        print(f"Median Latency: {statistics.median(latencies):.0f}ms")
        print(f"P95 Latency: {sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms")
        print(f"P99 Latency: {sorted(latencies)[int(len(latencies)*0.99)]:.0f}ms")

        # Latency by iteration count
        print(f"\nLatency by Iteration Count:")
        iter_latency = defaultdict(list)
        for r in successful:
            iter_latency[r["iterations"]].append(r["latency_ms"])
        for i in sorted(iter_latency.keys()):
            avg_lat = statistics.mean(iter_latency[i])
            print(f"  {i} iterations: {avg_lat:.0f}ms avg")

        print(f"\n💡 INSIGHT: ", end="")
        if statistics.mean(latencies) > 10000:
            print("⚠️  HIGH latency (>10s avg) - Claude Opus is slow, consider caching")
        elif statistics.mean(latencies) > 5000:
            print("✅ Moderate latency (~5-10s) - Acceptable for complex queries")
        else:
            print("✅ Good latency (<5s avg)")
        print()

        # === 6. MEMORY EVOLUTION ===
        print("📊 6. MEMORY EVOLUTION")
        print("-"*80)
        mem_growth = self.stats["memory_growth"]
        print(f"Average Memory Facts: {statistics.mean(mem_growth):.1f}")
        print(f"Median Memory Facts: {statistics.median(mem_growth):.0f}")
        print(f"Queries with 5+ facts: {len(self.patterns['memory_heavy_queries'])} "
              f"({len(self.patterns['memory_heavy_queries'])/total*100:.1f}%)")

        print(f"\n💡 INSIGHT: ", end="")
        if statistics.mean(mem_growth) < 1:
            print("⚠️  LOW memory usage (agent not learning from iterations)")
        elif statistics.mean(mem_growth) > 5:
            print("⚠️  HIGH memory growth (may indicate thrashing)")
        else:
            print("✅ Moderate memory evolution")
        print()

        # === 7. CATEGORY BREAKDOWN ===
        print("📊 7. PERFORMANCE BY CATEGORY")
        print("-"*80)
        by_category = defaultdict(list)
        for r in successful:
            if r.get("category"):
                by_category[r["category"]].append(r)

        for cat in sorted(by_category.keys()):
            results = by_category[cat]
            avg_iter = statistics.mean([r["iterations"] for r in results])
            avg_lat = statistics.mean([r["latency_ms"] for r in results])
            avg_results = statistics.mean([r["num_results"] for r in results])
            print(f"\n{cat}:")
            print(f"  Queries: {len(results)}")
            print(f"  Avg Iterations: {avg_iter:.2f}")
            print(f"  Avg Results: {avg_results:.1f}")
            print(f"  Avg Latency: {avg_lat:.0f}ms")
        print()

        # === 8. PROBLEM QUERIES ===
        print("📊 8. PROBLEM QUERIES (For Debugging)")
        print("-"*80)

        # High iteration queries
        if self.patterns["high_iteration_queries"]:
            print(f"\n⚠️  High Iteration Queries ({len(self.patterns['high_iteration_queries'])} queries):")
            for qid in self.patterns["high_iteration_queries"][:5]:
                r = self.results[qid-1]
                print(f"  [{qid}] {r['query'][:60]}... ({r['iterations']} iters)")

        # Zero result queries
        if self.patterns["zero_result_queries"]:
            print(f"\n⚠️  Zero-Result Queries ({len(self.patterns['zero_result_queries'])} queries):")
            for qid in self.patterns["zero_result_queries"][:5]:
                r = self.results[qid-1]
                print(f"  [{qid}] {r['query'][:60]}...")

        # Fast converge queries (good!)
        if self.patterns["fast_converge_queries"]:
            print(f"\n✅ Fast Convergence Queries ({len(self.patterns['fast_converge_queries'])} queries):")
            for qid in self.patterns["fast_converge_queries"][:3]:
                r = self.results[qid-1]
                print(f"  [{qid}] {r['query'][:60]}... (1 iter, {r['novelty_score']:.3f} novelty)")
        print()

        # === 9. KEY FINDINGS ===
        print("="*80)
        print(" 🔍 KEY FINDINGS & RECOMMENDATIONS".center(80))
        print("="*80 + "\n")

        findings = []

        # Success rate
        success_rate = len(successful) / total
        if success_rate < 0.9:
            findings.append(f"⚠️  SUCCESS RATE: {success_rate*100:.1f}% - Investigate failures")
        else:
            findings.append(f"✅ SUCCESS RATE: {success_rate*100:.1f}% - Good")

        # Iteration efficiency
        avg_iter = statistics.mean(iters)
        if avg_iter > 2.5:
            findings.append(f"⚠️  ITERATIONS: Avg {avg_iter:.2f} - Agent may be overworking. "
                          f"Consider:\n   - Increasing novelty epsilon (current: likely 0.3)\n"
                          f"   - Reducing max iterations\n"
                          f"   - Better initial query formulation")
        elif avg_iter < 1.5:
            findings.append(f"⚠️  ITERATIONS: Avg {avg_iter:.2f} - May stop too early. "
                          f"Consider lowering novelty threshold")
        else:
            findings.append(f"✅ ITERATIONS: Avg {avg_iter:.2f} - Well balanced")

        # Latency
        avg_lat = statistics.mean(latencies)
        if avg_lat > 10000:
            findings.append(f"⚠️  LATENCY: {avg_lat/1000:.1f}s avg - Claude Opus is slow. "
                          f"Recommend:\n   - Implement caching for repeated queries\n"
                          f"   - Consider Haiku for non-critical queries\n"
                          f"   - Batch processing for evaluation")

        # Zero results
        zero_pct = len(self.patterns['zero_result_queries']) / total * 100
        if zero_pct > 10:
            findings.append(f"⚠️  ZERO RESULTS: {zero_pct:.1f}% - Corpus may be too small or "
                          f"queries too specific")

        # Memory usage
        if statistics.mean(mem_growth) < 1:
            findings.append(f"⚠️  MEMORY: Low usage ({statistics.mean(mem_growth):.1f} facts avg) - "
                          f"Memory evolution not effective")

        for finding in findings:
            print(finding)
            print()

        print("="*80 + "\n")

    def save_results(self):
        """Save detailed results to files"""
        print("💾 Saving results...")

        # Save full results
        with open(f"{self.output_dir}/full_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        # Save statistics
        stats_summary = {
            "total_queries": len(self.results),
            "successful": sum(1 for r in self.results if r["success"]),
            "failed": self.stats["failed_queries"],
            "avg_iterations": statistics.mean(self.stats["iterations_per_query"]) if self.stats["iterations_per_query"] else 0,
            "avg_latency_ms": statistics.mean(self.stats["latencies_ms"]) if self.stats["latencies_ms"] else 0,
            "avg_results": statistics.mean(self.stats["retrieval_counts"]) if self.stats["retrieval_counts"] else 0,
            "circuit_breaker_rate": self.stats["circuit_breaker_triggers"] / len([r for r in self.results if r["success"]]) if self.results else 0,
            "patterns": {k: len(v) for k, v in self.patterns.items()}
        }

        with open(f"{self.output_dir}/statistics.json", "w") as f:
            json.dump(stats_summary, f, indent=2)

        # Save traces for first 10 queries
        traces = [r for r in self.results if "full_trace" in r]
        if traces:
            with open(f"{self.output_dir}/sample_traces.json", "w") as f:
                json.dump(traces, f, indent=2)

        print(f"✓ Results saved to {self.output_dir}/")
        print(f"  - full_results.json")
        print(f"  - statistics.json")
        if traces:
            print(f"  - sample_traces.json ({len(traces)} detailed traces)")
        print()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="LegalBench Analysis with Claude Opus")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries to run")
    parser.add_argument("--docs", type=int, default=100, help="Number of documents in corpus")
    parser.add_argument("--output", type=str, default="legalbench/analysis", help="Output directory")

    args = parser.parse_args()

    analyzer = LegalBenchAnalyzer(output_dir=args.output)

    try:
        analyzer.run_evaluation(num_queries=args.queries, num_docs=args.docs)
        analyzer.analyze_results()
        analyzer.save_results()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        if analyzer.results:
            print("Analyzing partial results...\n")
            analyzer.analyze_results()
            analyzer.save_results()

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
