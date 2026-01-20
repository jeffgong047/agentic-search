"""
LegalBench Evaluation Entry Point

Complete pipeline for benchmarking agentic search performance.

Usage:
    # Full pipeline (download → build → evaluate)
    python -m legalbench all

    # Individual steps
    python -m legalbench download
    python -m legalbench build-index
    python -m legalbench evaluate

    # With options
    python -m legalbench evaluate --max-queries 50 --category issue_spotting
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legalbench.download_tasks import LegalBenchDownloader
from legalbench.load_corpus import LegalCorpusManager
from legalbench.create_qrels import QrelsGenerator
from legalbench.build_index import LegalBenchIndexBuilder
from legalbench.run_evaluation import LegalBenchEvaluator
from legalbench.report import EvaluationReporter


def download_command(args):
    """Download tasks and generate corpus"""
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING DATA")
    print("="*70 + "\n")

    # Download tasks
    print("[1/3] Downloading/generating LegalBench tasks...")
    downloader = LegalBenchDownloader()
    tasks = downloader.download_tasks(
        output_dir=args.data_dir,
        source=args.source
    )

    # Print task statistics
    stats = downloader.get_task_statistics(tasks)
    print(f"\n✓ Generated {stats['total_tasks']} tasks")
    print(f"  By category: {len(stats['by_category'])} categories")
    print(f"  By domain: {len(stats['by_domain'])} domains")

    # Generate corpus
    print(f"\n[2/3] Generating legal document corpus ({args.num_docs} documents)...")
    corpus_manager = LegalCorpusManager()
    corpus = corpus_manager.load_or_generate_corpus(
        source="synthetic",
        num_docs=args.num_docs,
        output_dir=args.data_dir
    )

    # Print corpus statistics
    corpus_stats = corpus_manager.get_corpus_statistics(corpus)
    print(f"\n✓ Generated {corpus_stats['total_docs']} documents")
    print(f"  Average length: {corpus_stats['avg_length']:.0f} characters")
    print(f"  By type: {len(corpus_stats['by_doc_type'])} document types")

    # Generate qrels
    print(f"\n[3/3] Generating relevance judgments (qrels)...")
    qrels_generator = QrelsGenerator(embedding_model=args.embedding_model)
    qrels = qrels_generator.generate_qrels(
        tasks_path=f"{args.data_dir}/tasks.jsonl",
        corpus_path=f"{args.data_dir}/corpus.jsonl",
        output_path=f"{args.data_dir}/qrels.tsv",
        method=args.qrels_method,
        top_k=args.qrels_top_k,
        relevance_threshold=args.qrels_threshold
    )

    # Print qrels statistics
    qrels_stats = qrels_generator.get_qrels_statistics(qrels)
    print(f"\n✓ Generated {qrels_stats['total_judgments']} relevance judgments")
    print(f"  Tasks with qrels: {qrels_stats['total_tasks']}")
    print(f"  Avg relevant docs/task: {qrels_stats['avg_relevant_per_task']:.1f}")

    print("\n" + "="*70)
    print("DATA DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\nFiles created in {args.data_dir}/:")
    print("  - tasks.jsonl      (162 legal reasoning queries)")
    print(f"  - corpus.jsonl     ({corpus_stats['total_docs']} legal documents)")
    print(f"  - qrels.tsv        ({qrels_stats['total_judgments']} relevance judgments)")
    print("\nNext: python -m legalbench build-index")


def build_index_command(args):
    """Build semantic index"""
    print("\n" + "="*70)
    print("STEP 2: BUILDING SEMANTIC INDEX")
    print("="*70 + "\n")

    builder = LegalBenchIndexBuilder(
        es_url=args.es_url,
        embedding_model=args.embedding_model
    )

    # Get embedding model info
    model_info = builder.get_embedding_model_info()
    print(f"Embedding Model: {model_info.get('model_name')}")
    print(f"Dimensions: {model_info.get('dimensions', 'unknown')}")
    print(f"Max seq length: {model_info.get('max_seq_length', 'unknown')}\n")

    # Build index
    indexed_count = builder.build_index(
        corpus_path=f"{args.data_dir}/corpus.jsonl",
        index_name=args.index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_rebuild=args.force_rebuild
    )

    if indexed_count > 0:
        # Verify index
        print("\nVerifying index...")
        stats = builder.verify_index(args.index)

        print("\n" + "="*70)
        print("INDEX BUILD COMPLETE!")
        print("="*70)
        print(f"\nIndex: {args.index}")
        print(f"Documents: {stats.get('doc_count', 0)}")
        print(f"Size: {stats.get('size_bytes', 0) / 1024 / 1024:.2f} MB")
        print("\nNext: python -m legalbench evaluate")
    else:
        print("\n⚠ Index already exists. Use --force-rebuild to recreate.")


def evaluate_command(args):
    """Run evaluation"""
    print("\n" + "="*70)
    print("STEP 3: RUNNING EVALUATION")
    print("="*70 + "\n")

    # Create config
    config = {
        "es_url": args.es_url,
        "es_index": args.index,
        "tasks_path": f"{args.data_dir}/tasks.jsonl",
        "qrels_path": f"{args.data_dir}/qrels.tsv",
        "embedding_model": args.embedding_model
    }

    # Run evaluation
    evaluator = LegalBenchEvaluator(config)
    results = evaluator.run_evaluation(
        max_queries=args.max_queries,
        top_k=args.top_k,
        filter_category=args.category
    )

    # Generate report
    print("\nGenerating report...")
    reporter = EvaluationReporter()
    reporter.generate_report(results, args.output)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output}/")
    print("  - results.json     (raw per-query results)")
    print("  - summary.md       (aggregate metrics report)")
    print("  - plots/           (visualizations)")
    print(f"\nView report: cat {args.output}/summary.md")


def all_command(args):
    """Run complete pipeline"""
    print("\n" + "="*70)
    print("LEGALBENCH COMPLETE PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Download tasks and generate corpus")
    print("  2. Build semantic index in Elasticsearch")
    print("  3. Run evaluation and generate report")
    print()

    # Step 1: Download
    download_command(args)

    # Step 2: Build index
    build_index_command(args)

    # Step 3: Evaluate
    evaluate_command(args)

    print("\n" + "="*70)
    print("COMPLETE PIPELINE FINISHED!")
    print("="*70)
    print(f"\nResults: {args.output}/summary.md")


def main():
    parser = argparse.ArgumentParser(
        description="LegalBench Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python -m legalbench all

  # Run individual steps
  python -m legalbench download
  python -m legalbench build-index
  python -m legalbench evaluate

  # Evaluate subset
  python -m legalbench evaluate --max-queries 50 --category issue_spotting

  # Use different embedding model
  python -m legalbench build-index --embedding-model sentence-transformers/all-MiniLM-L6-v2
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--data-dir", default="legalbench/data", help="Data directory")
    parent_parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    parent_parser.add_argument("--index", default="legalbench_documents", help="Index name")
    parent_parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model")

    # Download command
    download_parser = subparsers.add_parser("download", parents=[parent_parser], help="Download tasks and corpus")
    download_parser.add_argument("--source", default="synthetic", choices=["synthetic", "huggingface", "github"], help="Task source")
    download_parser.add_argument("--num-docs", type=int, default=500, help="Number of corpus documents")
    download_parser.add_argument("--qrels-method", default="pseudo", choices=["pseudo", "llm", "manual"], help="Qrels generation method")
    download_parser.add_argument("--qrels-top-k", type=int, default=50, help="Top-K docs to judge per query")
    download_parser.add_argument("--qrels-threshold", type=float, default=0.5, help="Relevance threshold (pseudo method)")
    download_parser.set_defaults(func=download_command)

    # Build index command
    build_parser = subparsers.add_parser("build-index", parents=[parent_parser], help="Build semantic index")
    build_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    build_parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    build_parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild index")
    build_parser.set_defaults(func=build_index_command)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", parents=[parent_parser], help="Run evaluation")
    eval_parser.add_argument("--output", default="legalbench/results", help="Output directory")
    eval_parser.add_argument("--max-queries", type=int, help="Limit number of queries")
    eval_parser.add_argument("--top-k", type=int, default=20, help="Number of docs to retrieve")
    eval_parser.add_argument("--category", help="Filter by task category")
    eval_parser.set_defaults(func=evaluate_command)

    # All command
    all_parser = subparsers.add_parser("all", parents=[parent_parser], help="Run complete pipeline")
    all_parser.add_argument("--num-docs", type=int, default=500, help="Number of corpus documents")
    all_parser.add_argument("--qrels-method", default="pseudo", help="Qrels generation method")
    all_parser.add_argument("--qrels-top-k", type=int, default=50, help="Top-K docs to judge per query")
    all_parser.add_argument("--qrels-threshold", type=float, default=0.5, help="Relevance threshold")
    all_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    all_parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    all_parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild index")
    all_parser.add_argument("--output", default="legalbench/results", help="Output directory")
    all_parser.add_argument("--max-queries", type=int, help="Limit number of queries")
    all_parser.add_argument("--top-k", type=int, default=20, help="Number of docs to retrieve")
    all_parser.add_argument("--source", default="synthetic", help="Task source")
    all_parser.set_defaults(func=all_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
